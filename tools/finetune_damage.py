"""Fine-tune the last K DINOv2 blocks + heads of the damage model on cached
prefix features, and evaluate baseline vs candidate with one shared protocol.

Evaluation protocol (identical for both models, computed on cached 784px
features, no tiling/multi-scale): per image, ``score`` is the highest
any-damage threshold at which a connected region >= 0.1% of the image survives.
Damage-vs-clean separation (AUROC, recall at fixed clean false-alert rate) is
reported per split, plus the calibration-chosen operating point (97% damage
recall on the calibration split) applied to validation and test.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).parent))
from damage_cache import UP, suffix_features  # noqa: E402
from vehicle_damage.model import damage_probabilities, load_checkpoint  # noqa: E402

LOSS_CONFIG = dict(
    focal_gamma=2.0, tversky_alpha=0.3, tversky_beta=0.7, focal_weight=1.0, tversky_weight=1.0,
    type_weight=2.0, exterior_weight=0.3, hard_negative_weight=5.0,
    class_weights=[0.1, 1.5, 10.0, 1.0, 12.0, 3.0],
)
LEVELS = np.linspace(0.05, 0.99, 48)
EXTERIOR_FLOOR = 0.5
EIGHT = np.ones((3, 3))
DEPLOYED_THRESHOLD = 0.7356525659561157  # sweep point 0.75 is the nearest grid value


def forward(model, item, n_blocks, flip=False):
    tokens = item["tokens"].float()[None]
    grid = item["grid"]
    if flip:  # feature-map flip: approximate augmentation (ViT features ~ flip-equivariant)
        prefix = model.backbone.num_prefix_tokens
        patches = tokens[:, prefix:].reshape(1, *grid, -1).flip(2).reshape(1, grid[0] * grid[1], -1)
        tokens = torch.cat([tokens[:, :prefix], patches], dim=1)
    feats = suffix_features(model.backbone, tokens, n_blocks, grid)
    out = {}
    for name, head in (("presence", model.presence_head), ("type", model.type_head), ("exterior", model.exterior_head)):
        out[name] = F.interpolate(head(feats), scale_factor=UP, mode="bilinear", align_corners=False)
    return out


@torch.no_grad()
def item_maps(model, item, n_blocks):
    model.eval()
    out = forward(model, item, n_blocks)
    probs = damage_probabilities(out)[0]
    ext = out["exterior"].sigmoid()[0, 0]
    any_damage = (1.0 - probs[0]) * (EXTERIOR_FLOOR + (1 - EXTERIOR_FLOOR) * ext)
    return any_damage.numpy(), probs[1:].argmax(0).numpy() + 1


def case_score(any_damage: np.ndarray) -> float:
    """Highest threshold at which a region >= 0.1% of the image still exists."""
    minpx = max(8, int(0.001 * any_damage.size))

    def survives(i: int) -> bool:
        labels, count = ndimage.label(any_damage >= LEVELS[i], structure=EIGHT)
        return count > 0 and np.bincount(labels.ravel())[1:].max() >= minpx

    if not survives(0):
        return 0.0
    lo, hi = 0, len(LEVELS) - 1
    while lo < hi:  # survival is monotone in the threshold -> binary search
        mid = (lo + hi + 1) // 2
        lo, hi = (mid, hi) if survives(mid) else (lo, mid - 1)
    return float(LEVELS[lo])


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float((pos[:, None] > neg[None]).mean() + 0.5 * (pos[:, None] == neg[None]).mean())


def evaluate(model, split_items: dict[str, list], n_blocks: int, label: str) -> dict:
    scored: dict[str, dict] = {}
    for split, items in split_items.items():
        pos, neg, rows, wild = [], [], [], []
        for item in items:
            any_damage, type_map = item_maps(model, item, n_blocks)
            gt = item["damage"].numpy()
            score = case_score(any_damage)
            (pos if (gt > 0).any() else neg).append(score)
            if item["source"].startswith("Wikimedia"):
                wild.append(score)
            rows.append((any_damage, type_map, gt))
        scored[split] = {"pos": np.array(pos), "neg": np.array(neg), "rows": rows, "wild": np.array(wild)}
    cal_pos = scored["calibration"]["pos"]
    thr = float(np.quantile(cal_pos, 0.03))  # 97% damage-case recall on calibration
    report: dict = {"model": label, "operating_threshold_from_calibration": thr}
    for split, d in scored.items():
        pos, neg = d["pos"], d["neg"]
        row = {
            "damage_cases": len(pos), "clean_images": len(neg), "auroc": auroc(pos, neg),
            "recall_at_op": float((pos >= thr).mean()),
            "clean_false_alert_at_op": float((neg >= thr).mean()) if len(neg) else None,
        }
        row["clean_false_alert_at_deployed_threshold"] = float((neg >= DEPLOYED_THRESHOLD).mean()) if len(neg) else None
        row["recall_at_deployed_threshold"] = float((pos >= DEPLOYED_THRESHOLD).mean()) if len(pos) else None
        row["wild_clean_images"] = len(d["wild"])
        row["wild_false_alert_at_deployed_threshold"] = float((d["wild"] >= DEPLOYED_THRESHOLD).mean()) if len(d["wild"]) else None
        for fpr in (0.01, 0.05):
            cut = np.quantile(neg, 1 - fpr) if len(neg) else 1.0
            row[f"recall_at_{int(fpr * 100)}pct_clean_fpr"] = float((pos > cut).mean())
        sweep = np.arange(0.30, 0.96, 0.05)
        ious = {round(float(t), 2): [] for t in sweep}
        acc = []
        for any_damage, type_map, gt in d["rows"]:
            g = gt > 0
            if not g.any():
                continue
            for t in sweep:
                pred = any_damage >= t
                ious[round(float(t), 2)].append((pred & g).sum() / max(1, (pred | g).sum()))
            pred = any_damage >= DEPLOYED_THRESHOLD
            acc.append((type_map[g] == gt[g]).mean())
        mean_iou = {t: float(np.mean(v)) for t, v in ious.items() if v}
        if mean_iou:
            row["mean_damage_iou_at_deployed_threshold"] = mean_iou[0.75] if 0.75 in mean_iou else None
            best_t = max(mean_iou, key=mean_iou.get)
            row["mean_damage_iou_best_threshold"] = mean_iou[best_t]
            row["best_iou_threshold"] = best_t
        row["type_pixel_accuracy_on_damage"] = float(np.mean(acc)) if acc else None
        report[split] = row
    return report


def main() -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "source_losses",
        "C:/Users/admin/Desktop/Tie it Up/Port/vehicle-damage-segmentation/src/vehicle_damage/losses.py",
    )
    losses = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(losses)
    RecallOrientedLoss = losses.RecallOrientedLoss

    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--block-lr", type=float, default=1e-5)
    ap.add_argument("--head-lr", type=float, default=5e-5)
    ap.add_argument("--damage-fraction", type=float, default=0.5)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--baseline-only", action="store_true")
    ap.add_argument("--extra-cache-dir", help="wild clean cache (splits merged into train/val/cal/test)")
    ap.add_argument("--baseline-checkpoint", help="checkpoint for the comparison row (default: --checkpoint)")
    ap.add_argument("--samples-per-epoch", type=int, default=0)
    ap.add_argument("--extra-train", action="append", default=[], help="extra train-only cache shards (mined negatives)")
    ap.add_argument("--hard-boost", type=float, default=3.0, help="sampling weight for mined hard negatives")
    ap.add_argument("--hard-from-model", help="checkpoint whose false-alarm regions become the hard-negative mask "
                    "for clean wild photos (instead of weighting the whole frame)")
    ap.add_argument("--hard-region-threshold", type=float, default=0.2)
    ap.add_argument("--minor-boost", type=float, default=1.0,
                    help="sampling weight for damage photos with scratch or paint labels (dent photos get half the boost)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    torch.set_num_threads(10)
    cache = Path(args.cache_dir)
    data = {s: torch.load(cache / f"{s}.pt", weights_only=False) for s in ("validation", "calibration", "test")}
    n_blocks = data["validation"]["n_blocks"]
    eval_items = {s: d["items"] for s, d in data.items()}
    extra = Path(args.extra_cache_dir) if args.extra_cache_dir else None
    if extra:
        for split in eval_items:
            eval_items[split] = eval_items[split] + torch.load(extra / f"{split}.pt", weights_only=False)["items"]
    base_model, _ = load_checkpoint(args.baseline_checkpoint or args.checkpoint, "cpu")
    base = evaluate(base_model, eval_items, n_blocks, "baseline")
    del base_model
    model, ckpt = load_checkpoint(args.checkpoint, "cpu")
    print("BASELINE", json.dumps(base, indent=1), flush=True)
    if args.baseline_only:
        return
    train = torch.load(cache / "train.pt", weights_only=False)["items"]
    if extra:
        train = train + torch.load(extra / "train.pt", weights_only=False)["items"]
    for shard in args.extra_train:
        train = train + torch.load(shard, weights_only=False)["items"]
    print("train items", len(train), flush=True)
    if args.hard_from_model:
        # Targeted hard negatives: penalise only where the reference model fired on a clean photo.
        ref, _ = load_checkpoint(args.hard_from_model, "cpu")
        from scipy import ndimage as _nd
        n_regions = 0
        for it in train:
            if it["source"].startswith("Wikimedia"):
                any_damage, _ = item_maps(ref, it, n_blocks)
                region = _nd.binary_dilation(any_damage >= args.hard_region_threshold, iterations=3)
                it["hard"] = torch.from_numpy(region.astype(np.uint8))
                it["targeted_hard"] = True
                n_regions += int(region.any())
        del ref
        print(f"targeted hard-negative masks: {n_regions} wild photos have false-alarm regions", flush=True)
    for p in model.parameters():
        p.requires_grad = False
    blocks = list(model.backbone.blocks[n_blocks:]) + [model.backbone.norm]
    block_params = [p for m in blocks for p in m.parameters()]
    heads = [model.presence_head, model.type_head, model.exterior_head]
    head_params = [p for m in heads for p in m.parameters()]
    for p in block_params + head_params:
        p.requires_grad = True
    opt = torch.optim.AdamW(
        [{"params": block_params, "lr": args.block_lr}, {"params": head_params, "lr": args.head_lr}],
        weight_decay=1e-4,
    )
    per_epoch = args.samples_per_epoch or len(train)
    steps = args.epochs * per_epoch // args.accum
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[args.block_lr, args.head_lr], total_steps=steps + 5, pct_start=0.1
    )
    criterion = RecallOrientedLoss(len(ckpt["classes"]), **LOSS_CONFIG)
    is_damage = np.array([bool((it["damage"] > 0).any()) for it in train])
    weights = np.where(is_damage, args.damage_fraction / is_damage.sum(), (1 - args.damage_fraction) / (~is_damage).sum())
    def minor(it):
        labels = set(torch.unique(it["damage"]).tolist())
        if labels & {2, 4}:  # scratch, paint damage
            return args.minor_boost
        if 1 in labels:  # dent
            return 1.0 + (args.minor_boost - 1.0) / 2
        return 1.0
    boost = np.array([args.hard_boost if it.get("mined_score", 0.0) >= 0.15 else minor(it) for it in train])
    weights = weights * boost
    weights /= weights.sum()
    best, best_state = -1.0, None
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        model.backbone.eval()  # transformer blocks have no stochastic layers; heads keep dropout
        order = rng.choice(len(train), size=per_epoch, p=weights)
        total = 0.0
        for n, idx in enumerate(order, 1):
            item = train[idx]
            flip = bool(rng.random() < 0.5)
            out = forward(model, item, n_blocks, flip=flip)
            gt = item["damage"].long()[None]
            ext = item["exterior"].float()[None]
            hard = item["hard"].float()[None]
            if item["source"].startswith("Wikimedia") and not item.get("targeted_hard"):
                hard = torch.ones_like(hard)
            if flip:
                gt, ext, hard = gt.flip(2), ext.flip(2), hard.flip(2)
            loss, _ = criterion(out, gt, ext, hard, torch.ones(1), torch.tensor([float(item["has_exterior"])]))
            (loss / args.accum).backward()
            total += float(loss.detach())
            if n % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(block_params + head_params, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
            if n % 200 == 0:
                print(f"epoch {epoch} {n}/{len(order)} loss {total / n:.4f} {time.time() - start:.0f}s", flush=True)
        rep = evaluate(model, eval_items, n_blocks, f"epoch{epoch}")
        v = rep["validation"]
        print(
            f"EPOCH {epoch} val auroc {v['auroc']:.4f} recall@5%fpr {v['recall_at_5pct_clean_fpr']:.3f} "
            f"recall@1%fpr {v['recall_at_1pct_clean_fpr']:.3f} iou@best {v['mean_damage_iou_best_threshold']:.3f} "
            f"type_acc {v['type_pixel_accuracy_on_damage']:.3f} recall@deployed {v['recall_at_deployed_threshold']:.3f} "
            f"wild_FA {v['wild_false_alert_at_deployed_threshold']}",
            flush=True,
        )
        # selection uses validation only: mask quality + type accuracy, guarded by clean separation
        wild_fa = v["wild_false_alert_at_deployed_threshold"] or 0.0
        if args.output:
            epoch_ckpt = dict(ckpt)
            epoch_ckpt["model"] = copy.deepcopy(model.state_dict())
            torch.save(epoch_ckpt, str(args.output).replace(".pt", f".epoch{epoch}.pt"))
        metric = (v["mean_damage_iou_best_threshold"] + 0.5 * v["type_pixel_accuracy_on_damage"] + 0.5 * v["auroc"]
                  + 0.5 * v["recall_at_deployed_threshold"] - 1.5 * wild_fa)
        if metric > best:
            best, best_state = metric, copy.deepcopy(model.state_dict())
            if args.output:
                out_ckpt = dict(ckpt)
                out_ckpt["model"] = best_state
                torch.save(out_ckpt, args.output)
    model.load_state_dict(best_state)
    final = evaluate(model, eval_items, n_blocks, "finetuned")
    print("FINAL", json.dumps({"baseline": base, "finetuned": final}, indent=1))


if __name__ == "__main__":
    main()
