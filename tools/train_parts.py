"""Train a car-part segmentation head on the frozen damage backbone.

The backbone comes from the pinned damage checkpoint, so deployment shares one
ViT forward pass between damage and part prediction. Backbone features are
cached once (with and without horizontal flip -- part labels are front/back,
not left/right, so flips are label-preserving), then only the small conv head is
trained. Holds out 15% of images and reports per-part IoU.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vehicle_damage.inference import MEAN, STD  # noqa: E402
from vehicle_damage.model import load_checkpoint  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from prepare_part_masks import PARTS  # noqa: E402

LONG_SIDE = 644  # 46 DINOv2 patches
UP = 4  # label resolution = patch grid * UP


def make_head(feature_dim: int, classes: int) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(feature_dim, 256, 3, padding=1, bias=False),
        nn.GroupNorm(32, 256),
        nn.GELU(),
        nn.Dropout2d(0.1),
        nn.Conv2d(256, classes, 1),
    )


def load_pair(root: Path, stem: str):
    image = Image.open(root / "images" / f"{stem}.jpg").convert("RGB")
    mask = Image.open(root / "part_masks" / f"{stem}.png")
    scale = LONG_SIDE / max(image.size)
    w, h = max(14, round(image.width * scale / 14) * 14), max(14, round(image.height * scale / 14) * 14)
    image = image.resize((w, h), Image.Resampling.BICUBIC)
    mask = mask.resize((w // 14 * UP, h // 14 * UP), Image.Resampling.NEAREST)
    x = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 255.0
    return (x - MEAN) / STD, torch.from_numpy(np.asarray(mask).astype(np.int64))


@torch.no_grad()
def extract(model, root: Path, stems: list[str]):
    feats, labels = [], []
    start = time.time()
    for i, stem in enumerate(stems):
        x, y = load_pair(root, stem)
        batch = torch.stack([x, x.flip(-1)])
        f = model._extract_patch_features(batch).half()
        feats += [f[0], f[1]]
        labels += [y, y.flip(-1)]
        if i % 50 == 0:
            print(f"features {i}/{len(stems)} {time.time() - start:.0f}s", flush=True)
    return feats, labels


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts-dir", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    torch.set_num_threads(10)
    root = Path(args.parts_dir)
    model, _ = load_checkpoint(args.checkpoint, "cpu")
    for p in model.parameters():
        p.requires_grad = False
    stems = sorted(p.stem for p in (root / "images").glob("*.jpg"))
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(stems))
    held = {stems[i] for i in order[: int(0.15 * len(stems))]}
    train_stems = [s for s in stems if s not in held]
    val_stems = [s for s in stems if s in held]
    cache = Path(args.output).with_suffix(".features.pt")
    if cache.is_file():
        tr_f, tr_y, va_f, va_y = torch.load(cache)
    else:
        tr_f, tr_y = extract(model, root, train_stems)
        va_f, va_y = extract(model, root, val_stems)
        torch.save((tr_f, tr_y, va_f, va_y), cache)
    va_f, va_y = va_f[0::2], va_y[0::2]  # evaluate un-flipped only
    classes = len(PARTS)
    head = make_head(model.backbone.num_features, classes)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    steps = args.epochs * ((len(tr_f) + 7) // 8)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps)
    counts = torch.stack([torch.bincount(y.flatten(), minlength=classes) for y in tr_y]).sum(0).float()
    weight = (counts.sum() / (counts + 1)).pow(0.3)
    weight = weight / weight.mean()

    def logits_for(f: torch.Tensor) -> torch.Tensor:
        return F.interpolate(head(f.float()[None]), scale_factor=UP, mode="bilinear", align_corners=False)

    def evaluate() -> tuple[float, list[float]]:
        head.eval()
        inter, union = torch.zeros(classes), torch.zeros(classes)
        with torch.no_grad():
            for f, y in zip(va_f, va_y):
                pred = logits_for(f).argmax(1)[0]
                for c in range(classes):
                    a, b = pred == c, y == c
                    inter[c] += (a & b).sum()
                    union[c] += (a | b).sum()
        iou = inter / union.clamp_min(1)
        present = union > 0
        return float(iou[present].mean()), [float(v) for v in iou]

    for epoch in range(args.epochs):
        head.train()
        perm = torch.randperm(len(tr_f))
        total = 0.0
        for b in range(0, len(perm), 8):
            idx = perm[b : b + 8]
            loss = 0.0
            for i in idx.tolist():
                loss = loss + F.cross_entropy(logits_for(tr_f[i]), tr_y[i][None], weight=weight)
            loss = loss / len(idx)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += float(loss)
        if epoch % 5 == 4 or epoch == args.epochs - 1:
            miou, _ = evaluate()
            print(f"epoch {epoch + 1} loss {total / (len(perm) / 8):.4f} val_mIoU {miou:.4f}", flush=True)
    miou, iou = evaluate()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"head": head.state_dict(), "parts": list(PARTS), "feature_dim": model.backbone.num_features,
                "val_miou": miou, "val_stems": sorted(held)}, out)
    print(json.dumps({"val_miou": miou, "per_part_iou": dict(zip(PARTS, iou))}, indent=1))


if __name__ == "__main__":
    main()
