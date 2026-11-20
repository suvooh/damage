"""Mine hard negatives from clean car photos with the current model.

For each clean photo: compute the frozen prefix features (as damage_cache.py
does) and the current model's proxy case score. Keep every photo scoring at or
above --keep-score (the model "sees" damage on a clean car) plus a random
fraction of the easy ones, and save them as a training cache shard.

Usage: python tools/mine_hard_negatives.py CHECKPOINT OUT.pt MANIFEST [MANIFEST...]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).parent))
from damage_cache import load_row, prefix_tokens  # noqa: E402
from finetune_damage import case_score, item_maps  # noqa: E402
from vehicle_damage.model import load_checkpoint  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("output")
    ap.add_argument("manifests", nargs="+")
    ap.add_argument("--split", default="train")
    ap.add_argument("--keep-score", type=float, default=0.15)
    ap.add_argument("--easy-fraction", type=float, default=0.25)
    ap.add_argument("--long-side", type=int, default=784)
    ap.add_argument("--train-blocks", type=int, default=3)
    args = ap.parse_args()
    torch.set_num_threads(10)
    rng = random.Random(5)
    model, _ = load_checkpoint(args.checkpoint, "cpu")
    model.eval()
    n_blocks = len(model.backbone.blocks) - args.train_blocks
    rows = []
    for m in args.manifests:
        rows += [json.loads(line) for line in Path(m).read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [r for r in rows if r["split"] == args.split]
    kept, scores, start = [], [], time.time()
    for i, row in enumerate(rows):
        try:
            x, grid, damage, exterior, hard, has_ext = load_row(
                {"image": row["image"]}, Path("/"), args.long_side)
        except Exception as exc:
            print("skip", row["image"], exc, flush=True)
            continue
        with torch.no_grad():
            tokens = prefix_tokens(model.backbone, x[None], n_blocks)
        item = {"tokens": tokens[0].half(), "grid": grid, "damage": torch.from_numpy(damage),
                "exterior": torch.from_numpy(exterior), "hard": torch.from_numpy(hard), "has_exterior": False,
                "group": row["group_id"], "source": "Wikimedia Commons wild clean (mined)", "tags": ["wild_clean", "mined"]}
        score = case_score(item_maps(model, item, n_blocks)[0])
        scores.append((row["group_id"], score))
        if score >= args.keep_score or rng.random() < args.easy_fraction:
            item["mined_score"] = score
            kept.append(item)
        if i % 50 == 0:
            hard_n = sum(s >= args.keep_score for _, s in scores)
            print(f"{i}/{len(rows)} kept {len(kept)} (hard {hard_n}) {time.time() - start:.0f}s", flush=True)
    torch.save({"items": kept, "n_blocks": n_blocks, "long_side": args.long_side}, args.output)
    Path(args.output).with_suffix(".scores.json").write_text(json.dumps(scores), encoding="utf-8")
    hard_n = sum(s >= args.keep_score for _, s in scores)
    print(json.dumps({"scored": len(scores), "hard": hard_n, "alerts_at_0.381": sum(s >= 0.381 for _, s in scores),
                      "kept": len(kept)}), flush=True)


if __name__ == "__main__":
    main()
