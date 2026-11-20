"""Cache frozen DINOv2 prefix features (blocks 0..n-K-1) for damage fine-tuning.

Each manifest row becomes: tokens after the frozen prefix (fp16), grid size, and
damage/exterior/hard-negative label maps at 4x patch-grid resolution. Training
then only runs the last K transformer blocks + the three heads.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vehicle_damage.inference import MEAN, STD  # noqa: E402
from vehicle_damage.model import load_checkpoint  # noqa: E402

UP = 4


def prefix_tokens(backbone, x: torch.Tensor, n_blocks: int) -> torch.Tensor:
    x = backbone.patch_embed(x)
    x = backbone._pos_embed(x)
    x = backbone.patch_drop(x)
    x = backbone.norm_pre(x)
    for block in backbone.blocks[:n_blocks]:
        x = block(x)
    return x


def suffix_features(backbone, tokens: torch.Tensor, n_blocks: int, grid: tuple[int, int]) -> torch.Tensor:
    x = tokens
    for block in backbone.blocks[n_blocks:]:
        x = block(x)
    x = backbone.norm(x)
    x = x[:, backbone.num_prefix_tokens:]
    return x.transpose(1, 2).reshape(x.shape[0], x.shape[2], *grid)


def load_row(row: dict, root: Path, long_side: int):
    def path(key):
        v = row.get(key)
        return None if v is None else root / v

    image = Image.open(path("image")).convert("RGB")
    scale = long_side / max(image.size)
    w = max(14, round(image.width * scale / 14) * 14)
    h = max(14, round(image.height * scale / 14) * 14)
    gh, gw = h // 14, w // 14
    size = (gw * UP, gh * UP)

    def mask(key, default):
        p = path(key)
        if p is None:
            return np.full((size[1], size[0]), default, np.uint8), p is not None
        return np.asarray(Image.open(p).convert("L").resize(size, Image.Resampling.NEAREST)), True

    damage, _ = mask("mask", 0)
    exterior, has_ext = mask("exterior_mask", 0)
    hard, _ = mask("hard_negative_mask", 0)
    x = torch.from_numpy(np.asarray(image.resize((w, h), Image.Resampling.BICUBIC)).copy()).permute(2, 0, 1).float() / 255.0
    return (x - MEAN) / STD, (gh, gw), damage, (exterior > 0).astype(np.uint8), (hard > 0).astype(np.uint8), has_ext


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--long-side", type=int, default=784)
    ap.add_argument("--train-blocks", type=int, default=3)
    ap.add_argument("--splits", default="validation,calibration,test,train")
    args = ap.parse_args()
    torch.set_num_threads(10)
    model, _ = load_checkpoint(args.checkpoint, "cpu")
    backbone = model.backbone
    n_blocks = len(backbone.blocks) - args.train_blocks
    manifest = Path(args.manifest)
    rows = [json.loads(l) for l in manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    first = True
    for split in args.splits.split(","):
        target = out / f"{split}.pt"
        if target.is_file():
            continue
        items, start = [], time.time()
        subset = [r for r in rows if r["split"] == split]
        for i, row in enumerate(subset):
            x, grid, damage, exterior, hard, has_ext = load_row(row, manifest.parent, args.long_side)
            with torch.no_grad():
                tokens = prefix_tokens(backbone, x[None], n_blocks)
                if first:  # prove prefix+suffix == the deployed forward pass
                    ref = model._extract_patch_features(x[None])
                    got = suffix_features(backbone, tokens, n_blocks, grid)
                    err = float((ref - got).abs().max())
                    assert err < 1e-3, f"prefix/suffix split diverges from model: {err}"
                    print("prefix/suffix parity max abs err", err, flush=True)
                    first = False
            items.append({
                "tokens": tokens[0].half(), "grid": grid, "damage": torch.from_numpy(damage),
                "exterior": torch.from_numpy(exterior), "hard": torch.from_numpy(hard),
                "has_exterior": has_ext, "group": row["group_id"], "source": row["source"],
                "tags": row.get("tags", []),
            })
            if i % 25 == 0:
                print(f"{split} {i}/{len(subset)} {time.time() - start:.0f}s", flush=True)
        torch.save({"items": items, "n_blocks": n_blocks, "long_side": args.long_side}, target)
        print(f"saved {split}: {len(items)}", flush=True)


if __name__ == "__main__":
    main()
