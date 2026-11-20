"""Rebuild per-part label masks for the CC0 Humans in the Loop car-parts subset.

The original extraction only kept a binary "exterior" mask. Part names are
needed to say *which* panel is damaged, so this re-reads the annotation JSONs
(range requests, no full 3 GB download) and rasterises one class id per part.
The shuffle seed matches the original extraction, so indices line up with the
existing ``hitl_parts_*.jpg`` images; the union of the new part masks is checked
against the stored exterior mask to prove it.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

PARTS = (
    "background", "windshield", "back-windshield", "front-window", "back-window",
    "front-door", "back-door", "front-wheel", "back-wheel", "front-bumper",
    "back-bumper", "headlight", "tail-light", "hood", "trunk", "license-plate",
    "mirror", "roof", "grille", "rocker-panel", "quarter-panel", "fender",
)
PART_ID = {name: i for i, name in enumerate(PARTS)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--archive", required=True)
    p.add_argument("--source-scripts", required=True, help="Port/vehicle-damage-segmentation/scripts")
    p.add_argument("--parts-dir", required=True, help="data/hitl_parts_cc0 (images + exterior_masks)")
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=23)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()
    sys.path.insert(0, args.source_scripts)
    from prepare_hitl_parts_cc0 import _find_parts_root  # noqa: E402

    sys.path.insert(0, str(Path(__file__).parent))
    from localzip import LocalZip  # noqa: E402

    remote = LocalZip(args.archive)
    root, _ = _find_parts_root(remote)
    names = sorted(n for n in remote.entries if n.startswith(f"{root}/File1/ann/") and n.endswith(".json"))
    random.Random(args.seed).shuffle(names)
    parts_dir, out = Path(args.parts_dir), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    def convert(i: int) -> tuple[int, float]:
        stem = f"hitl_parts_{i:05d}"
        ext = np.asarray(Image.open(parts_dir / "exterior_masks" / f"{stem}.png")) > 0
        height, width = ext.shape
        ann = json.loads(remote.read(names[i]))
        size = (ann["size"]["width"], ann["size"]["height"])
        mask = Image.new("L", size, 0)
        draw = ImageDraw.Draw(mask)
        for obj in ann.get("objects", []):  # annotation order == paint order (later wins)
            pid = PART_ID[str(obj["classTitle"]).strip().lower()]
            pts = obj.get("points", {}).get("exterior", [])
            if len(pts) >= 3:
                draw.polygon([(float(x), float(y)) for x, y in pts], fill=pid)
        mask = mask.resize((width, height), Image.Resampling.NEAREST)
        arr = np.asarray(mask)
        iou = float(((arr > 0) & ext).sum() / max(1, ((arr > 0) | ext).sum()))
        Image.fromarray(arr).save(out / f"{stem}.png", optimize=True)
        return i, iou

    count = len(list(parts_dir.glob("images/*.jpg")))
    with ThreadPoolExecutor(args.workers) as pool:
        results = list(pool.map(convert, range(count)))
    ious = np.array([r[1] for r in results])
    print(json.dumps({"images": count, "min_iou": float(ious.min()), "mean_iou": float(ious.mean()),
                      "below_0.95": int((ious < 0.95).sum())}))


if __name__ == "__main__":
    main()
