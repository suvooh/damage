"""Score images with the REAL deployed pipeline (tiled, multi-scale, flip TTA, ONNX).

For every image we store the "case score": the highest any-damage threshold at
which a connected region of at least the profile's minimum component size still
exists. A threshold sweep (calibration, evaluation) can then be done offline
without re-running the model. Results are appended to a JSONL file; reruns skip
images already scored.

Usage: python tools/real_pipeline_scores.py OUT.jsonl SPEC.json
SPEC.json = [{"path": ..., "label": "damage"|"clean", "set": "calibration"|"test",
              "source": "hitl"|"coco"|"wild", "id": ...}, ...]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vehicle_damage.api import DamagePredictor, _score_map  # noqa: E402
from vehicle_damage.inference import (  # noqa: E402
    assess_quality, glare_zone, predict_profile_probabilities, restore_size, split_glare_components, working_image)

LEVELS = np.round(np.linspace(0.05, 0.99, 48), 4)
GRID = bool(int(__import__('os').environ.get('GLARE_GRID', '0')))
EIGHT = np.ones((3, 3))


def case_score(score_map: np.ndarray, minimum_pixels: int, zone: np.ndarray | None = None,
               dominance: float = 0.5) -> float:
    """Highest threshold at which a region of at least ``minimum_pixels`` survives.

    With ``zone`` (glare zone), regions mostly on lights or glare do not count, as in the app.
    """
    def survives(i: int) -> bool:
        above = score_map >= LEVELS[i]
        if zone is not None:
            above, _ = split_glare_components(above.astype(np.uint8), zone, dominance)
            above = above > 0
        labels, count = ndimage.label(above, structure=EIGHT)
        return count > 0 and int(np.bincount(labels.ravel())[1:].max()) >= minimum_pixels

    if not survives(0):
        return 0.0
    lo, hi = 0, len(LEVELS) - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        lo, hi = (mid, hi) if survives(mid) else (lo, mid - 1)
    return float(LEVELS[lo])


def main() -> None:
    out_path, spec_path = Path(sys.argv[1]), Path(sys.argv[2])
    items = json.loads(spec_path.read_text(encoding="utf-8"))
    done = set()
    if out_path.is_file():
        done = {json.loads(line)["id"] for line in out_path.read_text(encoding="utf-8").splitlines() if line}
    predictor = DamagePredictor()
    cal = predictor.calibration
    print(f"device {predictor.accelerator}; {len(items) - len(done)} images to score", flush=True)
    started = time.time()
    with out_path.open("a", encoding="utf-8") as sink:
        for n, item in enumerate(items, 1):
            if item["id"] in done:
                continue
            with Image.open(item["path"]) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
            quality = assess_quality(image)
            maps, _ = predict_profile_probabilities(
                predictor.model, working_image(image), device=predictor.device,
                triage_scales=cal.triage_scales, segmentation_scales=cal.effective_segmentation_scales,
                tile_size=cal.tile_size, overlap=cal.overlap,
                horizontal_flip_tta=cal.horizontal_flip_tta, mixed_precision=cal.mixed_precision,
            )
            damage, exterior = restore_size(maps, image.size)
            score_map = _score_map(damage, exterior, cal.exterior_floor)
            row = {**{k: item[k] for k in ("id", "label", "set", "source")},
                   "case_score": case_score(score_map, cal.minimum_component_pixels, glare_zone(image)),
                   "case_score_no_glare_filter": case_score(score_map, cal.minimum_component_pixels),
                   "glare_grid": ({f"{lv}|{halo}|{dom}": case_score(score_map, cal.minimum_component_pixels,
                                                                  glare_zone(image, lv, halo), dom)
                                   for lv in (0.95, 0.985) for halo in (0.008, 0.015, 0.025) for dom in (0.5, 0.7, 0.85)}
                                  if GRID else None),
                   "max_score": float(score_map.max()), "quality": list(quality.review_reasons),
                   "size": [image.width, image.height]}
            sink.write(json.dumps(row) + "\n")
            sink.flush()
            if n % 10 == 0:
                print(f"{n}/{len(items)} {time.time() - started:.0f}s", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
