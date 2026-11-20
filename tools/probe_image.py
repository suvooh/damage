"""Score images with the single-scale (784 px) proxy pipeline for several checkpoints.

Usage: python tools/probe_image.py NAME=ckpt.pt [NAME=ckpt.pt ...] -- img1.jpg img2.jpg
Prints, per image and model: the case score (highest threshold at which a
>=0.1% region survives), the top region's type and bbox (as % of the image).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).parent))
from finetune_damage import EXTERIOR_FLOOR, case_score  # noqa: E402
from vehicle_damage.inference import MEAN, STD  # noqa: E402
from vehicle_damage.model import damage_probabilities, load_checkpoint  # noqa: E402

CLASSES = ["background", "dent", "scratch", "crack_or_breakage", "paint_damage", "deformation_or_detachment"]


def prepare(path: str, long_side: int = 784):
    image = Image.open(path).convert("RGB")
    scale = long_side / max(image.size)
    w = max(14, round(image.width * scale / 14) * 14)
    h = max(14, round(image.height * scale / 14) * 14)
    x = torch.from_numpy(np.asarray(image.resize((w, h), Image.Resampling.BICUBIC)).copy()).permute(2, 0, 1).float() / 255.0
    return ((x - MEAN) / STD)[None]


@torch.no_grad()
def maps(model, x):
    out = model(x)
    probs = damage_probabilities(out)[0]
    ext = out["exterior"].sigmoid()[0, 0]
    return ((1 - probs[0]) * (EXTERIOR_FLOOR + (1 - EXTERIOR_FLOOR) * ext)).numpy(), probs[1:].argmax(0).numpy() + 1


def main() -> None:
    split = sys.argv.index("--")
    models = {}
    for spec in sys.argv[1:split]:
        name, path = spec.split("=", 1)
        models[name] = load_checkpoint(path, "cpu")[0].eval()
    for path in sys.argv[split + 1:]:
        x = prepare(path)
        for name, model in models.items():
            damage, types = maps(model, x)
            score = case_score(damage)
            line = f"{Path(path).name:28s} {name:10s} case_score={score:.2f}"
            if score > 0:
                mask = damage >= max(score - 0.02, 0.05)
                labels, count = ndimage.label(mask, structure=np.ones((3, 3)))
                sizes = np.bincount(labels.ravel())[1:]
                top = int(sizes.argmax()) + 1
                ys, xs = np.where(labels == top)
                height, width = damage.shape
                cls = CLASSES[int(np.bincount(types[labels == top]).argmax())]
                line += f" type={cls} bbox_pct=({xs.min() / width * 100:.0f},{ys.min() / height * 100:.0f})-({xs.max() / width * 100:.0f},{ys.max() / height * 100:.0f})"
            print(line, flush=True)


if __name__ == "__main__":
    main()
