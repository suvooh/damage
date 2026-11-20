"""How trustworthy is each predicted damage type, as a function of its confidence?

Uses the cached 784px features of the held-out splits and the shipped model.
For each predicted-type pixel inside predicted damage (the app's own threshold),
compare with the ground-truth type where ground truth marks damage.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).parent))
from damage_cache import UP, suffix_features  # noqa: E402
from vehicle_damage.model import damage_probabilities, load_checkpoint  # noqa: E402

NAMES = ["", "dent", "scratch", "crack_or_breakage", "paint_damage", "deformation_or_detachment"]
THRESH = 0.381
model, _ = load_checkpoint(sys.argv[1], "cpu")
rows = []  # (pred_type, confidence, gt_type)
for split in ("validation", "calibration", "test"):
    data = torch.load(Path(sys.argv[2]) / f"{split}.pt", weights_only=False)
    n_blocks = data["n_blocks"]
    for item in data["items"]:
        gt = item["damage"].numpy()
        if not (gt > 0).any():
            continue
        with torch.no_grad():
            feats = suffix_features(model.backbone, item["tokens"].float()[None], n_blocks, item["grid"])
            out = {k: torch.nn.functional.interpolate(h(feats), scale_factor=UP, mode="bilinear", align_corners=False)
                   for k, h in (("presence", model.presence_head), ("type", model.type_head), ("exterior", model.exterior_head))}
            probs = damage_probabilities(out)[0]
            ext = out["exterior"].sigmoid()[0, 0]
        anyd = ((1 - probs[0]) * (0.5 + 0.5 * ext)).numpy()
        tp = probs[1:]
        norm = (tp / tp.sum(0, keepdim=True)).numpy()
        pred = norm.argmax(0) + 1
        conf = norm.max(0)
        m = (anyd >= THRESH) & (gt > 0)
        rows.append(np.stack([pred[m], (conf[m] * 100).astype(int), gt[m]], 1))
r = np.concatenate(rows)
print("pixels:", len(r))
print("\nconfusion (rows = predicted, cols = true), row-normalised %:")
print(" " * 26 + "".join(f"{n[:9]:>10}" for n in NAMES[1:]))
for p in range(1, 6):
    sel = r[r[:, 0] == p]
    if len(sel):
        print(f"{NAMES[p]:26}" + "".join(f"{100 * (sel[:, 2] == t).mean():10.0f}" for t in range(1, 6)) + f"   n={len(sel)}")
for p in (3, 5, 2, 4, 1):
    sel = r[r[:, 0] == p]
    print(f"\n{NAMES[p]}: precision when confidence >= x")
    for x in (0, 50, 60, 70, 80, 90):
        s = sel[sel[:, 1] >= x]
        if len(s) > 200:
            print(f"  conf>={x:2d}%  precision {100 * (s[:, 2] == p).mean():5.1f}%   pixels kept {100 * len(s) / len(sel):5.1f}%")
