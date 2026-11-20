"""Per damage-type detection rate on labelled held-out photos (cached proxy features).

A labelled damage region of type c counts as found when the model's any-damage
map, at the model's own threshold, covers at least 10% of that region's pixels.
Usage: python tools/per_class_recall.py CACHE_DIR NAME=ckpt:threshold [...]
"""
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).parent))
from finetune_damage import item_maps  # noqa: E402
from vehicle_damage.model import load_checkpoint  # noqa: E402

NAMES = {1: "dent", 2: "scratch", 3: "crack_or_breakage", 4: "paint_damage", 5: "deformation"}
torch.set_num_threads(10)
cache = Path(sys.argv[1])
N_BLOCKS = torch.load(cache / "validation.pt", weights_only=False)["n_blocks"]
items = []
for split in ("validation", "test"):
    items += [it for it in torch.load(cache / f"{split}.pt", weights_only=False)["items"] if (it["damage"] > 0).any()]
regions = []  # (item index, class, mask)
for i, it in enumerate(items):
    gt = it["damage"].numpy()
    for c in NAMES:
        labels, n = ndimage.label(gt == c, structure=np.ones((3, 3)))
        for k in range(1, n + 1):
            m = labels == k
            if m.sum() >= 20:
                regions.append((i, c, m))
print("labelled regions:", {NAMES[c]: sum(1 for _, cc, _ in regions if cc == c) for c in NAMES})
for spec in sys.argv[2:]:
    name, rest = spec.split("=", 1)
    path, thr = rest.rsplit(":", 1)
    model, _ = load_checkpoint(path, "cpu")
    pred = [item_maps(model, it, N_BLOCKS)[0] >= float(thr) for it in items]
    line = f"{name:6s} thr {float(thr):.3f}: "
    for c in NAMES:
        rs = [(i, m) for i, cc, m in regions if cc == c]
        found = sum((pred[i][m]).mean() >= 0.10 for i, m in rs)
        line += f"{NAMES[c]} {found}/{len(rs)}={found / max(1, len(rs)):.0%}  "
    print(line, flush=True)
