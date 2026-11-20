"""Compare checkpoints on held-out clean and damaged photos (cached proxy features).

Clean set: a cache shard of fresh clean photos (e.g. eval_test2.pt).
Damaged set: validation + test damage photos from the main cache.
Reports false alerts and recall at fixed thresholds and at matched false-alert rates.

Usage: python tools/compare_models.py CLEAN.pt CACHE_DIR NAME=ckpt [NAME=ckpt ...]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).parent))
from finetune_damage import case_score, item_maps  # noqa: E402
from vehicle_damage.model import load_checkpoint  # noqa: E402


def scores(model, items, n_blocks):
    return np.array([case_score(item_maps(model, it, n_blocks)[0]) for it in items])


def main() -> None:
    torch.set_num_threads(10)
    clean_shard = torch.load(sys.argv[1], weights_only=False)
    clean, n_blocks = clean_shard["items"], clean_shard["n_blocks"]
    cache = Path(sys.argv[2])
    damage = []
    for split in ("validation", "test"):
        damage += [it for it in torch.load(cache / f"{split}.pt", weights_only=False)["items"] if (it["damage"] > 0).any()]
    print(f"clean photos {len(clean)}, damaged photos {len(damage)}")
    for spec in sys.argv[3:]:
        name, path = spec.split("=", 1)
        model, _ = load_checkpoint(path, "cpu")
        c, d = scores(model, clean, n_blocks), scores(model, damage, n_blocks)
        print(f"\n{name}")
        for t in (0.381, 0.45, 0.5):
            print(f"  threshold {t:.3f}: clean flagged {(c >= t).mean():6.1%}   damage caught {(d >= t).mean():6.1%}")
        for fa in (0.01, 0.02, 0.05):
            cut = np.quantile(c, 1 - fa)
            print(f"  at {fa:.0%} clean false alerts (thr {cut:.3f}): damage caught {(d > cut).mean():6.1%}")
        np.save(Path(sys.argv[1]).with_name(f"scores_{name}.npy"), {"clean": c, "damage": d}, allow_pickle=True)


if __name__ == "__main__":
    main()
