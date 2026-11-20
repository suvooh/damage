"""Export backbone + trained part head to a pinned ONNX graph and score it."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).parent))
from train_parts import make_head  # noqa: E402
from vehicle_damage.model import load_checkpoint  # noqa: E402
from vehicle_damage.onnx_backend import _static_position_embedding  # noqa: E402
from vehicle_damage.parts import SIDE, UP, PartSegmenter  # noqa: E402


class PartsNet(nn.Module):
    def __init__(self, model, head):
        super().__init__()
        self.model, self.head = model, head

    def forward(self, image):
        f = self.model._extract_patch_features(image)
        return F.interpolate(self.head(f), scale_factor=UP, mode="bilinear", align_corners=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--head", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--parts-dir", required=True, help="for held-out scoring")
    args = ap.parse_args()
    model, ckpt = load_checkpoint(args.checkpoint, "cpu")
    bundle = torch.load(args.head)
    head = make_head(bundle["feature_dim"], len(bundle["parts"]))
    head.load_state_dict(bundle["head"])
    net = PartsNet(model, head.eval()).eval()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with _static_position_embedding(model.backbone, SIDE):
        torch.onnx.export(net, torch.zeros(1, 3, SIDE, SIDE), str(out), input_names=["image"],
                          output_names=["parts"], opset_version=17, dynamo=False)
    meta = {"onnx_sha256": hashlib.sha256(out.read_bytes()).hexdigest(), "parts": bundle["parts"],
            "side": SIDE, "backbone_checkpoint_sha256": hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
            "train_val_miou": bundle["val_miou"]}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    seg = PartSegmenter(out)
    root, classes = Path(args.parts_dir), len(bundle["parts"])
    inter, union = np.zeros(classes), np.zeros(classes)
    for stem in bundle["val_stems"]:
        image = Image.open(root / "images" / f"{stem}.jpg")
        truth = np.asarray(Image.open(root / "part_masks" / f"{stem}.png"))
        pred = seg.predict(image)
        for c in range(classes):
            a, b = pred == c, truth == c
            inter[c] += (a & b).sum(); union[c] += (a | b).sum()
    iou = inter / np.maximum(union, 1)
    print(json.dumps({"deployed_pipeline_val_miou": float(iou[union > 0].mean()),
                      "per_part": dict(zip(bundle["parts"], map(float, iou)))}, indent=1))


if __name__ == "__main__":
    main()
