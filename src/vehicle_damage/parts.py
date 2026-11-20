"""Car-part segmentation (ONNX) used to name the part under each damage region.

One fixed 644x644 letterboxed pass per image: parts are large structures, so
this is cheap next to the tiled damage pass. The graph is the damage
checkpoint's frozen DINOv2 backbone plus a small trained part head.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from .inference import MEAN, STD

SIDE = 644
UP = 4
GRID = SIDE // 14 * UP


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PartSegmenter:
    def __init__(self, onnx_path: str | Path, meta_path: str | Path | None = None) -> None:
        onnx_path = Path(onnx_path)
        meta = json.loads(Path(meta_path or onnx_path.with_suffix(".json")).read_text(encoding="utf-8"))
        if _sha256(onnx_path) != meta["onnx_sha256"]:
            raise ValueError("part model failed its integrity check")
        self.part_names: list[str] = meta["parts"]
        self.meta = meta
        options = ort.SessionOptions()
        self.session = ort.InferenceSession(
            str(onnx_path), options, providers=["CPUExecutionProvider"]
        )

    def predict(self, image: Image.Image) -> np.ndarray:
        """Return a ``(height, width)`` uint8 part-id mask for ``image``."""
        rgb = image.convert("RGB")
        scale = SIDE / max(rgb.size)
        w, h = max(1, round(rgb.width * scale)), max(1, round(rgb.height * scale))
        resized = np.asarray(rgb.resize((w, h), Image.Resampling.BICUBIC), dtype=np.float32) / 255.0
        canvas = np.pad(resized, ((0, SIDE - h), (0, SIDE - w), (0, 0)), mode="edge")
        x = (canvas.transpose(2, 0, 1) - MEAN.numpy()) / STD.numpy()
        logits = self.session.run(None, {"image": x[None].astype(np.float32)})[0][0]
        ids = logits.argmax(0).astype(np.uint8)
        gh, gw = max(1, round(h / SIDE * GRID)), max(1, round(w / SIDE * GRID))
        ids = ids[:gh, :gw]
        return np.asarray(
            Image.fromarray(ids).resize(rgb.size, Image.Resampling.NEAREST)
        )
