import json

import numpy as np
import pytest
import torch
from PIL import Image

pytest.importorskip("onnxruntime")
pytest.importorskip("onnx")

from vehicle_damage.inference import predict_probabilities
from vehicle_damage.onnx_backend import EXPORT_FORMAT_VERSION, OnnxTileModel, _sha256


class _TinyHeads(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.presence = torch.nn.Conv2d(3, 1, 3, padding=1)
        self.damage_type = torch.nn.Conv2d(3, 2, 3, padding=1)
        self.exterior = torch.nn.Conv2d(3, 1, 3, padding=1)

    def forward(self, image):
        return {
            "presence": self.presence(image),
            "type": self.damage_type(image),
            "exterior": self.exterior(image),
        }


class _Tuple(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image):
        output = self.model(image)
        return output["presence"], output["type"], output["exterior"]


def _export(tmp_path, model, tile_size=48, checkpoint_sha256="a" * 64):
    path = tmp_path / "tile.onnx"
    torch.onnx.export(
        _Tuple(model).eval(),
        (torch.zeros(1, 3, tile_size, tile_size),),
        str(path),
        input_names=["image"],
        output_names=["presence", "type", "exterior"],
        opset_version=17,
        dynamo=False,
    )
    metadata = {
        "format_version": EXPORT_FORMAT_VERSION,
        "checkpoint_sha256": checkpoint_sha256,
        "classes": ["background", "dent", "scratch"],
        "tile_size": tile_size,
        "outputs": ["presence", "type", "exterior"],
        "role_split": False,
        "onnx_sha256": _sha256(path),
    }
    path.with_suffix(".json").write_text(json.dumps(metadata), encoding="utf-8")
    return path


def test_onnx_tile_model_matches_pytorch_through_calibrated_tiling(tmp_path):
    model = _TinyHeads().eval()
    onnx_model = OnnxTileModel(_export(tmp_path, model), device="cpu")
    assert onnx_model.description == "onnx:cpu:fp32"
    image = Image.fromarray(
        np.random.default_rng(3).integers(0, 255, (73, 91, 3), dtype=np.uint8)
    )
    with torch.inference_mode():
        reference = predict_probabilities(
            model, image, device="cpu", tile_size=48, overlap=12
        )
    produced = predict_probabilities(
        onnx_model, image, device="cpu", tile_size=48, overlap=12, mixed_precision=True
    )
    for expected, actual in zip(reference, produced):
        assert torch.allclose(expected, actual, atol=1e-5)


def test_onnx_tile_model_fails_closed_on_tampering_and_wrong_checkpoint(tmp_path):
    path = _export(tmp_path, _TinyHeads().eval())
    with pytest.raises(ValueError, match="different checkpoint"):
        OnnxTileModel(path, device="cpu", expected_checkpoint_sha256="b" * 64)
    with path.open("ab") as handle:
        handle.write(b"\0")
    with pytest.raises(ValueError, match="integrity"):
        OnnxTileModel(path, device="cpu")


def test_onnx_tile_model_rejects_uncalibrated_tile_size(tmp_path):
    onnx_model = OnnxTileModel(_export(tmp_path, _TinyHeads().eval()), device="cpu")
    with pytest.raises(ValueError, match="expects"):
        onnx_model(torch.zeros(1, 3, 32, 32))


def test_npu_context_cache_is_bound_to_its_source_export(tmp_path):
    from vehicle_damage.onnx_backend import _valid_context_cache

    cache = tmp_path / "tile.npu-fp16.ctx.onnx"
    cache.write_bytes(b"compiled")
    assert not _valid_context_cache(cache, "a" * 64)  # no binding record
    cache.with_suffix(".json").write_text(
        json.dumps({"source_onnx_sha256": "a" * 64, "context_sha256": _sha256(cache)}),
        encoding="utf-8",
    )
    assert _valid_context_cache(cache, "a" * 64)
    assert not _valid_context_cache(cache, "b" * 64)  # re-exported model
    cache.write_bytes(b"tampered")
    assert not _valid_context_cache(cache, "a" * 64)
