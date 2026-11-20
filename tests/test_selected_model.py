import hashlib
import json
from pathlib import Path

from vehicle_damage.api import ROOT


def test_selected_model_artifacts_are_pinned_and_development_only():
    selection = json.loads(
        (ROOT / "runs" / "SELECTED_DEVELOPMENT_MODEL.json").read_text(
            encoding="utf-8"
        )
    )
    assert selection["production_approved"] is False
    assert selection["status"] == "development_only_not_production_approved"
    assert set(selection["artifacts"]) == {"checkpoint", "development_profile", "parts_model"}
    for artifact in selection["artifacts"].values():
        path = ROOT / artifact["path"]
        assert path.is_file()
        assert path.stat().st_size == artifact["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact["sha256"]


def test_frozen_profile_is_self_consistent():
    selection = json.loads(
        (ROOT / "runs" / "SELECTED_DEVELOPMENT_MODEL.json").read_text(
            encoding="utf-8"
        )
    )
    profile = json.loads(
        (ROOT / selection["artifacts"]["development_profile"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    assert profile["any_damage_threshold"] == 0.381
    assert profile["tile_size"] == 768
    assert profile["overlap"] == 192
    assert profile["horizontal_flip_tta"] is True
    assert profile["mixed_precision"] is True
    assert profile["minimum_component_pixels"] == 1200
    assert profile["triage_scales"] == [1.0, 1.5]
    assert profile["segmentation_scales"] is None


def test_pinned_onnx_export_matches_checkpoint_and_metadata():
    selection = json.loads(
        (ROOT / "runs" / "SELECTED_DEVELOPMENT_MODEL.json").read_text(
            encoding="utf-8"
        )
    )
    checkpoint_sha = selection["artifacts"]["checkpoint"]["sha256"]
    checkpoint = ROOT / selection["artifacts"]["checkpoint"]["path"]
    onnx_dir = checkpoint.parent / "onnx"
    onnx_path = onnx_dir / f"{checkpoint.stem}-{checkpoint_sha[:12]}-tile768.onnx"
    metadata_path = onnx_path.with_suffix(".json")
    assert onnx_path.is_file()
    assert metadata_path.is_file()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["checkpoint_sha256"] == checkpoint_sha
    assert metadata["format_version"] == 1
    assert metadata["role_split"] is False
    assert metadata["classes"][0] == "background"
    assert metadata["onnx_sha256"] == hashlib.sha256(onnx_path.read_bytes()).hexdigest()


def test_part_segmenter_returns_valid_mask_for_sample():
    import numpy as np
    from PIL import Image

    from vehicle_damage.parts import PartSegmenter

    selection = json.loads(
        (ROOT / "runs" / "SELECTED_DEVELOPMENT_MODEL.json").read_text(encoding="utf-8")
    )
    segmenter = PartSegmenter(ROOT / selection["artifacts"]["parts_model"]["path"])
    image = Image.open(ROOT / "samples" / "damage" / "damaged.jpg").convert("RGB")
    mask = segmenter.predict(image)
    assert mask.shape == (image.height, image.width)
    assert mask.max() < len(segmenter.part_names)
    assert (mask > 0).mean() > 0.05  # a car photo should contain recognised parts
