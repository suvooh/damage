import pytest
import torch

from vehicle_damage.model import (
    Dinov2DamageSegmenter,
    damage_probabilities,
    load_checkpoint,
)


def test_dinov2_model_shapes_for_non_multiple_input():
    model = Dinov2DamageSegmenter(6).eval()
    output = model(torch.randn(1, 3, 57, 71))
    assert output["presence"].shape == (1, 1, 57, 71)
    assert output["type"].shape == (1, 5, 57, 71)
    assert output["exterior"].shape == (1, 1, 57, 71)
    probability = damage_probabilities(output)
    assert probability.shape == (1, 6, 57, 71)
    assert torch.allclose(probability.sum(dim=1), torch.ones(1, 57, 71), atol=1e-5)


def test_noncommercial_dinov2_variants_are_rejected():
    with pytest.raises(ValueError, match="reviewed allowlist"):
        Dinov2DamageSegmenter(6, backbone_name="xray_dinov2")


def test_only_the_selected_architecture_can_be_loaded(tmp_path):
    checkpoint = {
        "architecture": "deeplabv3_resnet50",
        "classes": ["background", "dent"],
        "model": {},
    }
    path = tmp_path / "other.pt"
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="only serves 'dinov2'"):
        load_checkpoint(str(path))


def test_selected_checkpoint_loads_and_scores_a_tile():
    from vehicle_damage.api import selected_artifacts

    artifacts = selected_artifacts()
    model, checkpoint = load_checkpoint(str(artifacts["checkpoint"]), "cpu")
    assert checkpoint["architecture"] == "dinov2"
    assert list(checkpoint["classes"])[0] == "background"
    with torch.inference_mode():
        output = model(torch.randn(1, 3, 56, 56))
    assert output["presence"].shape == (1, 1, 56, 56)
