import numpy as np
import pytest

from vehicle_damage.severity import DENT_TABLE, assign_part, severity_for


def test_rubric_worked_example_d2_dent_at_30_percent_is_3():
    assert severity_for("dent", 0.30, depth_class="d2")["score"] == 3


@pytest.mark.parametrize("depth", ["d1", "d2", "d3"])
@pytest.mark.parametrize("bucket", range(5))
def test_dent_table_matches_supplied_rubric(depth, bucket):
    fraction = bucket * 0.2 + 0.1  # middle of each 20% column
    assert severity_for("dent", fraction, depth_class=depth)["score"] == DENT_TABLE[depth][bucket]


def test_dent_table_values():
    assert DENT_TABLE == {"d1": (1, 2, 3, 4, 4), "d2": (2, 3, 4, 5, 5), "d3": (3, 4, 5, 5, 5)}


def test_column_boundaries_and_full_panel():
    assert severity_for("dent", 0.199, depth_class="d1")["score"] == 1
    assert severity_for("dent", 0.20, depth_class="d1")["score"] == 2
    assert severity_for("dent", 1.0, depth_class="d1")["score"] == 4


def test_scratch_rule():
    assert severity_for("scratch", 0.05)["score"] == 1
    assert severity_for("scratch", 0.199)["score"] == 1
    assert severity_for("scratch", 0.20)["score"] == 3
    assert severity_for("scratch", 1.0)["score"] == 3


def test_missing_depth_is_assumed_d2_and_flagged():
    result = severity_for("dent", 0.30)
    assert result["depth_class"] == "d2"
    assert result["depth_assumed"] is True
    assert severity_for("dent", 0.30, depth_class="d2")["depth_assumed"] is False


def test_crack_and_deformation_scale_with_panel_area_on_the_d3_row():
    assert severity_for("crack_or_breakage", 0.01)["score"] == 3
    assert severity_for("crack_or_breakage", 0.30)["score"] == 4
    assert severity_for("crack_or_breakage", 0.50)["score"] == 5
    assert severity_for("deformation_or_detachment", 0.10)["score"] == 3
    assert severity_for("deformation_or_detachment", 0.10)["basis"].startswith("extension")


def test_paint_is_rated_like_a_scratch_and_never_exceeds_3():
    assert severity_for("paint_damage", 0.5)["score"] == 3
    assert severity_for("paint_damage", 0.05)["score"] == 1
    assert severity_for("scratch", 1.0)["score"] == 3
    assert severity_for("scratch", 0.5)["basis"] == "rubric"


def test_severity_rejects_bad_input():
    with pytest.raises(ValueError):
        severity_for("mystery", 0.1)
    with pytest.raises(ValueError):
        severity_for("dent", float("nan"))
    with pytest.raises(ValueError):
        severity_for("dent", 0.1, depth_class="d9")


def test_assign_part_reports_panel_and_damage_pixels():
    names = ["background", "hood", "mirror"]
    part_mask = np.zeros((10, 10), dtype=np.uint8)
    part_mask[:, :6] = 1
    region = np.zeros((10, 10), dtype=bool)
    region[2:4, 1:3] = True
    found = assign_part(region, part_mask, names)
    assert found["name"] == "bonnet"
    assert found["id"] == 1
    assert found["damage_pixels"] == 4
    assert found["panel_pixels"] == 60


def test_assign_part_ignores_background_and_empty():
    names = ["background", "hood"]
    part_mask = np.zeros((10, 10), dtype=np.uint8)
    part_mask[:, :3] = 1
    region = np.zeros((10, 10), dtype=bool)
    region[0:2, 8:10] = True
    assert assign_part(region, part_mask, names) is None
    assert assign_part(np.zeros((10, 10), dtype=bool), part_mask, names) is None
