import io
import json

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from vehicle_damage.api import _components, create_app, selected_artifacts


class _FakePredictor:
    def health(self):
        return {
            "ok": True,
            "model_loaded": True,
            "weights_exist": True,
            "cuda": False,
            "backend": "onnx",
            "device": "onnx:cpu:fp32",
            "production_approved": False,
            "triage_threshold": 0.735,
            "segmentation_threshold": 0.735,
        }

    def predict_bytes(self, payload):
        assert payload
        return {
            "decision": "manual_review_required",
            "production_approved": False,
            "triage_alert": True,
            "high_confidence_alert": False,
            "detections": [],
            "review_candidates": [],
            "n_detections": 0,
            "runtime": {"backend": "onnx", "device": "onnx:cpu:fp32", "elapsed_ms": 1.0},
        }


def test_selected_frontend_artifacts_are_hash_verified():
    artifacts = selected_artifacts()
    assert artifacts["checkpoint"].name == "best.pt"
    assert artifacts["development_profile"].suffix == ".json"
    assert artifacts["selection"]["production_approved"] is False


def test_damage_api_health_upload_and_media_type_contract():
    app = create_app(lambda: _FakePredictor())
    with TestClient(app) as client:
        health = client.get("/api/health")
        assert health.status_code == 200
        assert health.json()["ok"] is True
        assert health.json()["damage"]["model_loaded"] is True
        assert "telemetry" not in health.json()

        rejected = client.post(
            "/api/damage/predict",
            files={"file": ("car.txt", b"not an image", "text/plain")},
        )
        assert rejected.status_code == 415

        buffer = io.BytesIO()
        Image.new("RGB", (32, 32), "gray").save(buffer, format="PNG")
        response = client.post(
            "/api/damage/predict",
            files={"file": ("car.png", buffer.getvalue(), "image/png")},
        )
        assert response.status_code == 200
        assert response.json()["decision"] == "manual_review_required"
        assert response.json()["source"] == "upload:car.png"


def test_bundled_samples_and_sample_prediction():
    app = create_app(lambda: _FakePredictor())
    with TestClient(app) as client:
        samples = client.get("/api/damage/samples").json()
        ids = {row["id"] for row in samples}
        assert {"damaged", "clean", "silver_quarter"} <= ids

        image = client.get("/api/damage/file/sample/damaged")
        assert image.status_code == 200
        assert image.headers["content-type"] == "image/jpeg"
        assert image.content

        report = client.get("/api/damage/predict/sample/damaged")
        assert report.status_code == 200
        assert report.json()["sample_id"] == "damaged"
        assert report.json()["source"] == "sample:damaged"


def test_selected_artifact_integrity_failure_is_explicit(tmp_path):
    selected = selected_artifacts()["selection"]
    tampered = json.loads(json.dumps(selected))
    tampered["artifacts"]["checkpoint"]["sha256"] = "0" * 64
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(tampered), encoding="utf-8")
    try:
        selected_artifacts(path)
    except ValueError as exc:
        assert "pinned integrity check" in str(exc)
    else:
        raise AssertionError("a mismatched selected checkpoint hash must fail closed")


def test_component_table_groups_connected_mixed_class_regions():
    mask = np.zeros((12, 12), dtype=np.uint8)
    mask[2:8, 2:8] = 1
    mask[4:6, 4:6] = 2
    rows = _components(
        mask,
        np.full(mask.shape, 0.8, dtype=np.float32),
        ["background", "dent", "scratch"],
        kind="segmentation",
        minimum_pixels=16,
    )
    assert len(rows) == 1
    assert rows[0]["class_name"] == "dent"
    assert rows[0]["area_pixels"] == 36


def test_frontend_page_and_api_info_are_served():
    with TestClient(create_app(lambda: _FakePredictor())) as client:
        page = client.get("/")
        assert page.status_code == 200
        assert "text/html" in page.headers["content-type"]
        assert "Vehicle Damage Scan" in page.text
        assert client.get("/api").json()["docs"] == "/docs"


def test_components_rate_against_panel_area_and_summary_groups_by_panel():
    from vehicle_damage.api import _panel_summary, _render_layers

    names = ["background", "hood", "front-door"]
    part_mask = np.zeros((100, 100), dtype=np.uint8)
    part_mask[:, :50] = 1  # hood: 5000 px
    part_mask[:, 50:] = 2  # front-door: 5000 px
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10:30, 10:30] = 1  # 400 px dent on the hood = 8% of the panel
    score = np.full((100, 100), 0.9, dtype=np.float32)
    detections = _components(
        mask, score, ["background", "dent"], kind="segmentation", minimum_pixels=1,
        part_mask=part_mask, part_names=names,
    )
    assert len(detections) == 1
    det = detections[0]
    assert det["part"] == "bonnet"
    assert abs(det["panel_area_pct"] - 8.0) < 1e-6
    assert det["severity"]["area_basis"] == "panel"
    assert det["severity"]["score"] == 2  # assumed d2, 0-20% column
    panels = _panel_summary(detections, mask, part_mask, names, ["background", "dent"])
    assert [p["part"] for p in panels] == ["bonnet"]
    assert abs(panels[0]["damage_area_pct_of_panel"] - 8.0) < 1e-6
    assert panels[0]["damage"][0]["class_name"] == "dent"
    layers = _render_layers(Image.new("RGB", (100, 100)), mask, part_mask, panels)
    assert {"image_jpeg_b64", "damage_overlay_png_b64", "panel_overlay_png_b64"} <= layers.keys()


def test_components_fall_back_to_vehicle_area_without_part_model():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[0:50, 0:50] = 1
    score = np.full((100, 100), 0.9, dtype=np.float32)
    det = _components(
        mask, score, ["background", "scratch"], kind="segmentation", minimum_pixels=1,
        vehicle_pixels=5000,
    )[0]
    assert det["part"] is None
    assert det["severity"]["area_basis"] == "vehicle"


def test_visible_parts_reports_panels_above_minimum_share():
    from vehicle_damage.api import _visible_parts

    part_mask = np.zeros((100, 100), dtype=np.uint8)
    part_mask[:, :50] = 1
    part_mask[0, 0:2] = 2  # 2 px of 10,000 = 0.02% -> below the minimum share
    visible = _visible_parts(part_mask, ["background", "hood", "mirror"])
    assert [v["part"] for v in visible] == ["bonnet"]
    assert abs(visible[0]["area_pct_of_image"] - 49.98) < 0.1
    assert _visible_parts(None, None) == []


def test_heatmap_car_image_is_served_as_a_static_file():
    with TestClient(create_app(lambda: _FakePredictor())) as client:
        image = client.get("/static/car_topdown.png")
        assert image.status_code == 200
        assert image.headers["content-type"] == "image/png"
        assert image.content[:8] == b"\x89PNG\r\n\x1a\n"


def test_panel_rates_each_type_on_its_total_area_and_keeps_types_separate():
    from vehicle_damage.api import _panel_summary

    names = ["background", "hood"]
    classes = ["background", "dent", "scratch", "crack_or_breakage"]
    part_mask = np.ones((100, 100), dtype=np.uint8)  # one panel = whole image, 10,000 px
    mask = np.zeros((100, 100), dtype=np.uint8)
    for i in range(3):  # three scratch fragments of 6% each = 18% total (still < 20%)
        mask[10 * i : 10 * i + 6, 0:100] = 2
    mask[40:50, 0:100] = 2  # a fourth fragment pushes the scratch union to 28%
    mask[60:70, 0:30] = 3  # crack: 3% of the panel
    detections = _components(
        mask, np.full((100, 100), 0.9, np.float32), classes, kind="segmentation",
        minimum_pixels=1, part_mask=part_mask, part_names=names,
    )
    panel = _panel_summary(detections, mask, part_mask, names, classes)[0]
    by_type = {d["class_name"]: d for d in panel["damage"]}
    assert by_type["scratch"]["severity"]["score"] == 3  # union 28% -> rubric 20-100% column
    assert by_type["crack_or_breakage"]["severity"]["score"] == 3  # 3% on the d3 row
    assert panel["rating"] == 3  # no flat 5 for a small crack


def test_startup_runs_predictor_warmup_and_survives_a_failing_one():
    calls = []

    class Warm(_FakePredictor):
        def warmup(self):
            calls.append(1)

    class Broken(_FakePredictor):
        def warmup(self):
            raise RuntimeError("cold start failed")

    with TestClient(create_app(lambda: Warm())) as client:
        assert client.get("/api/health").json()["ok"] is True
    assert calls == [1]
    with TestClient(create_app(lambda: Broken())) as client:
        assert client.get("/api/health").json()["ok"] is True
