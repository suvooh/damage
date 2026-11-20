"""Damage-only FastAPI service for the pinned DINOv2 segmentation model.

The service intentionally contains no telemetry, OCR, ETA, LLM, ERP, database,
or other platform features. It exposes health, sample images, and one prediction
endpoint backed by the frozen checkpoint and calibration profile.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import asdict, replace
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError
from scipy import ndimage

from .calibration import Calibration
from .inference import (
    assess_quality,
    predict_profile_probabilities,
    probabilities_to_mask,
    glare_zone,
    restore_size,
    route_inference_decision,
    split_glare_components,
    working_image,
)
from .model import load_checkpoint
from .onnx_backend import load_onnx_model
from .parts import PartSegmenter
from .severity import PART_DISPLAY, assign_part, severity_for
from .views import harmonise_ends, infer_view, side_of


_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get("DAMAGE_ROOT") or _PACKAGE_ROOT).resolve()
if not (ROOT / "runs" / "SELECTED_DEVELOPMENT_MODEL.json").is_file():
    # A non-editable install places the package in site-packages; prefer the
    # current working directory when it is a complete damage project.
    _cwd = Path.cwd().resolve()
    if (_cwd / "runs" / "SELECTED_DEVELOPMENT_MODEL.json").is_file():
        ROOT = _cwd
SELECTED_MODEL = ROOT / "runs" / "SELECTED_DEVELOPMENT_MODEL.json"
SAMPLES_DIR = ROOT / "samples" / "damage"
STATIC_DIR = Path(__file__).resolve().parent / "static"
MAX_UPLOAD_BYTES = 20_000_000
MAX_IMAGE_PIXELS = 50_000_000
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
CLASS_COLORS = (
    (234, 88, 12),
    (2, 132, 199),
    (220, 38, 38),
    (202, 138, 4),
    (147, 51, 234),
    (5, 150, 105),
)

SAMPLES: dict[str, tuple[str, str, bool]] = {
    "damaged": ("damaged.jpg", "Known damaged (smashed front)", True),
    "clean": ("clean.jpg", "Undamaged body lines / paint", False),
    "silver_quarter": ("user_dents/silver_quarter.jpg", "Close-up dent (silver quarter)", True),
    "torn_bumper": ("user_dents/torn_bumper.jpg", "Torn bumper / broken part", True),
    "blue_fender": ("user_dents/blue_fender.jpg", "Close-up dent (blue fender)", True),
    "taillight_dent": ("user_dents/taillight_dent.jpg", "Bumper dent by taillight", True),
    "black_mahindra": ("showroom/black_mahindra.jpg", "Showroom (black Mahindra)", False),
    "white_scorpio": ("showroom/white_scorpio.jpg", "Showroom (white Scorpio)", False),
    "white_bmw": ("showroom/white_bmw.jpg", "Intact listing (white sedan)", False),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def selected_artifacts(selection_path: str | Path = SELECTED_MODEL) -> dict[str, object]:
    """Load and hash-verify the selected checkpoint and calibration profile."""
    path = Path(selection_path).resolve()
    selection = json.loads(path.read_text(encoding="utf-8"))
    if (
        selection.get("status") != "development_only_not_production_approved"
        or selection.get("production_approved") is not False
    ):
        raise ValueError("selected model must retain its development-only safety status")
    result: dict[str, object] = {"selection": selection, "selection_path": path}
    for key in ("checkpoint", "development_profile"):
        declared = selection["artifacts"][key]
        artifact = (ROOT / declared["path"]).resolve()
        try:
            artifact.relative_to(ROOT)
        except ValueError as exc:
            raise ValueError(f"selected {key} path escapes the project root") from exc
        if not artifact.is_file():
            raise ValueError(f"missing selected {key}: {artifact}")
        if artifact.stat().st_size != declared["bytes"] or _sha256(artifact) != declared["sha256"]:
            raise ValueError(f"selected {key} failed its pinned integrity check")
        result[key] = artifact
    return result


def _score_map(
    damage_probability: torch.Tensor,
    exterior_probability: torch.Tensor,
    exterior_floor: float,
) -> np.ndarray:
    return (
        (1.0 - damage_probability[0])
        * (exterior_floor + (1.0 - exterior_floor) * exterior_probability)
    ).numpy()


def _components(
    mask: np.ndarray,
    score: np.ndarray,
    classes: list[str],
    *,
    kind: str,
    minimum_pixels: int,
    maximum_components: int = 100,
    vehicle_pixels: int | None = None,
    part_mask: np.ndarray | None = None,
    part_names: list[str] | None = None,
) -> list[dict[str, object]]:
    height, width = mask.shape
    records: list[dict[str, object]] = []
    labels, count = ndimage.label(
        mask > 0, structure=np.ones((3, 3), dtype=np.uint8)
    )
    for component_id in range(1, count + 1):
        region = labels == component_id
        area = int(region.sum())
        if area < minimum_pixels:
            continue
        class_counts = np.bincount(mask[region], minlength=len(classes))
        class_id = int(class_counts[1:].argmax()) + 1
        ys, xs = np.where(region)
        part = (
            assign_part(region, part_mask, part_names)
            if part_mask is not None and part_names
            else None
        )
        if part:
            fraction = part["damage_pixels"] / max(1, part["panel_pixels"])
            basis = "panel"
        else:
            fraction = area / max(1, vehicle_pixels or height * width)
            basis = "vehicle"
        records.append(
            {
                "class_name": classes[class_id],
                "part": part["name"] if part else None,
                "part_id": part["id"] if part else None,
                "part_overlap": part["overlap"] if part else None,
                "panel_area_pct": float(min(1.0, fraction) * 100.0),
                "severity": severity_for(
                    classes[class_id], min(1.0, fraction), area_basis=basis
                ),
                "score": float(score[region].mean()),
                "bbox": [
                    int(xs.min()),
                    int(ys.min()),
                    int(xs.max()) + 1,
                    int(ys.max()) + 1,
                ],
                "area_pixels": area,
                "area_frac": area / (height * width),
                "kind": kind,
            }
        )
    return sorted(records, key=lambda row: float(row["score"]), reverse=True)[
        :maximum_components
    ]


def _annotated_jpeg(image: Image.Image, mask: np.ndarray, review_mask: np.ndarray) -> str:
    canvas = image.convert("RGBA")
    overlay = np.zeros((image.height, image.width, 4), dtype=np.uint8)
    for class_id in np.unique(mask):
        if not class_id:
            continue
        color = CLASS_COLORS[(int(class_id) - 1) % len(CLASS_COLORS)]
        overlay[mask == class_id] = (*color, 122)
    overlay[(review_mask > 0) & (mask == 0)] = (245, 158, 11, 105)
    rendered = Image.alpha_composite(canvas, Image.fromarray(overlay, mode="RGBA"))
    rendered.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    rendered.convert("RGB").save(buffer, format="JPEG", quality=88, optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


SEVERITY_RGB = {
    1: (46, 158, 107),
    2: (140, 174, 45),
    3: (217, 164, 0),
    4: (232, 113, 10),
    5: (201, 48, 44),
}
LAYER_MAX_SIDE = 1600


def _layer_size(image: Image.Image) -> tuple[int, int]:
    scale = min(1.0, LAYER_MAX_SIDE / max(image.size))
    return max(1, round(image.width * scale)), max(1, round(image.height * scale))


def _png_b64(rgba: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buffer, format="PNG", optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _resize_ids(ids: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return np.asarray(Image.fromarray(ids.astype(np.uint8)).resize(size, Image.Resampling.NEAREST))


def _render_layers(
    image: Image.Image,
    mask: np.ndarray,
    part_mask: np.ndarray | None,
    panels: list[dict[str, object]],
) -> dict[str, str]:
    """Plain photo + transparent overlays so the UI can toggle layers."""
    size = _layer_size(image)
    photo = io.BytesIO()
    image.convert("RGB").resize(size, Image.Resampling.LANCZOS).save(
        photo, format="JPEG", quality=90, optimize=True
    )
    layers = {"image_jpeg_b64": base64.b64encode(photo.getvalue()).decode("ascii")}

    small_mask = _resize_ids(mask, size)
    damage = np.zeros((size[1], size[0], 4), dtype=np.uint8)
    for class_id in np.unique(small_mask):
        if class_id:
            damage[small_mask == class_id] = (*CLASS_COLORS[(int(class_id) - 1) % len(CLASS_COLORS)], 150)
    layers["damage_overlay_png_b64"] = _png_b64(damage)

    panel_layer = np.zeros((size[1], size[0], 4), dtype=np.uint8)
    if part_mask is not None:
        small_parts = _resize_ids(part_mask, size)
        for panel in panels:
            color = SEVERITY_RGB[int(panel["rating"])]
            region = small_parts == int(panel["part_id"])
            edge = region & ~ndimage.binary_erosion(region, iterations=3)
            panel_layer[region] = (*color, 85)
            panel_layer[edge] = (*color, 255)
    layers["panel_overlay_png_b64"] = _png_b64(panel_layer)
    layers["layer_size"] = {"width": size[0], "height": size[1]}
    return layers


def _visible_parts(
    part_mask: np.ndarray | None,
    part_names: list[str] | None,
    *,
    minimum_fraction: float = 0.004,
) -> list[dict[str, object]]:
    """Panels that occupy a meaningful share of the photo (damaged or not)."""
    if part_mask is None or not part_names:
        return []
    from .severity import PART_DISPLAY

    counts = np.bincount(part_mask.ravel(), minlength=len(part_names))
    total = max(1, part_mask.size)
    return [
        {
            "part": PART_DISPLAY.get(part_names[i], part_names[i]),
            "part_id": int(i),
            "area_pct_of_image": 100.0 * float(counts[i]) / total,
        }
        for i in range(1, len(part_names))
        if counts[i] / total >= minimum_fraction
    ]


def _panel_summary(
    detections: list[dict[str, object]],
    mask: np.ndarray,
    part_mask: np.ndarray | None,
    part_names: list[str] | None,
    classes: list[str],
) -> list[dict[str, object]]:
    """One row per damaged panel with a rubric rating per damage type.

    The rubric rates damage area as a share of the panel, so each type's area is
    the union of all its fragments inside the panel, not each fragment alone.
    """
    if part_mask is None or not part_names:
        return []
    from .severity import PART_DISPLAY

    panels: list[dict[str, object]] = []
    for part_id in sorted({d["part_id"] for d in detections if d.get("part_id")}):
        in_panel = part_mask == part_id
        panel_pixels = max(1, int(in_panel.sum()))
        damage: list[dict[str, object]] = []
        for class_id in np.unique(mask[in_panel & (mask > 0)]):
            pixels = int(((mask == class_id) & in_panel).sum())
            fraction = min(1.0, pixels / panel_pixels)
            damage.append(
                {
                    "class_name": classes[int(class_id)],
                    "area_pct_of_panel": 100.0 * fraction,
                    "severity": severity_for(classes[int(class_id)], fraction, area_basis="panel"),
                }
            )
        if not damage:
            continue
        damage.sort(key=lambda d: (-d["severity"]["score"], -d["area_pct_of_panel"]))
        ys, xs = np.where(in_panel)
        worst = damage[0]["severity"]
        panels.append(
            {
                "part": PART_DISPLAY.get(part_names[part_id], part_names[part_id]),
                "part_id": int(part_id),
                "damage": damage,
                "damage_types": [d["class_name"] for d in damage],
                "damage_count": sum(1 for d in detections if d.get("part_id") == part_id),
                "damage_area_pct_of_panel": 100.0 * float((mask[in_panel] > 0).sum()) / panel_pixels,
                "rating": int(worst["score"]),
                "rating_label": worst["label"],
                "rating_reason": worst["reason"],
                "depth_assumed": any(d["severity"]["depth_assumed"] for d in damage),
                "bbox": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
            }
        )
    return sorted(panels, key=lambda p: (-p["rating"], -p["damage_area_pct_of_panel"]))


class DamagePredictor:
    """Thread-safe wrapper around the pinned PyTorch or ONNX damage model."""

    def __init__(self, *, device: str | None = None) -> None:
        artifacts = selected_artifacts(
            os.environ.get("VEHICLE_DAMAGE_SELECTION") or SELECTED_MODEL
        )
        self.selection = artifacts["selection"]
        self.checkpoint_path = Path(artifacts["checkpoint"])
        self.profile_path = Path(artifacts["development_profile"])
        self.calibration = Calibration.load(self.profile_path)
        requested = (device or os.environ.get("VEHICLE_DAMAGE_DEVICE") or "auto").lower()
        backend = os.environ.get("VEHICLE_DAMAGE_BACKEND", "auto").lower()
        if backend not in {"auto", "torch", "onnx"}:
            raise ValueError("VEHICLE_DAMAGE_BACKEND must be 'auto', 'torch', or 'onnx'")
        if backend == "auto":
            backend = (
                "torch"
                if requested == "cuda"
                or (requested == "auto" and torch.cuda.is_available())
                else "onnx"
            )
        self.backend = backend
        if backend == "torch":
            self.device = torch.device("cuda" if requested == "auto" else requested)
            if self.calibration.mixed_precision and self.device.type != "cuda":
                raise RuntimeError(
                    "the selected profile requires CUDA mixed-precision under PyTorch; "
                    "use VEHICLE_DAMAGE_BACKEND=onnx for CPU/GPU/NPU inference"
                )
            self.model, checkpoint = load_checkpoint(str(self.checkpoint_path), self.device)
            self.classes = list(checkpoint["classes"])
            self.accelerator = str(self.device)
        else:
            self.device = torch.device("cpu")
            self.model = load_onnx_model(
                self.checkpoint_path,
                tile_size=self.calibration.tile_size,
                device=requested,
                checkpoint_sha256=self.selection["artifacts"]["checkpoint"]["sha256"],
            )
            self.classes = self.model.classes
            self.accelerator = self.model.description
        self.parts: PartSegmenter | None = None
        declared = self.selection["artifacts"].get("parts_model")
        if declared:
            part_path = (ROOT / declared["path"]).resolve()
            part_path.relative_to(ROOT)
            if part_path.is_file() and _sha256(part_path) == declared["sha256"]:
                self.parts = PartSegmenter(part_path)
        self.lock = threading.Lock()

    def warmup(self) -> float:
        """Run one synthetic photo so the first real request is not the cold one."""
        buffer = io.BytesIO()
        Image.new("RGB", (768, 768), (128, 128, 128)).save(buffer, format="JPEG")
        started = time.perf_counter()
        self.predict_bytes(buffer.getvalue())
        return time.perf_counter() - started

    def health(self) -> dict[str, object]:
        return {
            "ok": True,
            "model_loaded": True,
            "weights_exist": True,
            "cuda": self.device.type == "cuda",
            "backend": self.backend,
            "device": self.accelerator,
            "model_status": self.selection["status"],
            "production_approved": False,
            "checkpoint_sha256": self.selection["artifacts"]["checkpoint"]["sha256"],
            "profile_sha256": self.selection["artifacts"]["development_profile"]["sha256"],
            "triage_threshold": self.calibration.any_damage_threshold,
            "segmentation_threshold": self.calibration.effective_segmentation_threshold,
        }

    def predict_bytes(self, payload: bytes) -> dict[str, object]:
        started = time.perf_counter()
        if not payload:
            raise ValueError("uploaded image is empty")
        if len(payload) > MAX_UPLOAD_BYTES:
            raise ValueError(f"uploaded image exceeds {MAX_UPLOAD_BYTES} bytes")
        try:
            with Image.open(io.BytesIO(payload)) as source:
                if source.width * source.height > MAX_IMAGE_PIXELS:
                    raise ValueError(f"image exceeds {MAX_IMAGE_PIXELS} pixels")
                image = ImageOps.exif_transpose(source).convert("RGB")
                image.load()
        except (OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
            raise ValueError("upload is not a supported safe image") from exc
        quality = assess_quality(image)
        work = working_image(image)
        with self.lock:
            triage_maps, mask_maps = (
                predict_profile_probabilities(
                    self.model,
                    work,
                    device=self.device,
                    triage_scales=self.calibration.triage_scales,
                    segmentation_scales=self.calibration.effective_segmentation_scales,
                    tile_size=self.calibration.tile_size,
                    overlap=self.calibration.overlap,
                    horizontal_flip_tta=self.calibration.horizontal_flip_tta,
                    mixed_precision=self.calibration.mixed_precision,
                )
            )
        (triage_damage, triage_exterior) = restore_size(triage_maps, image.size)
        (mask_damage, mask_exterior) = restore_size(mask_maps, image.size)
        triage_mask, uncertainty = probabilities_to_mask(
            triage_damage,
            triage_exterior,
            self.calibration.any_damage_threshold,
            self.calibration.exterior_floor,
            self.calibration.minimum_component_pixels,
            self.calibration.type_probability_multipliers,
        )
        mask, _ = probabilities_to_mask(
            mask_damage,
            mask_exterior,
            self.calibration.effective_segmentation_threshold,
            self.calibration.exterior_floor,
            self.calibration.effective_segmentation_minimum_component_pixels,
            self.calibration.type_probability_multipliers,
        )
        # Marks sitting mostly on lamps, LED strips or specular glare are not trusted as
        # damage: they are moved to review so they can neither pass as damage nor vanish.
        zone = glare_zone(image)
        triage_mask, triage_glare = split_glare_components(triage_mask, zone)
        mask, mask_glare = split_glare_components(mask, zone)
        glare_marks = np.where(mask_glare > 0, mask_glare, triage_glare).astype(np.uint8)
        review_mask = np.where((triage_mask > 0) & (mask == 0), triage_mask, 0).astype(
            np.uint8
        )
        review_mask = np.where(glare_marks > 0, glare_marks, review_mask).astype(np.uint8)
        near_threshold = float((uncertainty < 0.03).mean())
        decision = route_inference_decision(
            quality,
            triage_mask,
            mask,
            near_threshold_fraction=near_threshold,
            max_uncertain_fraction=0.10,
        )
        if glare_marks.any() and decision.decision in {"damage_detected", "no_damage_detected"}:
            decision = replace(
                decision,
                decision="manual_review_required",
                automated_decision_allowed=False,
                manual_review=True,
                manual_review_required=True,
                decision_reasons=(*decision.decision_reasons, "damage_like_marks_on_light_or_glare"),
            )
        elif glare_marks.any():
            decision = replace(
                decision,
                decision_reasons=(*decision.decision_reasons, "damage_like_marks_on_light_or_glare"),
            )
        triage_score = _score_map(
            triage_damage, triage_exterior, self.calibration.exterior_floor
        )
        mask_score = _score_map(
            mask_damage, mask_exterior, self.calibration.exterior_floor
        )
        vehicle_pixels = max(1, int((mask_exterior >= 0.5).sum().item()))
        part_mask = self.parts.predict(image) if self.parts else None
        part_names_for_fix = self.parts.part_names if self.parts else None
        part_mask = harmonise_ends(part_mask, part_names_for_fix)
        part_names = self.parts.part_names if self.parts else None
        detections = _components(
            mask,
            mask_score,
            self.classes,
            kind="segmentation",
            minimum_pixels=max(
                1, self.calibration.effective_segmentation_minimum_component_pixels
            ),
            vehicle_pixels=vehicle_pixels,
            part_mask=part_mask,
            part_names=part_names,
        )
        review_candidates = _components(
            review_mask,
            triage_score,
            self.classes,
            kind="triage_only",
            minimum_pixels=max(16, self.calibration.minimum_component_pixels),
            vehicle_pixels=vehicle_pixels,
            part_mask=part_mask,
            part_names=part_names,
        )
        reason = {
            "damage_detected": "Damage evidence passed both calibrated branches.",
            "no_damage_detected": "No damage evidence passed the frozen thresholds.",
            "manual_review_required": "The result is uncertain and requires human review.",
            "recapture_required": "Image quality is insufficient; capture another photograph.",
        }[decision.decision]
        annotated_jpeg_b64 = _annotated_jpeg(image, mask, review_mask)
        view = infer_view(part_mask, part_names)
        for record in (*detections, *review_candidates):
            x0, _, x1, _ = record["bbox"]
            record["side"] = side_of(record.get("part"), view, (x0 + x1) / 2 / image.width)
        panels = _panel_summary(detections, mask, part_mask, part_names, self.classes)
        for panel in panels:
            sides = [d["side"] for d in detections if d.get("part_id") == panel["part_id"] and d.get("side")]
            panel["side"] = max(set(sides), key=sides.count) if sides else None
        layers = _render_layers(image, mask, part_mask, panels)
        return {
            "model_status": self.selection["status"],
            "production_approved": False,
            "decision": decision.decision,
            "automated_decision_allowed": decision.automated_decision_allowed,
            "manual_review": decision.manual_review,
            "manual_review_required": decision.manual_review_required,
            "recapture_required": decision.recapture_required,
            "decision_reasons": list(decision.decision_reasons),
            "recapture_reasons": list(decision.recapture_reasons),
            "reason": reason,
            "damage_present": bool(np.any(triage_mask)),
            "triage_alert": bool(np.any(triage_mask)),
            "high_confidence_alert": bool(np.any(mask)),
            "quality": asdict(quality),
            "detections": detections,
            "overall_severity": max(
                (d["severity"]["score"] for d in detections), default=0
            ),
            "part_model": "dinov2-parts-v1" if self.parts else "unavailable",
            "panels": panels,
            "view": view,
            "parts_visible": _visible_parts(part_mask, part_names),
            "part_catalogue": [
                PART_DISPLAY[n] for n in (part_names or []) if n in PART_DISPLAY
            ],
            "layers": layers,
            "review_candidates": review_candidates,
            "n_detections": len(detections),
            "annotated_jpeg_b64": annotated_jpeg_b64,
            "image": {"width": image.width, "height": image.height},
            "runtime": {
                "backend": self.backend,
                "device": self.accelerator,
                "elapsed_ms": (time.perf_counter() - started) * 1000.0,
            },
            "thresholds": {
                "triage": self.calibration.any_damage_threshold,
                "segmentation": self.calibration.effective_segmentation_threshold,
            },
        }


def _sample_path(sample_id: str) -> Path:
    item = SAMPLES.get(sample_id)
    if item is None:
        raise HTTPException(404, "unknown sample")
    path = (SAMPLES_DIR / item[0]).resolve()
    try:
        path.relative_to(SAMPLES_DIR.resolve())
    except ValueError as exc:
        raise HTTPException(404, "sample missing on disk") from exc
    if not path.is_file():
        raise HTTPException(404, "sample missing on disk")
    return path


def create_app(
    predictor_factory: Callable[[], DamagePredictor] = DamagePredictor,
) -> FastAPI:
    """Build the damage-only API; the factory is injectable for tests."""
    state: dict[str, object] = {"predictor": None, "error": None}

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            state["predictor"] = predictor_factory()
            warmup = getattr(state["predictor"], "warmup", None)
            if warmup is not None:
                try:
                    await run_in_threadpool(warmup)
                except Exception:  # a failed warm-up must not take the service down
                    pass
        except Exception as exc:  # health must explain startup failures
            state["error"] = str(exc)
        yield

    app = FastAPI(
        title="Vehicle Damage Detection API",
        version="1.0.0",
        description=(
            "DINOv2 exterior damage segmentation. Development-only checkpoint; "
            "not production approved."
        ),
        lifespan=lifespan,
    )

    def health_payload() -> dict[str, object]:
        predictor = state["predictor"]
        if predictor is None:
            return {
                "ok": False,
                "damage": {
                    "ok": False,
                    "model_loaded": False,
                    "weights_exist": SELECTED_MODEL.is_file(),
                    "error": state["error"],
                },
            }
        payload = predictor.health()
        return {"ok": bool(payload.get("ok")), "damage": payload}

    def require_predictor() -> DamagePredictor:
        predictor = state["predictor"]
        if predictor is None:
            raise HTTPException(
                status_code=503,
                detail=str(state["error"] or "damage model is unavailable"),
            )
        return predictor  # type: ignore[return-value]

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html", media_type="text/html")

    @app.get("/api")
    def root() -> dict[str, object]:
        return {
            "service": "vehicle damage detection",
            "status": "development_only_not_production_approved",
            "docs": "/docs",
            "health": "/api/health",
        }

    @app.get("/api/health")
    def health() -> dict[str, object]:
        return health_payload()

    @app.get("/api/damage/health")
    def damage_health() -> dict[str, object]:
        return health_payload()

    @app.get("/api/damage/samples")
    def samples() -> list[dict[str, object]]:
        return [
            {
                "id": key,
                "file": Path(rel).name,
                "label": label,
                "expected_damage": expected_damage,
                "url": f"/api/damage/file/sample/{key}",
            }
            for key, (rel, label, expected_damage) in SAMPLES.items()
            if (SAMPLES_DIR / rel).is_file()
        ]

    @app.get("/api/damage/file/sample/{sample_id}")
    def sample_file(sample_id: str) -> FileResponse:
        return FileResponse(_sample_path(sample_id), media_type="image/jpeg")

    @app.post("/api/damage/predict")
    async def predict(file: UploadFile = File(...)) -> dict[str, object]:
        predictor = require_predictor()
        if file.content_type not in ALLOWED_CONTENT_TYPES:
            raise HTTPException(415, "upload a JPEG, PNG, or WebP image")
        payload = await file.read(MAX_UPLOAD_BYTES + 1)
        try:
            report = await run_in_threadpool(predictor.predict_bytes, payload)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        report["source"] = f"upload:{Path(file.filename or 'upload').name}"
        return report

    @app.get("/api/damage/predict/sample/{sample_id}")
    def predict_sample(sample_id: str) -> dict[str, object]:
        predictor = require_predictor()
        path = _sample_path(sample_id)
        try:
            report = predictor.predict_bytes(path.read_bytes())
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        report["source"] = f"sample:{sample_id}"
        report["sample_id"] = sample_id
        return report

    return app


app = create_app()


def main() -> None:
    uvicorn.run(
        "vehicle_damage.api:app",
        host=os.environ.get("DAMAGE_API_HOST")
        or os.environ.get("VEHICLE_DAMAGE_API_HOST", "127.0.0.1"),
        port=int(
            os.environ.get("DAMAGE_API_PORT")
            or os.environ.get("VEHICLE_DAMAGE_API_PORT", "8001")
        ),
        reload=False,
    )


if __name__ == "__main__":
    main()
