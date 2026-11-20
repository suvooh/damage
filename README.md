# Damage

Standalone vehicle damage detection extracted from the dealer platform
(`prod/backend`, with model provenance from
`Port/vehicle-damage-segmentation`). This project contains **only** the car
damage model: a DINOv2 ViT-S/14 dense segmentation network with its pinned
checkpoint, frozen calibration profile, ONNX runtime artifacts, sample images,
and a damage-only FastAPI service.

Everything else from the source platform was removed: telemetry diagnosis,
OBD-II models, gate OCR, odometer reader, ETA engine, local LLM, ERP/service
journey, SQLite database, mobile app, and dashboard frontend. Other segmentation
architectures (DeepLabV3/ResNet-50, the four-layer and role-split DINOv2
variants) and all training/evaluation/annotation tooling were also removed.
Only the selected architecture can be loaded.

> **Status: development-only.** The served model (`runs/ft_v3_mined_hardneg`)
> is the DINOv2 checkpoint with its last three blocks and heads fine-tuned on
> open-licensed clean car photos as hard negatives, including 537 mined from
> 1,674 Indian-market and reflective-paint photos (false alerts on fresh clean
> photos 14.0% -> 3.3%). Its
> threshold (0.381) comes from a single-scale proxy calibration plus a small
> check in the tiled pipeline; it is not production approved. The previous
> selection is kept in `runs/SELECTED_DEVELOPMENT_MODEL.previous.json` for
> rollback (`...previous_v2.json` for the v2 model). See [MODEL_CARD.md](MODEL_CARD.md).

## Layout

```
damage/
  src/vehicle_damage/
    api.py             damage-only FastAPI service and DamagePredictor
    cli.py             single-image command-line inference
    inference.py       tiled / multi-scale / flip-TTA scoring, quality gate,
                       fail-closed decision routing, component extraction
    model.py           selected DINOv2 architecture + checkpoint loader
    calibration.py     frozen profile reader with validation
    onnx_backend.py    ONNX Runtime backend (CPU / Qualcomm GPU / NPU)
  runs/
    SELECTED_DEVELOPMENT_MODEL.json
    pilot_cc0_dinov2_parts_hardneg_long/
      best.pt                          pinned PyTorch checkpoint
      calibration-fp16-comp1200.json   frozen inference profile
      onnx/                            pinned ONNX exports + metadata
  samples/damage/      bundled inspection images
  tests/               API, checkpoint-integrity, inference, and ONNX tests
```

## Install

Python 3.10+ is required. On Windows PowerShell:

```powershell
cd damage
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

For tests:

```powershell
pip install -r requirements-dev.txt
```

The model artifacts are included (about 260 MB total); no download is needed.
If you install the package rather than running from this directory, use an
editable install so the pinned `runs/` and `samples/` remain discoverable:

```powershell
pip install -e .
```

## Run the API

```powershell
.\run_backend.ps1
```

Then open:

- `http://127.0.0.1:8001/docs` - interactive upload and try-out UI
- `http://127.0.0.1:8001/api/health` - model/backend status
- `http://127.0.0.1:8001/api/damage/samples` - bundled sample list

Equivalent without the script:

```powershell
$env:PYTHONPATH = "src"
python -m vehicle_damage.api
```

### Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | Damage model status, backend, device, thresholds |
| GET | `/api/damage/health` | Same damage status under the damage prefix |
| GET | `/api/damage/samples` | Bundled inspection images and expected labels |
| GET | `/api/damage/file/sample/{id}` | Download one bundled sample image |
| POST | `/api/damage/predict` | Multipart `file` upload (JPEG, PNG, or WebP, <= 20 MB) |
| GET | `/api/damage/predict/sample/{id}` | Run inference on a bundled sample |

Example:

```powershell
curl.exe -X POST http://127.0.0.1:8001/api/damage/predict `
  -F "file=@samples/damage/damaged.jpg"
```

The response contains the fail-closed `decision`
(`damage_detected`, `no_damage_detected`, `manual_review_required`,
`recapture_required`), detections with class/score/bbox/area, review candidates,
quality metrics, thresholds, runtime, and a base64 JPEG overlay in
`annotated_jpeg_b64`.

### Panels, intensity and parts

The web page at `/` shows damaged panels highlighted on the photo with a
per-panel table. Each detection carries `part`, `panel_area_pct` and
`severity` (1-5); `panels` summarises damage per panel and `layers` holds the
photo and transparent overlays. Intensity follows the supplied panel rubric
(dent depth x damage % of panel; scratch under 20% = 1 else 3). Not in the
rubric: paint damage is rated as a scratch, and cracks/breakage/deformation are
rated on the d3 (deepest dent) row by panel area. Each damage type on a panel is
rated on its own total area and the panel shows its worst rating. Dent depth is
not measured yet, so dents are assumed d2 and flagged. The car side of each
panel (left/right as seen from the driver's seat) is inferred from the photo
angle by `views.py`; when the angle is unclear `side` is null. The model often confuses
scratch, paint damage and crack/breakage (see MODEL_CARD.md). See `severity.py`. `tools/` holds the data preparation, part-model, fine-tuning and
probing scripts used to build this.

## Command-line inference

```powershell
$env:PYTHONPATH = "src"
python -m vehicle_damage.cli samples/damage/damaged.jpg --output-dir predictions
```

This writes `predictions/<name>.json` and `predictions/<name>.jpg`. The JSON
omits the inline base64 overlay; the annotated image is written separately.

## Backends

On a Snapdragon (Windows on ARM) machine, install `onnxruntime-qnn` (it is in
`requirements.txt`) and the default `auto` device uses the Adreno GPU: the same
photo took about 24 s on the GPU versus about 284 s on the CPU in one measured
run, with identical output. Without it the app silently falls back to CPU.

`VEHICLE_DAMAGE_BACKEND` selects `auto` (default), `torch`, or `onnx`.
`auto` uses the calibrated PyTorch FP16 path when CUDA is available and ONNX
Runtime everywhere else.

`VEHICLE_DAMAGE_DEVICE` selects `auto` (default), `cuda`, `gpu`, `npu`, or
`cpu`. The ONNX path supports CPU out of the box. Qualcomm Adreno GPU / Hexagon
NPU use the optional QNN execution provider:

```powershell
pip install onnxruntime-qnn>=2.6
$env:VEHICLE_DAMAGE_DEVICE = "npu"
```

The ONNX export is bound to the checkpoint SHA-256, and each ONNX file is
verified against its metadata on load. PyTorch inference with the selected
profile requires CUDA because the frozen profile uses FP16 mixed precision.

## Model and safety notes

- Input: RGB photograph of a vehicle exterior.
- Output classes: dent, scratch, crack_or_breakage, paint_damage,
  deformation_or_detachment.
- Inference uses 768 px tiles, overlap, horizontal-flip TTA, and the frozen
  1.0x/1.5x triage scales.
- Low resolution, severe exposure/glare, or blur is routed to
  `recapture_required`. Glare, near-threshold uncertainty, or disagreement
  between the triage and precise branches is routed to
  `manual_review_required`.
- The model does not estimate repair cost and cannot detect hidden structural
  or mechanical damage.
- The checkpoint and thresholds are development-only. Fresh first-party
  calibration and an untouched clean-inclusive test are still required before
  production use.

## Tests

```powershell
python -m pytest
```

The suite verifies the pinned checkpoint/profile hashes, API contracts,
inference math, quality gating, and the ONNX tile backend. It loads the selected
checkpoint for an architecture check but does not run full-image inference; a
real-image smoke test can be done through the API or CLI.
