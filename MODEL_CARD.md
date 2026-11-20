# Vehicle damage segmentation model card

## Update: inference resolution normalisation

- `working_image()` resizes each photo to a 1,024-1,600 px long side before inference;
  `restore_size()` maps the probabilities back. Held-out damage caught 87.7% -> 96.3% with
  no change in clean false alerts (0/122 test, 15/456 fresh).

## Update: ft_v3 (served model)

- v2 plus 537 hard negatives mined with v2 from 1,674 clean Wikimedia car photos
  (`tools/fetch_wild_pool.py`, `tools/mine_hard_negatives.py`); last 3 blocks + heads,
  3 x 900 samples, mined photos sampled 3x, whole frame weighted as hard negative.
- Real pipeline at 0.381: fresh clean photos flagged 14.0% -> 3.3% (456 photos);
  held-out damage caught 88.9% -> 87.7% (81 photos). See PRODUCTION_READINESS.md.

## Update: ft_v2 (previous model)

- Fine-tuned the last 3 DINOv2 blocks and the heads on cached 784 px features
  with 251 Wikimedia Commons clean car photos (CC BY / CC BY-SA / CC0 / PD,
  licence and attribution in the fetch manifest) as full-frame hard negatives.
- Held-out Wikimedia clean photos (proxy protocol, test split, 72 images): alert
  rate at the old threshold 54.2% -> 0.0%; damage recall at 1% clean false
  alerts 0.457 -> 0.963. Mask IoU (about 0.38) and type accuracy (about 0.77)
  did not improve.
- Tiled pipeline check: 10 held-out clean photos, previous model flagged 8, ft_v2
  flagged 0; the user-reported red TR4 is no longer flagged; all 5 damaged
  samples are still detected.
- Threshold 0.381 was chosen on a proxy protocol, the checkpoint on validation,
  and the clean images are curated Wikimedia photos, not customer photos.
  Dent depth (d1-d3) is not modelled. The NPU context graph was not rebuilt.
- Real tiled pipeline on 203 held-out test images (81 damaged, 50 COCO clean, 72 wild
  clean): damaged cases caught 88.9% (95% CI 80-94%), clean photos falsely flagged
  2.5% (CI 0.8-7.0%). The previous checkpoint on the same images: 96.3% and 29.5%.
  See `PRODUCTION_READINESS.md` for the full gate table and blockers.
- Damage-type reliability (held-out splits, pixels inside predicted damage,
  `tools/type_reliability.py`): a predicted `crack_or_breakage` matches the
  label 87% of the time and `deformation_or_detachment` 73%, but a predicted
  `scratch` matches only 32% (36% are labelled crack, 23% dent) and `paint_damage`
  36%. Confidence does not separate right from wrong (crack precision 87% at any
  confidence, 91% at 90%), so no confidence gate is applied. The labels come from
  "broken part" polygons that often cover whole scuffed bumpers, which is why
  scratches and scuffs are frequently called cracks. More scratch/paint-labelled
  data is the fix.
- The figures in the sections below describe the previous checkpoint.

## Status

**Development-only; not production approved.**

The pinned checkpoint
`runs/pilot_cc0_dinov2_parts_hardneg_long/best.pt` is the hard-negative
refinement candidate from the source project. Its frozen profile applies a
1,200-pixel connected-component guard. On the source project's held-out mixed
validation it detected 91/98 damage cases (92.86%) and raised alerts on 1/100
clean COCO images (1.0%). All three supplied clean regression images produced no
damage mask or detections. Recall is below the source project's release bar, so
this is not a production accuracy claim.

The checkpoint and calibration profile are SHA-256 pinned in
`runs/SELECTED_DEVELOPMENT_MODEL.json`; the ONNX exports are pinned separately
in their metadata files. All are verified on load. Every prediction response
carries `model_status` and `production_approved: false`.

## Intended use

Visible exterior vehicle damage triage from RGB photographs:

- dents
- scratches
- crack or breakage
- paint damage
- deformation or detachment

Outputs are a semantic mask, connected-component detections, and a fail-closed
disposition. The model is **not** a repair-cost estimator and cannot detect
hidden structural or mechanical damage.

## Architecture

- Backbone: DINOv2 ViT-S/14 (`vit_small_patch14_dinov2.lvd142m`), image size
  configured at 392 during training.
- Decoder: one shared backbone with three lightweight dense heads:
  damage presence, damage type, and vehicle exterior.
- Only this architecture is supported by this project's `load_checkpoint`.
  DeepLabV3/ResNet-50, four-layer, role-split, and other variants from the
  source research repository are rejected.
- Inference: 768 px tiles with 192 px overlap, horizontal-flip test-time
  augmentation, multi-scale triage (1.0x and 1.5x), and a precise mask at the
  frozen segmentation threshold.
- Runtime paths: CUDA PyTorch FP16 (calibrated) or ONNX Runtime FP32 on CPU /
  Qualcomm GPU, and ONNX Runtime FP16 on Qualcomm NPU.

## Decision routing

The API never turns an uncertain image into a positive or negative answer just
because pixels are present.

| Decision | Meaning |
|---|---|
| `damage_detected` | Damage evidence passed both calibrated branches and no quality/review reason applies. |
| `no_damage_detected` | No evidence passed the frozen thresholds. |
| `manual_review_required` | Localized glare, near-threshold ambiguity, triage-only evidence, segmentation-only evidence, or damage-type disagreement. |
| `recapture_required` | Low resolution, severe underexposure/overexposure, or blur/low detail. |

`automated_decision_allowed` is false for review and recapture outcomes;
`manual_review` remains true for both so older consumers fail closed.

## Calibration profile

`runs/pilot_cc0_dinov2_parts_hardneg_long/calibration-fp16-comp1200.json`:

- triage threshold: `0.7356525659561157`
- segmentation threshold: same (no separate mask threshold in this profile)
- target recall: 0.97, measured recall: 1.0 on the calibration cases
- negative-pixel FPR: 0.00765
- exterior floor: 0.5
- tile size: 768, overlap: 192
- horizontal flip TTA: enabled
- mixed precision: enabled
- minimum component pixels: 1200
- triage scales: 1.0, 1.5
- segmentation scales: null (uses triage scales)

## Known limitations

- Thresholds, component filter, and checkpoint selection consumed development
  data. They are not independently validated.
- The source validation split has no independently reviewed clean groups and no
  true clean-inclusive test set.
- Synthetic stress results from the source project are diagnostics, not real
  nuisance-slice evidence.
- Severe low resolution can add false regions.
- Glare, grime, seams, dark/metallic paint, subtle damage, glass, occlusion,
  unseen cameras, and prior repairs require first-party held-out testing.
- The precise mask had weak connected-region precision/recall in the source
  evaluation. High case recall does not mean accurate localization.
- CPU inference can be slow (tens of seconds per photo depending on tile count).

## Provenance

This project is a stripped extraction of the damage portion of the dealer
platform in the source workspace. Only the selected checkpoint, its calibration
profile, ONNX exports, inference code, and bundled damage samples were retained.
The source model card and pilot results in the original workspace remain the
authoritative record of the training and validation experiments.
