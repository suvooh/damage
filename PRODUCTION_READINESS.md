# Production readiness (status as of 2026-10-27)

**Verdict: not production ready.** The model is a clear, measured improvement over
the previous checkpoint and is usable for supervised triage (a human reviews every
result), but several gates below fail or have never been measured. Nothing here is
a claim of customer-grade accuracy.

## Current served configuration (2026-10-28)

v3 + resolution normalisation + conservative glare filter + bumper front/rear correction.
Real tiled pipeline, threshold 0.381:

| | Value |
|---|---|
| Held-out damage caught (81) | 78 (96.3%, CI 90-99%) |
| Held-out test clean flagged (122) | 0 (CI 0-3.1%) |
| Fresh clean photos flagged (456) | 11 (2.4%, CI 1.3-4.3%) |
| User scratch photos (2) | both detected |
| User clean Scorpio with LED strip | still flagged (known false positive) |

- Glare filter: damage regions at least 85% inside a 2.5% halo around clipped pixels
  (>= 0.985) go to manual review. Chosen from an 18-setting grid on these held-out photos
  (so optimistic): removed 4 of 15 fresh false alerts, lost 0 of 78 detections. Settings
  that also removed the LED-strip false alert lost 6-8 real damages.
- Bumper correction: when one end of the car (grille/headlights/bonnet/windshield vs
  tail lights/boot/rear windshield) outweighs the other 3:1, the bumper is relabelled.
- v5 (retrained from the original model with targeted hard negatives and scratch/dent
  oversampling) looked better on the single-scale proxy (scratch recall 81% vs 71%) but
  lost in the real pipeline: 74/81 damage and 33/456 fresh false alerts. Rejected. The
  proxy is not reliable for model choice; only real-pipeline numbers decide.

## Update: resolution normalisation (served)

Photos are now resized so their long side is 1,024-1,600 px before inference (the range
the model was trained on); results are mapped back to the original photo. Small photos
had made fine scratches invisible: a 612 px photo of a scratched red door scored 0.00
natively and 0.94 at 1,024 px. Real pipeline, v3, threshold 0.381:

| | Before | With normalisation |
|---|---|---|
| Held-out damage caught (81) | 71 (87.7%) | 78 (96.3%, CI 90-99%) |
| Held-out clean flagged (122) | 0 | 0 |
| Fresh clean photos flagged (456) | 15 (3.3%) | 15 (3.3%) |
| User-reported scratch photos (2) | missed | both detected |

Lesson: the damage test set is dominated by large damage, so minor-scratch recall must be
tracked separately (`tools/per_class_recall.py`); that regression was missed at first.

## Update: v3 (served model) - false positives

v3 = v2 + 537 hard negatives mined from 1,674 clean Wikimedia car photos (Indian-market
models, silver/white/black/red paint, parked and showroom scenes). Real tiled pipeline,
same threshold 0.381:

| | v2 | v3 |
|---|---|---|
| Fresh clean photos flagged (456, never trained on) | 64 (14.0%, CI 11.1-17.5) | 15 (3.3%, CI 2.0-5.4) |
| Held-out test clean flagged (122) | 3 (2.5%) | 0 (0.0%, CI 0-3.1) |
| Held-out test damage caught (81) | 72 (88.9%) | 71 (87.7%, CI 79-93) |
| Damage caught at 2% false alerts on all 578 clean | 42.0% | 86.4% |

Remaining false alerts are mostly silver/white cars under showroom lighting; a few
"clean" photos show real wear (weathered old cars). The old calibration split could not
set v3's threshold (it chose 0.05, which flags 13% of fresh photos) because it lacks this
hard clean domain, so 0.381 was kept. A first-party calibration set is still needed.

## Measured in the real deployed pipeline (v2, earlier)

Tiled 768 px, scales 1.0x/1.5x, flip TTA, ONNX Runtime on the Adreno GPU. 203 held-out
test images that neither threshold nor checkpoint selection touched: 81 damaged (CC0
Humans in the Loop), 50 COCO clean, 72 Wikimedia "wild" clean (classic cars, chrome,
car shows). Raw scores: `reports/real_pipeline_2026-10-27/`. Reproduce with
`tools/real_pipeline_scores.py` and `tools/real_pipeline_report.py`.

| | Previous model (thr 0.736) | Current model (thr 0.381) |
|---|---|---|
| Damaged cases caught | 96.3% (CI 89.7-98.7) | 88.9% (CI 80.2-94.0) |
| Clean photos falsely flagged | 29.5% (CI 22.1-38.1) | 2.5% (CI 0.8-7.0) |
| Wild clean falsely flagged | 47.2% (CI 36.1-58.6) | 4.2% (CI 1.4-11.5) |
| Recall at 5% clean false alerts | 56.8% | 93.8% |

The threshold was chosen on a separate calibration split. That sweep suggested 0.49, but
on the test set 0.49 gave the same false alerts and lower recall (80.2%), so 0.381 was kept.

## Gates

| Gate | Target | Measured | Status |
|---|---|---|---|
| False alerts on clean photos | < 5% | 2.4% on 456 fresh photos (upper CI 4.3%); 0% on 122 test photos | Passes on public photos; not measured on customer photos |
| Damaged-case recall | >= 97% | v3 + normalisation: 96.3% (CI 90-99%) | Borderline (point estimate just under) |
| Damage type: scratch / paint | usable | a predicted scratch is a scratch 32% of the time, paint 36% | FAIL |
| Damage type: crack / deformation | usable | 87% / 73% precision | Weak |
| Mask quality | mask mAP@50 > 0.65 | mean IoU about 0.38 (proxy protocol; mAP not measured) | FAIL |
| Part naming | usable | mIoU 0.62; bonnet 0.88, windshield 0.91, bumpers ~0.78; side mirror 0.28, front window 0.23 | Partial |
| Left / right side of the car | needed for the heatmap | not modelled | Not built |
| Dent depth (d1-d3) | needed by the rating rubric | not modelled, assumed d2 | Not built |
| Severity vs human judgement | agree with assessors | never compared with human labels | Not measured |
| Customer / phone photos | held-out first-party set | none; public data only | Not measured |
| Latency | < 10 s per photo | model pass 8.1 s average on a laptop Adreno GPU (13-24 s per API request incl. report rendering); 60-280 s on CPU | Passes on GPU only |
| On-device phone runtime | CoreML / TFLite parity > 0.98 | ONNX/QNN on a Snapdragon laptop only; no mobile export | Not demonstrated |
| Operations | auth, monitoring, retention | none; single worker; no authentication | Not built |

## Blockers, in order of impact

1. **Labelled scratch/paint/dent data.** Type accuracy and mask quality are limited by
   about 570 labelled damage images whose "broken part" polygons cover whole bumpers. More,
   better-labelled data is the main lever. CarDD (4,000 images) needs a licensing form
   sent to its authors and is probably research-only; check before relying on it.
2. **First-party validation set.** 100+ clean and 100+ damaged photos from real phone
   captures, labelled by an assessor, split by vehicle. Without it the numbers above
   cannot be trusted for your customers.
3. **Severity validation.** Have assessors rate 100+ panels 1-5 and compare with the rubric
   output; add a dent-depth classifier or collect depth labels.
4. **Licensing review.** Wikimedia images are CC BY / CC BY-SA (attribution list in
   `data_provenance/`; share-alike implications for model weights need legal review). The
   COCO images carry mixed per-image Flickr licences, some non-commercial. HITL is CC0.
5. **Mobile packaging.** Export and parity-test for iOS/Android; the current model is a
   ViT-S run at several scales and is heavy for phones.
6. **Service hardening.** Authentication, rate limits, request logging, monitoring,
   model/version pinning in responses, and an image-retention policy.

## What is in place

Hash-pinned checkpoint, profile, ONNX export and part model verified on load; fail-closed
decisions (`manual_review_required`, `recapture_required`); startup warm-up; GPU inference
via `onnxruntime-qnn`; 66 unit/integration tests; rollback file
`runs/SELECTED_DEVELOPMENT_MODEL.previous.json`.
