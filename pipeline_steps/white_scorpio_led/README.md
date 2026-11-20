# Pipeline steps for white_scorpio_led.jpg

Each folder is one stage of the damage pipeline, in the order the app runs it.

- `step_01_original` - Step 1 - Uploaded photo
- `step_02_quality_gate` - Step 2 - Quality gate
- `step_03_working_resolution` - Step 3 - Resize into the model's working range
- `step_04_scale_pyramid` - Step 4 - Scale pyramid
- `step_05_tiling` - Step 5 - Tiling and flip augmentation
- `step_06_probability_maps` - Step 6 - Dense probability maps from the ONNX damage model
- `step_07_threshold_and_regions` - Step 7 - Threshold 0.381 and region filtering
- `step_08_glare_filter` - Step 8 - Glare / light filter
- `step_09_part_segmentation` - Step 9 - Car part segmentation (second ONNX model)
- `step_10_part_correction_and_side` - Step 10 - Front/rear correction and photo angle
- `step_11_panel_assignment_and_rating` - Step 11 - Damage on panels and intensity rating
- `step_12_final_report` - Step 12 - Final decision (the real predictor, end to end)
