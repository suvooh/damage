Step 6 - Dense probability maps from the ONNX damage model
==========================================================

Three heads per pixel: presence (damage yes/no), type (5 classes), exterior (is this vehicle surface).
any-damage score = (1 - p_background) x (0.5 + 0.5 x p_exterior). Blue = 0, red = 1.
Files: score per scale, the fused score (average of scales), exterior map, and the dominant damage type.
