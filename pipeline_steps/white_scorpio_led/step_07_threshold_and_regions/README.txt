Step 7 - Threshold 0.381 and region filtering
=============================================

Pixels with score >= 0.381 are damage candidates. Connected regions (8-neighbour) smaller than 1200 px are dropped. Each region takes the damage type most of its pixels voted for.
Legend: dent = RGB(234, 88, 12), scratch = RGB(2, 132, 199), crack_or_breakage = RGB(220, 38, 38), paint_damage = RGB(202, 138, 4), deformation_or_detachment = RGB(147, 51, 234)
