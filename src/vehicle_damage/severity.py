"""Damage intensity (1-5) per panel and car-part assignment for detections.

Rating follows the panel-based rubric supplied by the product owner (the
"Damage Intensity Rating" used by the reference portal):

* Dents: a depth class (d1/d2/d3) is combined with the damage area as a
  percentage of the panel area.

      depth   0-20%  20-40%  40-60%  60-80%  80-100%
      d1      1      2       3       4       4
      d2      2      3       4       5       5
      d3      3      4       5       5       5

* Scratches: damage area under 20% of the panel -> 1, otherwise -> 3.

Extensions that are NOT in the supplied rubric (flagged in every result via
``basis``):

* ``crack_or_breakage`` and ``deformation_or_detachment`` have no rubric entry;
  they are rated on the rubric's deepest-dent row (d3) by panel area
  (0-20% -> 3, 20-40% -> 4, 40%+ -> 5) instead of a flat maximum.
* ``paint_damage`` is rated like a scratch.
* There is no dent-depth model yet (the training data carries no depth labels),
  so dents are rated as d2 and the result says so (``depth_assumed``).
* When no part model/overlap is available the area percentage is taken against
  the vehicle instead of the panel (``area_basis == "vehicle"``).
"""

from __future__ import annotations

import numpy as np

SEVERITY_LABELS = {
    1: "trivial",
    2: "minor",
    3: "moderate",
    4: "major",
    5: "replace",
}

DENT_TABLE = {
    "d1": (1, 2, 3, 4, 4),
    "d2": (2, 3, 4, 5, 5),
    "d3": (3, 4, 5, 5, 5),
}
ASSUMED_DEPTH = "d2"
REPLACE_TYPES = frozenset({"crack_or_breakage", "deformation_or_detachment"})
SCRATCH_LIKE = frozenset({"scratch", "paint_damage"})

# Fine dataset labels -> user-facing part names.
PART_DISPLAY = {
    "windshield": "windshield",
    "back-windshield": "rear_windshield",
    "front-window": "front_window",
    "back-window": "rear_window",
    "front-door": "front_door",
    "back-door": "rear_door",
    "front-wheel": "front_wheel",
    "back-wheel": "rear_wheel",
    "front-bumper": "front_bumper",
    "back-bumper": "rear_bumper",
    "headlight": "headlight",
    "tail-light": "tail_light",
    "hood": "bonnet",
    "trunk": "boot",
    "license-plate": "license_plate",
    "mirror": "side_mirror",
    "roof": "roof",
    "grille": "grille",
    "rocker-panel": "side_skirt",
    "quarter-panel": "quarter_panel",
    "fender": "fender",
}


def area_bucket(area_fraction: float) -> int:
    """Index of the 20%-wide column for a 0-1 area fraction (100% -> last)."""
    return min(4, int(area_fraction * 5))


def severity_for(
    class_name: str,
    area_fraction: float,
    *,
    depth_class: str | None = None,
    area_basis: str = "panel",
) -> dict[str, object]:
    """Rate one damage; ``area_fraction`` is damage area / panel area in [0, 1]."""
    if not np.isfinite(area_fraction) or area_fraction < 0:
        raise ValueError("area_fraction must be finite and non-negative")
    area_fraction = min(1.0, float(area_fraction))
    depth_assumed = False
    if class_name == "dent":
        if depth_class is None:
            depth_class, depth_assumed = ASSUMED_DEPTH, True
        if depth_class not in DENT_TABLE:
            raise ValueError(f"unknown dent depth class {depth_class!r}")
        score = DENT_TABLE[depth_class][area_bucket(area_fraction)]
        basis = "rubric"
        reason = f"dent {depth_class}{' (assumed)' if depth_assumed else ''}, {area_fraction:.0%} of {area_basis}"
    elif class_name in SCRATCH_LIKE:
        score = 1 if area_fraction < 0.20 else 3
        basis = "rubric" if class_name == "scratch" else "extension:paint_as_scratch"
        reason = f"{class_name.replace('_', ' ')}, {area_fraction:.0%} of {area_basis}"
    elif class_name in REPLACE_TYPES:
        score = DENT_TABLE["d3"][area_bucket(area_fraction)]
        basis = "extension:rated_on_d3_row"
        reason = (
            f"{class_name.replace('_', ' ')}, {area_fraction:.0%} of {area_basis} "
            "(not in the rubric; rated on the d3 row)"
        )
    else:
        raise ValueError(f"unknown damage class {class_name!r}")
    return {
        "score": score,
        "label": SEVERITY_LABELS[score],
        "reason": reason,
        "basis": basis,
        "depth_class": depth_class if class_name == "dent" else None,
        "depth_assumed": depth_assumed,
        "area_basis": area_basis,
        "area_fraction": area_fraction,
    }


def assign_part(
    region: np.ndarray,
    part_mask: np.ndarray,
    part_names: list[str],
    *,
    minimum_overlap: float = 0.2,
) -> dict[str, object] | None:
    """Majority part under a damage region (``part_mask`` ids index ``part_names``).

    Returns ``None`` when the region does not overlap any part enough to name.
    ``part_names[0]`` is background and never returned. The result includes the
    part id, the damage pixels inside it, and the panel's total pixels.
    """
    if region.shape != part_mask.shape:
        raise ValueError("region and part_mask must have the same shape")
    total = int(region.sum())
    if total == 0:
        return None
    counts = np.bincount(part_mask[region].astype(np.int64), minlength=len(part_names))
    counts[0] = 0
    best = int(counts.argmax())
    if best == 0 or counts[best] / total < minimum_overlap:
        return None
    name = part_names[best]
    return {
        "id": best,
        "name": PART_DISPLAY.get(name, name),
        "overlap": float(counts[best] / total),
        "damage_pixels": int(counts[best]),
        "panel_pixels": int((part_mask == best).sum()),
    }
