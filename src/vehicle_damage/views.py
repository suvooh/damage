"""Which side of the car a photo shows, and the side of each damaged panel.

Left and right are the car's own sides as seen from the driver's seat.

Rule (geometry of the part mask, no extra model): every panel has a position
along the car from front (0) to rear (6). In a side or three-quarter photo the
panels line up along the image x axis. If the front panels sit to the image
left of the rear panels, the camera is looking at the car's LEFT side; if they
sit to the right, the RIGHT side. Straight-on front or rear photos have no
such spread; there the side of a panel follows from which half of the car it
is in (mirrored for the front view). Anything else is "unknown" and callers
must not guess.
"""

from __future__ import annotations

import numpy as np

from .severity import PART_DISPLAY

LONGITUDINAL = {
    "front_bumper": 0.0, "grille": 0.0, "license_plate": 0.2, "headlight": 0.4, "bonnet": 1.0,
    "fender": 1.4, "front_wheel": 1.5, "windshield": 2.0, "side_mirror": 2.3, "front_door": 3.0,
    "front_window": 3.0, "roof": 3.5, "side_skirt": 3.5, "rear_door": 4.0, "rear_window": 4.0,
    "rear_windshield": 5.0, "quarter_panel": 5.0, "rear_wheel": 5.2, "boot": 5.5, "tail_light": 5.8,
    "rear_bumper": 6.0,
}
SIDE_PARTS = frozenset({
    "headlight", "tail_light", "fender", "front_wheel", "rear_wheel", "side_mirror", "front_door",
    "rear_door", "front_window", "rear_window", "side_skirt", "quarter_panel",
})
MIN_SHARE = 0.004  # a panel must cover 0.4% of the photo to count
SIDE_CORRELATION = 0.5
MIN_SPREAD = 1.0
END_VIEW_SHARE = 0.7


def infer_view(part_mask: np.ndarray | None, part_names: list[str] | None) -> dict[str, object]:
    """Return ``{"view": left|right|front|rear|unknown, "confidence", "car_center_x"}``."""
    unknown = {"view": "unknown", "confidence": 0.0, "car_center_x": None}
    if part_mask is None or not part_names:
        return unknown
    total = part_mask.size
    width = part_mask.shape[1]
    order, xpos, weight = [], [], []
    for i, name in enumerate(part_names):
        if i == 0:
            continue
        display = PART_DISPLAY.get(name, name)
        if display not in LONGITUDINAL:
            continue
        cols = np.nonzero(part_mask == i)[1]
        if cols.size / total < MIN_SHARE:
            continue
        order.append(LONGITUDINAL[display])
        xpos.append(cols.mean() / width)
        weight.append(cols.size / total)
    if not order:
        return unknown
    o, x, w = np.array(order), np.array(xpos), np.array(weight)
    center = float(np.average(x, weights=w))
    if len(o) >= 2 and o.max() - o.min() >= MIN_SPREAD and o.std() > 0 and x.std() > 0:
        cov = np.cov(o, x, aweights=w)
        corr = float(cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1]))
        if abs(corr) >= SIDE_CORRELATION:
            return {"view": "left" if corr > 0 else "right", "confidence": abs(corr), "car_center_x": center}
    front_share = float(w[o <= 1.0].sum() / w.sum())
    rear_share = float(w[o >= 5.0].sum() / w.sum())
    if front_share >= END_VIEW_SHARE:
        return {"view": "front", "confidence": front_share, "car_center_x": center}
    if rear_share >= END_VIEW_SHARE:
        return {"view": "rear", "confidence": rear_share, "car_center_x": center}
    return {**unknown, "car_center_x": center}


def side_of(part: str | None, view: dict[str, object], centroid_x: float) -> str | None:
    """Side ("left"/"right") of a damaged panel, or None when unknown or not sided.

    ``centroid_x`` is the damage centroid as a fraction of image width.
    """
    if part not in SIDE_PARTS:
        return None
    kind = view.get("view")
    if kind in ("left", "right"):
        return str(kind)
    center = view.get("car_center_x")
    if center is None:
        return None
    on_image_left = centroid_x < float(center)
    if kind == "front":  # facing the car: its left is the image's right
        return "right" if on_image_left else "left"
    if kind == "rear":  # behind the car: its left is the image's left
        return "left" if on_image_left else "right"
    return None


# Parts that exist only at one end of the car; the bumpers are the labels that get confused.
FRONT_ONLY = ("grille", "headlight", "hood", "windshield")
REAR_ONLY = ("tail-light", "trunk", "back-windshield")
END_DOMINANCE = 3.0
MIN_END_SHARE = 0.01


def harmonise_ends(part_mask: np.ndarray | None, part_names: list[str] | None) -> np.ndarray | None:
    """Relabel a bumper to the end of the car that the photo clearly shows.

    The part model sometimes calls a front bumper "rear bumper" (and vice versa). When
    one end's unambiguous parts (grille, headlights, bonnet, windshield vs tail lights,
    boot, rear windshield) outweigh the other end's by at least 3x, the opposite-end
    bumper label is changed to this end's bumper.
    """
    if part_mask is None or not part_names:
        return part_mask
    ids = {name: i for i, name in enumerate(part_names)}
    if "front-bumper" not in ids or "back-bumper" not in ids:
        return part_mask
    counts = np.bincount(part_mask.ravel(), minlength=len(part_names)) / part_mask.size
    front = sum(counts[ids[n]] for n in FRONT_ONLY if n in ids)
    rear = sum(counts[ids[n]] for n in REAR_ONLY if n in ids)
    fixed = part_mask.copy()
    if front >= MIN_END_SHARE and front >= END_DOMINANCE * rear:
        fixed[part_mask == ids["back-bumper"]] = ids["front-bumper"]
    elif rear >= MIN_END_SHARE and rear >= END_DOMINANCE * front:
        fixed[part_mask == ids["front-bumper"]] = ids["back-bumper"]
    return fixed
