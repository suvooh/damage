import numpy as np

from vehicle_damage.views import infer_view, side_of

NAMES = ["background", "front-bumper", "fender", "front-door", "back-door", "quarter-panel",
         "back-bumper", "headlight", "hood", "grille", "tail-light", "trunk"]


def _strip(order):
    """Paint parts left-to-right across the image in the given order."""
    mask = np.zeros((100, 120), dtype=np.uint8)
    width = 120 // len(order)
    for i, name in enumerate(order):
        mask[30:70, i * width:(i + 1) * width] = NAMES.index(name)
    return mask


def test_front_on_image_left_means_cars_left_side():
    mask = _strip(["front-bumper", "fender", "front-door", "back-door", "quarter-panel", "back-bumper"])
    view = infer_view(mask, NAMES)
    assert view["view"] == "left" and view["confidence"] > 0.9
    assert side_of("fender", view, 0.2) == "left"
    assert side_of("bonnet", view, 0.2) is None  # central panels have no side


def test_front_on_image_right_means_cars_right_side():
    mask = _strip(["back-bumper", "quarter-panel", "back-door", "front-door", "fender", "front-bumper"])
    assert infer_view(mask, NAMES)["view"] == "right"


def test_straight_front_view_mirrors_left_and_right():
    mask = np.zeros((100, 120), dtype=np.uint8)
    mask[60:90, 10:110] = NAMES.index("front-bumper")
    mask[20:60, 20:100] = NAMES.index("hood")
    mask[50:60, 10:30] = NAMES.index("headlight")
    mask[50:60, 90:110] = NAMES.index("headlight")
    view = infer_view(mask, NAMES)
    assert view["view"] == "front"
    assert side_of("headlight", view, 0.1) == "right"  # image left = car's right when facing it
    assert side_of("headlight", view, 0.9) == "left"


def test_straight_rear_view_keeps_left_and_right():
    mask = np.zeros((100, 120), dtype=np.uint8)
    mask[60:90, 10:110] = NAMES.index("back-bumper")
    mask[20:60, 20:100] = NAMES.index("trunk")
    mask[50:60, 10:30] = NAMES.index("tail-light")
    view = infer_view(mask, NAMES)
    assert view["view"] == "rear"
    assert side_of("tail_light", view, 0.1) == "left"


def test_no_parts_or_ambiguous_layout_is_unknown():
    assert infer_view(None, None)["view"] == "unknown"
    assert infer_view(np.zeros((10, 10), np.uint8), NAMES)["view"] == "unknown"
    assert side_of("fender", {"view": "unknown", "car_center_x": 0.5}, 0.2) is None


def test_harmonise_ends_relabels_bumper_to_the_visible_end():
    from vehicle_damage.views import harmonise_ends

    names = ["background", "front-bumper", "back-bumper", "grille", "headlight", "hood", "tail-light", "trunk"]
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[70:90, 10:90] = names.index("back-bumper")  # mislabelled bumper
    mask[50:60, 30:70] = names.index("grille")
    mask[40:50, 10:30] = names.index("headlight")
    mask[10:40, 20:80] = names.index("hood")
    fixed = harmonise_ends(mask, names)
    assert (fixed == names.index("front-bumper")).sum() == 1600
    assert (fixed == names.index("back-bumper")).sum() == 0
    rear = np.zeros((100, 100), dtype=np.uint8)
    rear[70:90, 10:90] = names.index("front-bumper")
    rear[20:60, 20:80] = names.index("trunk")
    assert (harmonise_ends(rear, names) == names.index("back-bumper")).sum() == 1600
    mixed = mask.copy()
    mixed[0:10, 0:100] = names.index("tail-light")  # both ends visible -> leave alone
    mixed[10:40, 20:80] = names.index("trunk")
    assert (harmonise_ends(mixed, names) == names.index("back-bumper")).sum() == 1600
