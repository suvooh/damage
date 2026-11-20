"""Save the output of every pipeline step for ONE photo into numbered folders.

It calls the same functions, in the same order, as DamagePredictor.predict_bytes,
so each folder shows what that stage really produces. The last step runs the real
predictor end to end and checks it against the manual run.

Usage: python tools/export_pipeline_steps.py PHOTO OUT_DIR
Folders: step_01_original ... step_12_final_report, each with images and a README.txt.
"""

from __future__ import annotations

import base64
import io
import json
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vehicle_damage.api import (  # noqa: E402
    CLASS_COLORS, SEVERITY_RGB, DamagePredictor, _components, _panel_summary, _render_layers, _score_map,
    _visible_parts,
)
from vehicle_damage.inference import (  # noqa: E402
    _starts, assess_quality, fuse_probability_pyramid, glare_zone, predict_probability_pyramid,
    probabilities_to_mask, restore_size, split_glare_components, working_image,
)
from vehicle_damage.views import harmonise_ends, infer_view, side_of  # noqa: E402

PART_PALETTE = [(0, 0, 0)] + [tuple(int(v * 255) for v in c) for c in (
    (0.12, 0.47, 0.71), (0.68, 0.78, 0.91), (1.0, 0.5, 0.05), (1.0, 0.73, 0.47), (0.17, 0.63, 0.17), (0.6, 0.87, 0.54),
    (0.84, 0.15, 0.16), (1.0, 0.6, 0.6), (0.58, 0.4, 0.74), (0.77, 0.69, 0.84), (0.55, 0.34, 0.29), (0.77, 0.61, 0.58),
    (0.89, 0.47, 0.76), (0.97, 0.71, 0.82), (0.5, 0.5, 0.5), (0.78, 0.78, 0.78), (0.74, 0.74, 0.13), (0.86, 0.86, 0.55),
    (0.09, 0.75, 0.81), (0.62, 0.85, 0.9), (0.4, 0.2, 0.0), (0.2, 0.2, 0.6))]


def heat(score: np.ndarray) -> Image.Image:
    """0..1 -> blue-green-yellow-red image."""
    v = np.clip(score, 0, 1)
    r = np.clip(4 * v - 2, 0, 1)
    g = np.clip(np.minimum(4 * v, 4 - 4 * v), 0, 1)
    b = np.clip(2 - 4 * v, 0, 1)
    return Image.fromarray((np.stack([r, g, b], -1) * 255).astype(np.uint8))


def blend(image: Image.Image, overlay: Image.Image, alpha: float = 0.55) -> Image.Image:
    return Image.blend(image.convert("RGB"), overlay.convert("RGB").resize(image.size), alpha)


def colour_ids(ids: np.ndarray, palette) -> Image.Image:
    """Colour a part-id mask; id 0 (background) stays black."""
    out = np.zeros((*ids.shape, 3), dtype=np.uint8)
    for i in np.unique(ids):
        if i:
            out[ids == i] = palette[int(i) % len(palette)]
    return Image.fromarray(out)


def decode(b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64)))


class Steps:
    def __init__(self, root: Path) -> None:
        self.root = root
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True)
        self.index: list[tuple[str, str]] = []

    def folder(self, name: str, title: str, text: str) -> Path:
        path = self.root / name
        path.mkdir()
        (path / "README.txt").write_text(f"{title}\n{'=' * len(title)}\n\n{text.strip()}\n", encoding="utf-8")
        self.index.append((name, title))
        return path


def main() -> None:
    photo, out = Path(sys.argv[1]), Path(sys.argv[2])
    torch.set_num_threads(10)
    predictor = DamagePredictor()
    cal, model = predictor.calibration, predictor.model
    steps = Steps(out)
    thr = cal.any_damage_threshold
    classes = predictor.classes

    # 01 ---------------------------------------------------------------- original
    raw = Image.open(photo)
    image = ImageOps.exif_transpose(raw).convert("RGB")
    d = steps.folder("step_01_original", "Step 1 - Uploaded photo",
                     f"The file as received: {raw.size[0]} x {raw.size[1]} px, mode {raw.mode}.\n"
                     f"After EXIF rotation and RGB conversion: {image.width} x {image.height} px.\n"
                     "Limits enforced by the API: JPEG/PNG/WebP, 20 MB, 50 megapixels.")
    image.save(d / "original.png")

    # 02 ---------------------------------------------------------------- quality gate
    quality = assess_quality(image)
    rgb = np.asarray(image, dtype=np.float32) / 255.0
    lum = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    sat = (rgb.max(2) - rgb.min(2)) / np.maximum(rgb.max(2), 1e-6)
    d = steps.folder("step_02_quality_gate", "Step 2 - Quality gate",
                     "Cheap checks on the raw photo before any model runs. Images below are masks of what each check counts.\n"
                     f"Short side {min(image.size)} px (recapture below 480). Dark pixels {quality.dark_fraction:.1%} (limit 35%).\n"
                     f"Clipped highlights {quality.clipped_highlight_fraction:.2%} (limit 20%). Specular {quality.specular_highlight_fraction:.2%} (review above 1.5%).\n"
                     f"Sharpness {quality.sharpness:.5f} (recapture below 0.00035).\n"
                     f"Result: {', '.join(quality.review_reasons) or 'no quality issue'}.")
    Image.fromarray(((lum < 0.04) * 255).astype(np.uint8)).save(d / "dark_pixels.png")
    Image.fromarray(((lum > 0.985) * 255).astype(np.uint8)).save(d / "clipped_highlights.png")
    Image.fromarray((((lum > 0.90) & (sat < 0.12)) * 255).astype(np.uint8)).save(d / "specular_highlights.png")
    (d / "quality.json").write_text(json.dumps(asdict(quality), indent=1), encoding="utf-8")

    # 03 ---------------------------------------------------------------- working resolution
    work = working_image(image)
    d = steps.folder("step_03_working_resolution", "Step 3 - Resize into the model's working range",
                     f"The model was trained on photos with a 1,024-1,600 px long side. This photo ({max(image.size)} px long side) "
                     f"is resized to {work.width} x {work.height} px (bicubic) so thin scratches keep enough pixels.\n"
                     "All probability maps are mapped back to the original size at the end.")
    work.save(d / "working_image.png")

    # 04 ---------------------------------------------------------------- scale pyramid
    scales = tuple(cal.triage_scales)
    d = steps.folder("step_04_scale_pyramid", "Step 4 - Scale pyramid",
                     f"The working image is analysed at scales {scales}. The 1.5x copy makes small damage larger.")
    scaled = {}
    for s in scales:
        im = work if s == 1.0 else work.resize((round(work.width * s), round(work.height * s)), Image.Resampling.BICUBIC)
        scaled[s] = im
        im.save(d / f"scale_{s:.1f}x_{im.width}x{im.height}.png")

    # 05 ---------------------------------------------------------------- tiling
    T, O = cal.tile_size, cal.overlap
    d = steps.folder("step_05_tiling", "Step 5 - Tiling and flip augmentation",
                     f"Each scaled image is cut into {T} x {T} px tiles with {O} px overlap; every tile also runs mirrored (horizontal flip) "
                     "and the two predictions are averaged. Tiles are blended with a Hann window so there are no seams.\n"
                     "grid_*.png shows the tile positions; sample_tile*.png is one tile and its mirrored copy.")
    palette = [(230, 50, 50), (50, 120, 230), (40, 180, 90), (240, 170, 20), (150, 80, 200), (20, 190, 200)]
    for s, im in scaled.items():
        w, h = im.size
        pw, ph = max(w, T), max(h, T)
        grid = im.convert("RGB").copy()
        draw = ImageDraw.Draw(grid)
        for iy, top in enumerate(_starts(ph, T, O)):
            for ix, left in enumerate(_starts(pw, T, O)):
                c = palette[(iy * 3 + ix) % len(palette)]
                draw.rectangle([left, top, left + T - 1, top + T - 1], outline=c, width=4)
                draw.text((left + 8, top + 8), f"r{iy}c{ix}", fill=c)
        grid.save(d / f"grid_{s:.1f}x.png")
    tile = scaled[1.0].crop((0, 0, min(T, work.width), min(T, work.height)))
    tile.save(d / "sample_tile.png")
    ImageOps.mirror(tile).save(d / "sample_tile_flipped.png")

    # 06 ---------------------------------------------------------------- probability maps
    pyramid = predict_probability_pyramid(
        model, work, device=predictor.device, scale_factors=scales, tile_size=T, overlap=O,
        horizontal_flip_tta=cal.horizontal_flip_tta, mixed_precision=cal.mixed_precision)
    d = steps.folder("step_06_probability_maps", "Step 6 - Dense probability maps from the ONNX damage model",
                     "Three heads per pixel: presence (damage yes/no), type (5 classes), exterior (is this vehicle surface).\n"
                     "any-damage score = (1 - p_background) x (0.5 + 0.5 x p_exterior). Blue = 0, red = 1.\n"
                     "Files: score per scale, the fused score (average of scales), exterior map, and the dominant damage type.")
    for s, (dm, ex) in pyramid.items():
        sc = _score_map(dm, ex, cal.exterior_floor)
        heat(sc).save(d / f"score_{s:.1f}x.png")
        blend(work, heat(sc)).save(d / f"score_{s:.1f}x_on_photo.png")
    fused = fuse_probability_pyramid(pyramid, scales)
    damage, exterior = restore_size(fused, image.size)
    score = _score_map(damage, exterior, cal.exterior_floor)
    heat(score).save(d / "score_fused_original_size.png")
    blend(image, heat(score)).save(d / "score_fused_on_photo.png")
    heat(exterior.numpy()).save(d / "exterior_map.png")
    typ = damage[1:].argmax(0).numpy() + 1
    Image.fromarray(np.where(score[..., None] >= thr, np.array([CLASS_COLORS[(t - 1) % 6] for t in typ.ravel()]).reshape(*typ.shape, 3), 0).astype(np.uint8)).save(d / "dominant_type_where_above_threshold.png")

    # 07 ---------------------------------------------------------------- threshold and regions
    mask, uncertainty = probabilities_to_mask(damage, exterior, thr, cal.exterior_floor, cal.minimum_component_pixels,
                                              cal.type_probability_multipliers)
    d = steps.folder("step_07_threshold_and_regions", f"Step 7 - Threshold {thr:.3f} and region filtering",
                     f"Pixels with score >= {thr:.3f} are damage candidates. Connected regions (8-neighbour) smaller than "
                     f"{cal.minimum_component_pixels} px are dropped. Each region takes the damage type most of its pixels voted for.\n"
                     "Legend: " + ", ".join(f"{classes[i + 1]} = RGB{CLASS_COLORS[i % 6]}" for i in range(len(classes) - 1)))
    Image.fromarray(((score >= thr) * 255).astype(np.uint8)).save(d / "above_threshold_raw.png")
    mimg = np.zeros((*mask.shape, 3), np.uint8)
    for cid in np.unique(mask):
        if cid:
            mimg[mask == cid] = CLASS_COLORS[(int(cid) - 1) % 6]
    blend(image, Image.fromarray(mimg), 0.5).save(d / "regions_on_photo.png")
    Image.fromarray(mimg).save(d / "regions_mask.png")

    # 08 ---------------------------------------------------------------- glare filter
    zone = glare_zone(image)
    kept, removed = split_glare_components(mask, zone)
    d = steps.folder("step_08_glare_filter", "Step 8 - Glare / light filter",
                     "Lamps, LED strips and reflections are near-white blobs the model can misread as damage. A halo around clipped pixels "
                     "(red in glare_zone.png) is computed; a damage region that lies >= 85% inside it is NOT trusted as damage "
                     "and the photo is sent to manual review instead.\n"
                     f"Regions kept as damage: {int((ndimage.label(kept > 0)[1]))}; regions moved to review as glare: {int((ndimage.label(removed > 0)[1]))}.")
    gz = np.asarray(image).copy()
    gz[zone] = (0.5 * gz[zone] + 0.5 * np.array([255, 0, 0])).astype(np.uint8)
    Image.fromarray(gz).save(d / "glare_zone.png")
    for name, m in (("kept_as_damage", kept), ("removed_as_glare", removed)):
        img = np.zeros((*m.shape, 3), np.uint8)
        for cid in np.unique(m):
            if cid:
                img[m == cid] = CLASS_COLORS[(int(cid) - 1) % 6]
        blend(image, Image.fromarray(img), 0.5).save(d / f"{name}.png")

    # 09 ---------------------------------------------------------------- part segmentation
    names = predictor.parts.part_names
    raw_parts = predictor.parts.predict(image)
    d = steps.folder("step_09_part_segmentation", "Step 9 - Car part segmentation (second ONNX model)",
                     "The photo is letterboxed to 644 x 644 and a ViT-S/14 + conv head labels each pixel with one of 21 car parts. "
                     "Output of the model before any correction.\nVisible parts: " + ", ".join(
                         f"{p['part']} {p['area_pct_of_image']:.1f}%" for p in _visible_parts(raw_parts, names)))
    cp = colour_ids(raw_parts, PART_PALETTE)
    cp.save(d / "parts_raw.png")
    blend(image, cp, 0.55).save(d / "parts_raw_on_photo.png")

    # 10 ---------------------------------------------------------------- correction + side
    parts = harmonise_ends(raw_parts, names)
    view = infer_view(parts, names)
    d = steps.folder("step_10_part_correction_and_side", "Step 10 - Front/rear correction and photo angle",
                     "If one end of the car clearly dominates (grille, headlights, bonnet, windshield vs tail lights, boot, rear glass by 3x), "
                     "a bumper labelled for the other end is relabelled. Then the photo angle is inferred from where the front and rear parts "
                     "sit in the image, which gives left/right for each damaged panel.\n"
                     f"Bumper pixels changed: {int((parts != raw_parts).sum())}.\nInferred view: {view['view']} (confidence {view['confidence']:.2f}).")
    cp2 = colour_ids(parts, PART_PALETTE)
    cp2.save(d / "parts_corrected.png")
    blend(image, cp2, 0.55).save(d / "parts_corrected_on_photo.png")
    (d / "view.json").write_text(json.dumps(view, indent=1), encoding="utf-8")

    # 11 ---------------------------------------------------------------- panel assignment + rating
    mask_score = _score_map(damage, exterior, cal.exterior_floor)
    vehicle_pixels = max(1, int((exterior >= 0.5).sum().item()))
    detections = _components(kept, mask_score, classes, kind="segmentation",
                             minimum_pixels=max(1, cal.effective_segmentation_minimum_component_pixels),
                             vehicle_pixels=vehicle_pixels, part_mask=parts, part_names=names)
    for rec in detections:
        x0, _, x1, _ = rec["bbox"]
        rec["side"] = side_of(rec.get("part"), view, (x0 + x1) / 2 / image.width)
    panels = _panel_summary(detections, kept, parts, names, classes)
    layers = _render_layers(image, kept, parts, panels)
    lines = []
    for p in panels:
        lines.append(f"{(p.get('side') or '')} {p['part']}: panel rating {p['rating']} ({p['rating_label']}), damage covers "
                     f"{p['damage_area_pct_of_panel']:.1f}% of the panel")
        for dmg in p["damage"]:
            lines.append(f"    - {dmg['class_name']}: {dmg['area_pct_of_panel']:.1f}% of panel -> {dmg['severity']['score']}  ({dmg['severity']['reason']})")
    d = steps.folder("step_11_panel_assignment_and_rating", "Step 11 - Damage on panels and intensity rating",
                     "Each damage region takes the part under most of its pixels. Damage area is measured as a % of that panel for each damage "
                     "type, then rated 1-5 by the rubric (dent depth x area; scratch < 20% = 1 else 3; paint like scratch; crack/deformation on the d3 row). "
                     "The panel takes its worst rating.\n\n" + ("\n".join(lines) or "No damaged panel identified."))
    layer_panels = decode(layers["panel_overlay_png_b64"])
    base = decode(layers["image_jpeg_b64"]).convert("RGBA")
    Image.alpha_composite(base, layer_panels.convert("RGBA")).convert("RGB").save(d / "damaged_panels_on_photo.png")
    Image.alpha_composite(base, decode(layers["damage_overlay_png_b64"]).convert("RGBA")).convert("RGB").save(d / "damage_mask_on_photo.png")
    (d / "panels.json").write_text(json.dumps(panels, indent=1, default=float), encoding="utf-8")
    (d / "detections.json").write_text(json.dumps(detections, indent=1, default=float), encoding="utf-8")

    # 12 ---------------------------------------------------------------- real predictor, end to end
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    final = predictor.predict_bytes(buffer.getvalue())
    d = steps.folder("step_12_final_report", "Step 12 - Final decision (the real predictor, end to end)",
                     f"Decision: {final['decision']}.  Reason: {final['reason']}\n"
                     f"Decision reasons: {', '.join(final['decision_reasons']) or 'none'}.\n"
                     f"Overall severity: {final['overall_severity']}.  Photo angle: {final['view']['view']}.\n"
                     f"Damaged panels: {', '.join((p.get('side') or '') + ' ' + p['part'] for p in final['panels']) or 'none'}.\n"
                     f"Model time on this device: {final['runtime']['elapsed_ms'] / 1000:.1f} s ({final['runtime']['device']}).\n\n"
                     f"Check against the manual steps above: panels match = {[p['part'] for p in final['panels']] == [p['part'] for p in panels]}.")
    decode(final["annotated_jpeg_b64"]).convert("RGB").save(d / "annotated_final.png")
    slim = {k: v for k, v in final.items() if k not in ("annotated_jpeg_b64", "layers")}
    (d / "result.json").write_text(json.dumps(slim, indent=1, default=float), encoding="utf-8")

    (out / "README.md").write_text(
        f"# Pipeline steps for {photo.name}\n\nEach folder is one stage of the damage pipeline, in the order the app runs it.\n\n"
        + "\n".join(f"- `{n}` - {t}" for n, t in steps.index) + "\n", encoding="utf-8")
    print("wrote", len(steps.index), "step folders to", out)


if __name__ == "__main__":
    main()
