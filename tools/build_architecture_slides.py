"""Insert technical-architecture slides into the Maruti/Infosys concept deck.

Drives PowerPoint through COM (the source deck is an .odp). Works on a copy and
saves a new .pptx; the original file is never modified.

Usage: python tools/build_architecture_slides.py SOURCE.odp OUTPUT.pptx
"""

from __future__ import annotations

import sys

import win32com.client

SRC, OUT = sys.argv[1], sys.argv[2]
INSERT_AFTER = 3  # after "End-to-end solution flow"


def rgb(r, g, b):  # PowerPoint colours are BGR integers
    return r + g * 256 + b * 65536


NAVY, TEAL, DARK = rgb(36, 69, 142), rgb(0, 166, 178), rgb(18, 58, 120)
INK, GREY, LINE = rgb(23, 33, 59), rgb(91, 101, 119), rgb(214, 221, 232)
WHITE, PALE, AMBER_BG, AMBER = rgb(255, 255, 255), rgb(231, 247, 248), rgb(255, 245, 226), rgb(236, 190, 110)
GREEN, RED, ORANGE, PURPLE = rgb(34, 150, 94), rgb(201, 48, 44), rgb(232, 113, 10), rgb(112, 84, 200)
FONT = "Aptos"


class Deck:
    def __init__(self, slide):
        self.s = slide

    def text(self, x, y, w, h, lines, align=1, anchor=1, margin=2, space_after=0):
        """lines: str or list of (text, size, bold, colour)."""
        if isinstance(lines, str):
            lines = [(lines, 10, False, INK)]
        box = self.s.Shapes.AddTextbox(1, x, y, w, h)
        tf = box.TextFrame
        tf.WordWrap, tf.AutoSize = True, 0
        tf.MarginLeft = tf.MarginRight = tf.MarginTop = tf.MarginBottom = margin
        tf.VerticalAnchor = anchor
        tf.TextRange.Text = "\r".join(t for t, *_ in lines)
        for i, (_, size, bold, colour) in enumerate(lines, 1):
            p = tf.TextRange.Paragraphs(i)
            p.Font.Name, p.Font.Size, p.Font.Bold, p.Font.Color.RGB = FONT, size, bold, colour
            p.ParagraphFormat.Alignment = align
            p.ParagraphFormat.SpaceAfter = space_after
        box.Height = h
        return box

    def rect(self, x, y, w, h, fill=WHITE, line=LINE, rounded=True, weight=0.75, transparency=0.0):
        shp = self.s.Shapes.AddShape(5 if rounded else 1, x, y, w, h)
        if rounded:
            try:
                shp.Adjustments[1] = min(0.5, 6.0 / min(w, h))  # ~6 pt radius like the deck's cards
            except Exception:
                pass
        shp.Fill.ForeColor.RGB, shp.Fill.Transparency = fill, transparency
        if line is None:
            shp.Line.Visible = False
        else:
            shp.Line.ForeColor.RGB, shp.Line.Weight = line, weight
        shp.Shadow.Visible = False
        return shp

    def arrow(self, x1, y1, x2, y2, colour=NAVY, weight=1.75):
        c = self.s.Shapes.AddConnector(1, x1, y1, x2, y2)
        c.Line.ForeColor.RGB, c.Line.Weight, c.Line.EndArrowheadStyle = colour, weight, 2
        return c

    def card(self, x, y, w, h, title, body, accent=NAVY, body_size=9.5, title_size=12):
        self.rect(x, y, w, h)
        self.rect(x, y, 5, h, fill=accent, line=None, rounded=False)
        self.text(x + 14, y + 8, w - 24, 20, [(title, title_size, True, INK)])
        lines = [(t, body_size, False, GREY) for t in (body if isinstance(body, list) else [body])]
        self.text(x + 14, y + 8 + title_size + 12, w - 24, h - title_size - 22, lines, space_after=5)

    def block(self, x, y, w, h, title, sub, fill, ink=WHITE, n=None, sub_size=8.5):
        self.rect(x, y, w, h, fill=fill, line=None)
        lines = [(title, 10.5, True, ink)] + ([(sub, sub_size, False, ink)] if sub else [])
        self.text(x + 4, y + 2, w - 8, h - 4, lines, align=2, anchor=3)
        if n is not None:
            dot = self.s.Shapes.AddShape(9, x - 7, y - 7, 18, 18)
            dot.Fill.ForeColor.RGB, dot.Line.ForeColor.RGB, dot.Line.Weight = WHITE, fill, 1.5
            dot.Shadow.Visible = False
            tr = dot.TextFrame.TextRange
            tr.Text = str(n)
            tr.Font.Size, tr.Font.Bold, tr.Font.Name, tr.Font.Color.RGB = 8.5, True, FONT, fill
            dot.TextFrame.MarginLeft = dot.TextFrame.MarginRight = dot.TextFrame.MarginTop = dot.TextFrame.MarginBottom = 0


def new_slide(pres, layout, position, title, subtitle):
    slide = pres.Slides.AddSlide(position, layout)
    for shp in list(slide.Shapes):  # drop any empty layout placeholders except the slide number
        if shp.Type == 14 and shp.PlaceholderFormat.Type != 13 and not (shp.HasTextFrame and shp.TextFrame.HasText):
            shp.Delete()
    d = Deck(slide)
    num = slide.Shapes.AddTextbox(1, 880, 508, 52, 18)
    num.TextFrame.TextRange.InsertSlideNumber()
    nf = num.TextFrame.TextRange
    nf.Font.Name, nf.Font.Size, nf.Font.Color.RGB = FONT, 7, rgb(113, 128, 147)
    nf.ParagraphFormat.Alignment = 3
    d.text(37, 52, 880, 34, [(title, 24, True, INK)], margin=2)
    d.text(39, 88, 880, 22, [(subtitle, 10.5, False, GREY)], margin=2)
    return d


# ----------------------------------------------------------------------------- slides
def slide_overview(pres, layout, pos):
    d = new_slide(pres, layout, pos, "Technical architecture: system overview",
                  "A local FastAPI service runs two ONNX graphs on the GPU; the browser renders the report from its JSON.")
    w, h, gap = 195, 74, 33
    xs = [40 + i * (w + gap) for i in range(4)]
    row1 = [("Browser UI", "Multi-photo upload, gallery, heatmap, matrix (HTML/JS/SVG)", NAVY),
            ("FastAPI service", "POST /api/damage/predict, thread-safe predictor, size and format checks", NAVY),
            ("Quality gate", "Resolution, exposure, glare and sharpness metrics on the raw photo", TEAL),
            ("Scale pyramid + tiling", "1.0x and 1.5x, 768 px tiles with 192 px overlap", TEAL)]
    row2 = [("Rating engine", "Rubric: damage % of panel per type, panel = worst", PURPLE),
            ("Part model (ONNX)", "ViT-S/14 + conv head, 644 px letterbox, 22 classes", PURPLE),
            ("Post-processing", "Score map, threshold 0.381, components, fail-closed routing", DARK),
            ("Damage model (ONNX)", "DINOv2 ViT-S/14, 3 dense heads, flip TTA, Hann blending", DARK)]
    for i, (t, s, c) in enumerate(row1):
        d.block(xs[i], 135, w, h, t, s, c, n=i + 1)
        if i < 3:
            d.arrow(xs[i] + w, 172, xs[i + 1], 172)
    d.arrow(xs[3] + w / 2, 135 + h, xs[3] + w / 2, 235)
    for i, (t, s, c) in enumerate(row2):
        d.block(xs[i], 235, w, h, t, s, c, n=8 - i)
        if i < 3:
            d.arrow(xs[i + 1], 272, xs[i] + w, 272)
    d.arrow(xs[0] + w / 2, 235 + h, xs[0] + w / 2, 345)
    d.rect(40, 345, 450, 78, fill=PALE, line=TEAL)
    d.text(54, 351, 424, 66, [("JSON response", 11, True, INK),
                              ("decision, detections, panels with per-type rating, parts visible, quality metrics, "
                               "plus the photo and transparent overlay layers as base64", 10.2, False, GREY)])
    d.card(520, 345, 385, 112, "Technology stack",
           ["FastAPI + Uvicorn  |  ONNX Runtime 1.30 + Qualcomm QNN provider",
            "DINOv2 ViT-S/14 (timm)  |  NumPy, SciPy, Pillow",
            "PyTorch only for training and ONNX export",
            "No cloud calls: all inference runs on the local device"], accent=TEAL, body_size=10.2)
    d.text(40, 435, 450, 40, [("Numbers show execution order. Model files are SHA-256 pinned and verified at load.", 9, False, GREY)])


def slide_pipeline(pres, layout, pos):
    d = new_slide(pres, layout, pos, "How each photo is processed",
                  "Seven stages between the uploaded file and the dense damage probabilities.")
    steps = [("Decode and sanitise", "EXIF rotation, RGB; JPEG/PNG/WebP, 20 MB and 50 MP limits"),
             ("Quality gate", "Short side < 480 px, > 35% dark, > 20% clipped, > 1.5% specular, blur: recapture or review"),
             ("Normalise", "ImageNet mean and standard deviation"),
             ("Scale pyramid", "1.0x and 1.5x (bicubic); results resampled back to photo size"),
             ("Tile", "768 x 768 windows, 192 px overlap, edge-padded; thin scratches stay at native detail"),
             ("Flip test-time augmentation", "Each tile and its mirror image through the ONNX graph, averaged"),
             ("Fuse", "Hann-window blending removes tile seams; scales averaged into full-size maps")]
    y = 124
    for i, (t, s) in enumerate(steps, 1):
        d.block(40, y + 2, 26, 26, "", "", TEAL if i != 2 else ORANGE)
        d.text(40, y + 2, 26, 26, [(str(i), 11, True, WHITE)], align=2, anchor=3, margin=0)
        d.text(76, y - 2, 520, 50, [(t, 11, True, INK), (s, 9.5, False, GREY)])
        y += 52
    # tile illustration
    d.rect(625, 128, 285, 190, fill=rgb(236, 240, 246), line=LINE)
    d.text(625, 132, 285, 16, [("Photo (any size)", 9, False, GREY)], align=2)
    tiles = [(640, 152, 150, 120, TEAL), (730, 152, 150, 120, NAVY), (640, 200, 150, 112, PURPLE), (730, 200, 150, 112, ORANGE)]
    for x, y0, w, h, c in tiles:
        r = d.rect(x, y0, w, h, fill=c, line=c, rounded=False, transparency=0.78)
        r.Line.Weight = 1.5
    d.text(625, 320, 285, 16, [("Overlapping 768 px tiles (192 px overlap)", 9, False, GREY)], align=2)
    d.card(625, 348, 285, 128, "Why this design",
           ["768 px tiles keep scratches visible at native resolution.",
            "The 1.5x pass helps small damage.",
            "Overlap + blending avoids seam artefacts; flip averaging stabilises masks."], accent=TEAL, body_size=10)


def slide_model(pres, layout, pos):
    d = new_slide(pres, layout, pos, "Damage model: DINOv2 ViT-S/14 with three dense heads",
                  "A vision transformer feature extractor with lightweight convolutional heads; 24.7 M parameters.")
    y, h = 128, 84
    d.block(40, y, 104, h, "Input tile", "3 x 768 x 768", NAVY)
    d.block(166, y, 128, h, "Patch embedding", "14 x 14, stride 14 (pad to 770)", NAVY)
    d.block(316, y, 128, h, "12 transformer blocks", "dim 384, 6 heads, 22.1 M params", DARK)
    d.block(466, y, 112, h, "Patch map", "384 x 55 x 55", NAVY)
    heads = [("Presence", "1 channel", TEAL), ("Damage type", "5 classes", PURPLE), ("Vehicle exterior", "1 channel", ORANGE)]
    for i, (t, s, c) in enumerate(heads):
        d.block(602, y - 6 + i * 31, 170, 28, t + "  (" + s + ")", "", c)
        d.arrow(578, y + h / 2, 602, y - 6 + i * 31 + 14, colour=GREY, weight=1.0)
    d.block(796, y, 114, h, "Upsample", "bilinear x14 to 768 x 768 logits", NAVY)
    for a, b in ((144, 166), (294, 316), (444, 466), (772, 796)):
        d.arrow(a, y + h / 2, b, y + h / 2)
    d.text(602, y + h + 4, 170, 30, [("each head: Conv3x3 384 to 256, GroupNorm, GELU, Dropout, Conv1x1", 8.5, False, GREY)])
    d.text(40, y + h + 6, 530, 30, [("3,025 tokens + 1 class token per tile; attention over the whole tile gives panel-level context.", 9, False, GREY)])
    d.card(40, 262, 430, 212, "How the heads become probabilities",
           ["p(background) = 1 - sigmoid(presence)",
            "p(type k) = sigmoid(presence) x softmax(type)[k]",
            "any-damage score = (1 - p(background)) x (0.5 + 0.5 x sigmoid(exterior))",
            "The exterior head softly down-weights damage-like marks outside the vehicle.",
            "Classes: dent, scratch, crack or breakage, paint damage, deformation or detachment."],
           accent=TEAL, body_size=10.6)
    d.card(490, 262, 420, 212, "Training and refinement",
           ["Labelled data: 570 train / 98 validation CC0 damage photos, plus clean COCO and user photos as negatives.",
            "Loss: focal + Tversky (recall-oriented) + type cross-entropy + exterior BCE; hard negatives weighted x5.",
            "Latest refinement: last 3 blocks + heads fine-tuned with 251 wild clean photos (Wikimedia) as full-frame hard negatives.",
            "Why: the earlier model flagged about half of clean classic or chrome cars."],
           accent=PURPLE, body_size=10.4)


def slide_decision(pres, layout, pos):
    d = new_slide(pres, layout, pos, "From probabilities to a decision",
                  "Thresholds, region filtering and a fail-closed router: uncertain results go to a human, never to a silent verdict.")
    steps = [("Score map", "score = (1 - p_background) x (0.5 + 0.5 x p_exterior), full photo size"),
             ("Threshold", "score >= 0.381, calibrated on held-out data (0.736 for the earlier model)"),
             ("Connected components", "8-neighbour regions; regions under 1,200 px are dropped"),
             ("Per-region record", "class by pixel majority, bounding box, mean score, area"),
             ("Branch comparison", "triage branch vs segmentation branch: any one-sided evidence is kept for review")]
    y = 126
    for i, (t, s) in enumerate(steps, 1):
        d.rect(40, y, 400, 62, fill=WHITE)
        d.rect(40, y, 5, 62, fill=TEAL, line=None, rounded=False)
        d.text(54, y + 5, 380, 54, [(f"{i}. {t}", 11.5, True, INK), (s, 10.2, False, GREY)])
        y += 70
    rows = [("damage_detected", "Evidence passes both branches; no quality or review reason.", GREEN, "auto"),
            ("no_damage_detected", "Nothing passes the thresholds.", NAVY, "auto"),
            ("manual_review_required", "Glare, near-threshold area over 10%, one-branch-only evidence, or type disagreement.", ORANGE, "human"),
            ("recapture_required", "Under 480 px, too dark or bright, or blurred: retake the photo.", RED, "retake")]
    y = 126
    for t, s, c, tag in rows:
        d.rect(470, y, 440, 80, fill=WHITE)
        d.rect(470, y, 5, 80, fill=c, line=None, rounded=False)
        d.text(484, y + 6, 330, 20, [(t, 11, True, INK)])
        d.text(484, y + 26, 340, 52, [(s, 10.2, False, GREY)])
        d.block(840, y + 8, 60, 22, tag, "", c)
        y += 88
    d.text(470, 482, 440, 18, [("Both manual_review and recapture set manual_review = true for older clients.", 8.5, False, GREY)])


def slide_parts(pres, layout, pos):
    d = new_slide(pres, layout, pos, "Panel assignment and intensity rating",
                  "A second ONNX graph names the panels; damage area is then measured against each panel and rated by rule.")
    d.card(40, 126, 280, 262, "1. Part segmentation",
           ["Same ViT-S/14 architecture + a conv head; input letterboxed to 644 x 644 (a single pass per photo).",
            "22 classes: 21 panels (bonnet, doors, fenders, bumpers, windshield, mirrors, lights, wheels...) + background.",
            "Trained on 998 CC0 part-labelled photos; held-out mIoU 0.62.",
            "Strong: bonnet 0.88, windshield 0.91, bumpers about 0.78. Weak: side mirror 0.28, front window 0.23.",
            "Left and right are not distinguished."], accent=PURPLE, body_size=10.6)
    d.card(335, 126, 270, 262, "2. Damage on panel",
           ["Each damage region takes the panel under most of its pixels (at least 20% overlap).",
            "Area % of panel is computed per damage type as the union of its fragments on that panel.",
            "Rating per type from the rubric; the panel shows its worst rating.",
            "Output: panel, types with their own ratings, % of panel, bounding box, photo count.",
            "Heatmap and matrix aggregate the worst rating over all uploaded photos."], accent=TEAL, body_size=10.6)
    d.rect(620, 126, 290, 262)
    d.rect(620, 126, 5, 262, fill=NAVY, line=None, rounded=False)
    d.text(634, 133, 270, 20, [("3. Intensity rules (supplied rubric)", 12, True, INK)])
    tbl = d.s.Shapes.AddTable(4, 6, 634, 160, 266, 96).Table
    cells = [["Dent", "0-20%", "20-40", "40-60", "60-80", "80-100"], ["d1", "1", "2", "3", "4", "4"],
             ["d2", "2", "3", "4", "5", "5"], ["d3", "3", "4", "5", "5", "5"]]
    for r in range(4):
        for c in range(6):
            cell = tbl.Cell(r + 1, c + 1)
            cell.Shape.TextFrame.TextRange.Text = cells[r][c]
            f = cell.Shape.TextFrame.TextRange.Font
            f.Name, f.Size, f.Bold = FONT, 8.5, r == 0 or c == 0
            f.Color.RGB = WHITE if r == 0 else INK
            cell.Shape.Fill.ForeColor.RGB = NAVY if r == 0 else (rgb(240, 244, 250) if r % 2 else WHITE)
            cell.Shape.TextFrame.MarginLeft = cell.Shape.TextFrame.MarginRight = 2
            cell.Shape.TextFrame.TextRange.ParagraphFormat.Alignment = 2
    d.text(634, 262, 270, 124, [
        ("Scratch: under 20% of panel = 1, otherwise 3.", 10.3, False, GREY),
        ("Added, not in the rubric:", 10.3, True, INK),
        ("paint damage rated like a scratch; crack, breakage and deformation rated on the d3 row by panel area.", 10.3, False, GREY),
        ("Not yet modelled:", 10.3, True, INK),
        ("dent depth (assumed d2 and flagged), left/right side, cost.", 10.3, False, GREY)])
    d.rect(40, 400, 870, 76, fill=PALE, line=TEAL)
    d.text(54, 405, 842, 68, [("Worked example (real output, torn-bumper sample)", 11, True, INK),
        ("Front bumper: crack or breakage covers 62% of the panel -> d3 row, 60-80% column -> 5; paint damage 0% -> 1; panel = 5.   "
         "Rear bumper: crack 24% -> 4.   Fender: crack 10% -> 3.   Each type is rated on its own total area.", 10.3, False, GREY)])


def slide_runtime(pres, layout, pos):
    d = new_slide(pres, layout, pos, "Runtime, deployment and measured performance",
                  "Fixed-shape ONNX graphs on the Adreno GPU through ONNX Runtime's Qualcomm plugin, with integrity checks at load.")
    d.card(40, 126, 270, 214, "ONNX export",
           ["Opset 17, one fixed 768 x 768 tile; tiling and TTA run outside the graph.",
            "timm's anti-aliased bicubic position-embedding resize has no ONNX operator, so the table is computed once and baked in.",
            "Parity vs PyTorch: max logit error 1.2e-5 (presence), 6.7e-6 (type).",
            "Files: damage graph 101.6 MB, part graph 93.0 MB."], accent=NAVY, body_size=10)
    d.card(325, 126, 285, 214, "Execution and integrity",
           ["ONNX Runtime 1.30 + onnxruntime-qnn 2.7: Adreno GPU (FP32) is the default; CPU fallback.",
            "Hexagon NPU needs a compiled FP16 context graph (not built for the current model).",
            "Checkpoint, profile, ONNX and part model are SHA-256 pinned; loading fails on any mismatch.",
            "A warm-up photo runs at start-up so the first user request is not the cold one."], accent=TEAL, body_size=10)
    d.rect(625, 126, 285, 214)
    d.rect(625, 126, 5, 214, fill=PURPLE, line=None, rounded=False)
    d.text(639, 133, 262, 20, [("Time per photo (same image)", 12, True, INK)])
    maxw = 220.0
    for i, (label, sec, c) in enumerate((("Adreno GPU", 24, GREEN), ("CPU", 284, RED))):
        top = 175 + i * 62
        d.text(639, top, 100, 16, [(label, 9.5, True, INK)])
        d.rect(639, top + 18, max(6.0, maxw * sec / 284), 20, fill=c, line=None, rounded=False)
        d.text(639 + max(6.0, maxw * sec / 284) + 4, top + 18, 60, 20, [(f"{sec} s", 10, True, INK)], anchor=3)
    d.text(639, 292, 262, 44, [("Model pass averaged 8.1 s over a 374-photo batch on the GPU; a full API request with report rendering took 13-24 s.", 9, False, GREY)])
    d.rect(40, 356, 870, 84, fill=AMBER_BG, line=AMBER)
    d.text(54, 362, 842, 74, [
        ("What the numbers do and do not show", 11, True, rgb(120, 80, 10)),
        ("Measured on a Snapdragon (Windows on ARM) laptop through ONNX Runtime and the QNN provider. A phone build "
         "(CoreML / LiteRT) has not been exported or parity-tested, and the ViT-S model at two scales with flip averaging "
         "is heavy for a phone: expect a lighter student model or a reduced configuration for true on-device use.", 10.5, False, rgb(110, 75, 20))])


def slide_status(pres, layout, pos):
    d = new_slide(pres, layout, pos, "Prototype today versus the concept flow",
                  "The six processing stages of the concept, and what exists in the working prototype.")
    rows = [("Stage in the concept", "Status", "What exists today"),
            ("1  Proper-vehicle validation", "Partial", "Image-quality gate only; no vehicle-presence or required-angle check"),
            ("2  Panel and damage segmentation", "Built", "DINOv2 damage model + part model; part mIoU 0.62, weak on mirrors and windows"),
            ("3  Dent / scratch / tear classification", "Built, weak", "5 classes; predicted scratch correct 32%, paint 36%; no dent depth"),
            ("4  Severity + confidence", "Built", "Rubric rating per type per panel, score per detection; depth assumed d2"),
            ("5  Pricing engine", "Not built", "No parts, labour or paint rules"),
            ("6  Marker images + report", "Built", "Web report: markers, heatmap, matrix, panel table; no PDF or DMS hand-off")]
    colour = {"Built": GREEN, "Built, weak": ORANGE, "Partial": ORANGE, "Not built": RED}
    tbl = d.s.Shapes.AddTable(len(rows), 3, 40, 124, 870, 230).Table
    tbl.Columns(1).Width, tbl.Columns(2).Width, tbl.Columns(3).Width = 270, 90, 510
    for r, row in enumerate(rows):
        for c, val in enumerate(row):
            cell = tbl.Cell(r + 1, c + 1)
            cell.Shape.TextFrame.TextRange.Text = val
            f = cell.Shape.TextFrame.TextRange.Font
            f.Name, f.Size = FONT, 10
            f.Bold = r == 0 or c == 1
            f.Color.RGB = WHITE if r == 0 else (colour.get(val, INK) if c == 1 else INK)
            cell.Shape.Fill.ForeColor.RGB = NAVY if r == 0 else (rgb(244, 247, 251) if r % 2 else WHITE)
            cell.Shape.TextFrame.VerticalAnchor = 3
    d.rect(40, 378, 425, 96, fill=PALE, line=TEAL)
    d.text(54, 384, 400, 88, [("Measured in the real pipeline", 11, True, INK),
                              ("203 held-out images: damaged cases caught 88.9% (95% CI 80-94%); clean photos falsely flagged 2.5% "
                               "(CI 0.8-7.0%). The earlier model on the same images: 96.3% and 29.5%.", 10.4, False, GREY)])
    d.rect(485, 378, 425, 96, fill=AMBER_BG, line=AMBER)
    d.text(499, 384, 400, 88, [("Next to reach production level", 11, True, rgb(120, 80, 10)),
                               ("First-party phone photos labelled by assessors; more scratch / paint / dent data; severity compared "
                                "with human ratings; mobile export and parity tests; service security and monitoring.",
                                10.4, False, rgb(110, 75, 20))])


def main():
    app = win32com.client.Dispatch("PowerPoint.Application")
    pres = app.Presentations.Open(SRC, ReadOnly=False, WithWindow=False)
    pres.SaveAs(OUT, 24)  # convert to .pptx first; the source file is never written
    layout = pres.Slides(8).CustomLayout
    builders = [slide_overview, slide_pipeline, slide_model, slide_decision, slide_parts, slide_runtime, slide_status]
    for offset, build in enumerate(builders, 1):
        build(pres, layout, INSERT_AFTER + offset)
    pres.Save()
    print("slides now:", pres.Slides.Count)
    pres.Close()
    app.Quit()


if __name__ == "__main__":
    main()
