"""Command-line inference for the pinned vehicle damage model."""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

from .api import DamagePredictor


def _write_report(report: dict, output_dir: Path, stem: str, write_annotated: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = dict(report)
    annotated = payload.pop("annotated_jpeg_b64", None)
    (output_dir / f"{stem}.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    if write_annotated and annotated:
        (output_dir / f"{stem}.jpg").write_bytes(base64.b64decode(annotated))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run the pinned DINOv2 damage segmentation model on images."
    )
    parser.add_argument("images", nargs="+", help="JPEG, PNG, or WebP image paths")
    parser.add_argument(
        "--output-dir",
        default="predictions",
        help="Directory for JSON reports and annotated images (default: predictions)",
    )
    parser.add_argument(
        "--device",
        default=None,
        choices=("auto", "cuda", "gpu", "npu", "cpu"),
        help="Inference device (default: VEHICLE_DAMAGE_DEVICE or auto)",
    )
    parser.add_argument(
        "--no-annotated",
        action="store_true",
        help="Do not write the annotated JPEG overlay",
    )
    args = parser.parse_args(argv)

    predictor = DamagePredictor(device=args.device)
    output_dir = Path(args.output_dir)
    for image_path in args.images:
        path = Path(image_path)
        report = predictor.predict_bytes(path.read_bytes())
        report["source"] = f"file:{path.name}"
        _write_report(report, output_dir, path.stem, not args.no_annotated)
        print(
            f"{path.name}: {report['decision']} "
            f"(detections={report['n_detections']}, "
            f"{report['runtime']['elapsed_ms']:.0f} ms, {report['runtime']['device']})"
        )


if __name__ == "__main__":
    main()
