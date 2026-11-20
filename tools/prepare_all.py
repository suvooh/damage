"""Run the source repo's damage + parts preparation against a local archive,
then write multi-class part masks. Usage:
  python tools/prepare_all.py --archive A.zip --source <Port/..> --out <dir>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from localzip import LocalZip  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--archive", required=True)
    p.add_argument("--source", required=True, help="Port/vehicle-damage-segmentation")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    sys.path.insert(0, str(Path(args.source) / "scripts"))
    import prepare_hitl_cc0 as damage
    import prepare_hitl_parts_cc0 as parts

    make = lambda url: LocalZip(args.archive)  # noqa: E731
    damage.RemoteZip = make
    parts.RemoteZip = make
    out = Path(args.out)
    damage.prepare(argparse.Namespace(output=str(out / "hitl_cc0"), limit=0, max_side=1600, seed=17, workers=4))
    parts.prepare(argparse.Namespace(output=str(out / "hitl_parts_cc0"), limit=0, max_side=1200, seed=23, workers=4))


if __name__ == "__main__":
    main()
