"""Check a downloaded dataset and write ``manifest.json`` and ``verify_report.json``.

    python scripts/verify_data.py --root data/grasp_anything_pp

Exits with status 1 if the dataset is not usable.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lgrasp.data.manifest import save_manifest  # noqa: E402
from lgrasp.data.splits import load_all_splits  # noqa: E402
from lgrasp.data.verify import verify_dataset  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("data/grasp_anything_pp"))
    parser.add_argument(
        "--partial", action="store_true",
        help="the download was limited with --limit-scenes: do not fail on scenes that were not requested",
    )
    args = parser.parse_args()

    report, manifest = verify_dataset(args.root, load_all_splits())
    if args.partial:
        report["errors"] = [e for e in report["errors"] if "no image on disk" not in e]

    with open(args.root / "verify_report.json", "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k not in ("errors", "warnings")}, indent=2))
    for warning in report["warnings"]:
        print(f"WARNING  {warning}")
    for error in report["errors"]:
        print(f"ERROR    {error}")

    if report["errors"]:
        print("\nDataset is NOT usable. Fix the errors above (usually: re-run scripts/download_data.py).")
        return 1
    path = save_manifest(manifest, args.root)
    print(f"\nDataset OK. Wrote {path} with {len(manifest['samples'])} samples.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
