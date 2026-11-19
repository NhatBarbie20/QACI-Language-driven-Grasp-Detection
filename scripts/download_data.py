"""Download the Grasp-Anything++ subset used by the official LGD splits.

Examples
--------
    python scripts/download_data.py                       # everything the splits need (~12k scenes)
    python scripts/download_data.py --limit-scenes 20     # tiny smoke test
    python scripts/download_data.py --components part_mask  # add an optional component later
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lgrasp.data.download import COMPONENTS, DEFAULT_COMPONENTS, download_subset  # noqa: E402
from lgrasp.data.splits import load_all_splits, scene_of, scenes_in_splits  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("data/grasp_anything_pp"), help="output directory")
    parser.add_argument(
        "--components", nargs="+", default=list(DEFAULT_COMPONENTS), choices=sorted(COMPONENTS),
        help="which kinds of files to download",
    )
    parser.add_argument(
        "--limit-scenes", type=int, default=None,
        help="keep only the first N scenes of each split (for quick tests)",
    )
    parser.add_argument("--workers", type=int, default=32, help="parallel download threads")
    parser.add_argument("--token", default=None, help="Hugging Face token (not needed: the dataset is public)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    splits = load_all_splits()
    if args.limit_scenes is not None:
        splits = {
            name: [i for i in ids if scene_of(i) in set(sorted({scene_of(j) for j in ids})[: args.limit_scenes])]
            for name, ids in splits.items()
        }
    scenes = scenes_in_splits(splits)
    print(f"Selected {len(scenes)} scenes from splits: " + ", ".join(f"{k}={len(v)} objects" for k, v in splits.items()))

    results = download_subset(scenes, args.root, args.components, args.workers, args.token)

    print("\ncomponent              expected  downloaded  already-there  failed")
    for result in results:
        print(f"{result.component:<22}{result.expected:>9}{result.downloaded:>12}{result.skipped:>15}{len(result.failed):>8}")
        for message in result.failed[:5]:
            print(f"    {message}")
    if not all(result.ok for result in results):
        print("\nSome files failed. Re-run the same command to retry them.")
        return 1
    print("\nDone. Next: python scripts/verify_data.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
