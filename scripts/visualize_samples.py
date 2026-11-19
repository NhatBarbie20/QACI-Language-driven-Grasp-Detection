"""Save a grid of dataset samples with their ground-truth grasps.

    python scripts/visualize_samples.py --split train --num 12
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lgrasp.data.dataset import SPLITS, GraspAnythingPP  # noqa: E402
from lgrasp.utils.visualize import save_sample_grid  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("data/grasp_anything_pp"))
    parser.add_argument("--split", default="train", choices=SPLITS)
    parser.add_argument("--num", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    dataset = GraspAnythingPP(args.root, args.split)
    indices = random.Random(args.seed).sample(range(len(dataset)), min(args.num, len(dataset)))
    out = args.out or Path("outputs") / f"samples_{args.split}.png"
    print(f"{len(dataset)} samples in '{args.split}'. Saved {save_sample_grid(dataset, indices, out)}")


if __name__ == "__main__":
    main()
