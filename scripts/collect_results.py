"""Print the results of all evaluated runs as one Markdown table.

    python scripts/collect_results.py --outputs outputs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--outputs", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    rows = []
    for path in sorted(args.outputs.glob("*/test_metrics.json")):
        with open(path, encoding="utf-8") as file:
            result = json.load(file)
        seen, unseen = result["test_seen"], result["test_unseen"]
        rows.append(
            f"| {result['name']} | {seen['success_rate']:.3f} | {unseen['success_rate']:.3f} | {result['H']:.3f} "
            f"| {seen['ignore_rate']:.3f} | {unseen['ignore_rate']:.3f} "
            f"| {seen['paired_sr']:.3f} | {unseen['paired_sr']:.3f} |"
        )
    if not rows:
        print(f"No test_metrics.json found under {args.outputs}/")
        return
    print("| Run | SR seen | SR unseen | H | Ignore seen | Ignore unseen | Paired seen | Paired unseen |")
    print("|---|---|---|---|---|---|---|---|")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
