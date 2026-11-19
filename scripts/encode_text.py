"""Encode every grasp instruction once with the frozen CLIP text encoder.

    python scripts/encode_text.py --root data/grasp_anything_pp
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lgrasp.text import DEFAULT_TEXT_MODEL, TextEmbeddings, build_text_embeddings  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("data/grasp_anything_pp"))
    parser.add_argument("--model", default=DEFAULT_TEXT_MODEL)
    args = parser.parse_args()

    path = build_text_embeddings(args.root, args.model)
    table = TextEmbeddings(args.root, args.model)
    print(f"Saved {path}: {len(table.embeddings)} instructions, dimension {table.dim}, model {table.model_name}")


if __name__ == "__main__":
    main()
