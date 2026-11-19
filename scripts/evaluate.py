"""Evaluate a trained run on the official test splits.

    python scripts/evaluate.py --run outputs/film_cf
    python scripts/evaluate.py --run outputs/film_cf --figure 8   # also save qualitative results

Writes ``test_metrics.json`` (and ``qualitative_<split>.png``) into the run folder.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lgrasp.engine import evaluate, make_loader  # noqa: E402
from lgrasp.metrics import harmonic_mean  # noqa: E402
from lgrasp.models.grasp_net import build_model  # noqa: E402
from lgrasp.text import TextEmbeddings  # noqa: E402
from lgrasp.utils.visualize import save_prediction_grid  # noqa: E402

TEST_SPLITS = ("test_seen", "test_unseen")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True, help="run folder created by train.py")
    parser.add_argument("--checkpoint", default="best.pt")
    parser.add_argument("--figure", type=int, default=0, help="number of qualitative examples to save per split")
    parser.add_argument("--root", type=Path, default=None, help="dataset folder (default: the one used for training)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.run / args.checkpoint, map_location=device, weights_only=True)
    config = checkpoint["config"]
    if args.root is not None:
        config["data"]["root"] = str(args.root)

    embeddings = TextEmbeddings(config["data"]["root"])
    model = build_model(config["model"], checkpoint["text_dim"]).to(device)
    model.load_state_dict(checkpoint["model"])

    results = {"name": config["name"], "epoch": checkpoint["epoch"]}
    for split in TEST_SPLITS:
        loader = make_loader(config, split)
        results[split], records = evaluate(model, loader, embeddings, device, collect=args.figure)
        if records:
            save_prediction_grid(loader.dataset, records, args.run / f"qualitative_{split}.png")
    results["H"] = harmonic_mean(results["test_seen"]["success_rate"], results["test_unseen"]["success_rate"])

    with open(args.run / "test_metrics.json", "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    print(f"{config['name']} (epoch {checkpoint['epoch']})")
    print("split         SR     ignore  paired   samples")
    for split in TEST_SPLITS:
        m = results[split]
        print(f"{split:<12}{m['success_rate']:6.3f}{m['ignore_rate']:9.3f}{m['paired_sr']:8.3f}{m['samples']:9d}")
    print(f"H = {results['H']:.3f}")


if __name__ == "__main__":
    main()
