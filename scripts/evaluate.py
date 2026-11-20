"""Evaluate a trained run on the official test splits.

    python scripts/evaluate.py --run outputs/film_cf
    python scripts/evaluate.py --run outputs/film_cf --figure 8   # also save qualitative results
    python scripts/evaluate.py --run outputs/factor_all --grounding none     # geometry factor alone
    python scripts/evaluate.py --run outputs/factor_all --grounding oracle   # ground-truth grounding

Writes ``test_metrics.json`` (and ``qualitative_<split>.png``) into the run folder. With a
``--grounding`` override the file is ``test_metrics_<override>.json`` and no figure is saved.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lgrasp.engine import GROUNDING_OVERRIDES, evaluate, make_loader  # noqa: E402
from lgrasp.metrics import harmonic_mean  # noqa: E402
from lgrasp.config import with_defaults  # noqa: E402
from lgrasp.models import build_model, load_trainable_state, text_model_name  # noqa: E402
from lgrasp.text import TextEmbeddings  # noqa: E402
from lgrasp.utils.visualize import save_prediction_grid  # noqa: E402

TEST_SPLITS = ("test_seen", "test_unseen")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True, help="run folder created by train.py")
    parser.add_argument("--checkpoint", default="best.pt")
    parser.add_argument("--figure", type=int, default=0, help="number of qualitative examples to save per split")
    parser.add_argument("--root", type=Path, default=None, help="dataset folder (default: the one used for training)")
    parser.add_argument("--grounding", default="model", choices=GROUNDING_OVERRIDES,
                        help="diagnostic for factorised models: drop the grounding factor or use the ground truth")
    parser.add_argument("--workers", type=int, default=None, help="data loader workers (default: as in training)")
    parser.add_argument("--max-negative-overlap", type=float, default=None,
                        help="swap the instruction only towards objects whose grasp region overlaps the target at most "
                             "this much (default: as in training); results go to test_metrics_clean.json")
    args = parser.parse_args()
    override = args.grounding != "model"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.run / args.checkpoint, map_location=device, weights_only=True)
    config = with_defaults(checkpoint["config"])
    if args.root is not None:
        config["data"]["root"] = str(args.root)
    if args.workers is not None:
        config["data"]["num_workers"] = args.workers
    clean = args.max_negative_overlap is not None
    if clean:
        config["data"]["max_negative_overlap"] = args.max_negative_overlap

    embeddings = TextEmbeddings(config["data"]["root"], text_model_name(config["model"]), config["model"]["prompt"])
    model = build_model(config["model"], checkpoint["text_dim"]).to(device)
    load_trainable_state(model, checkpoint["model"])

    results = {"name": config["name"], "epoch": checkpoint["epoch"]}
    for split in TEST_SPLITS:
        loader = make_loader(config, split)
        results[split], records = evaluate(
            model, loader, embeddings, device, collect=args.figure, grounding=args.grounding
        )
        if records and not override and not clean:
            save_prediction_grid(loader.dataset, records, args.run / f"qualitative_{split}.png")
    results["H"] = harmonic_mean(results["test_seen"]["success_rate"], results["test_unseen"]["success_rate"])

    if override:
        results["grounding"] = args.grounding
    name = f"test_metrics_{args.grounding}.json" if override else "test_metrics.json"
    if clean:
        results["max_negative_overlap"] = args.max_negative_overlap
        name = name.replace(".json", "_clean.json")
    with open(args.run / name, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    print(f"{config['name']} (epoch {checkpoint['epoch']}, grounding: {args.grounding})")
    print("split         SR     ignore  paired   samples")
    for split in TEST_SPLITS:
        m = results[split]
        print(f"{split:<12}{m['success_rate']:6.3f}{m['ignore_rate']:9.3f}{m['paired_sr']:8.3f}{m['samples']:9d}")
    print(f"H = {results['H']:.3f}")


if __name__ == "__main__":
    main()
