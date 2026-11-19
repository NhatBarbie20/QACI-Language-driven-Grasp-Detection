"""Re-estimate the BatchNorm running statistics of a trained run.

Training with the counterfactual loss sends two kinds of batches through the
text-conditioned layers (matching and swapped instructions). Their activations
differ strongly, so the exponential running averages kept by BatchNorm end up
describing neither, and the network fails in evaluation mode although it trains
normally. This script recomputes the statistics as a plain average over
training batches with matching instructions ("precise BatchNorm"), which is the
input distribution used at test time. No weight is changed.

    python scripts/recalibrate_bn.py --run outputs/film_cf --batches 200

Writes ``<checkpoint stem>_bn.pt`` next to the original checkpoint.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lgrasp.config import with_defaults  # noqa: E402
from lgrasp.engine import make_loader, seed_everything  # noqa: E402
from lgrasp.models import build_model, load_trainable_state, text_model_name, trainable_state  # noqa: E402
from lgrasp.text import TextEmbeddings  # noqa: E402


@torch.no_grad()
def recalibrate(model: nn.Module, loader, embeddings: TextEmbeddings, device: torch.device, batches: int) -> int:
    """Replace every BatchNorm running statistic by its average over ``batches`` training batches."""
    norms = [module for module in model.modules() if isinstance(module, nn.modules.batchnorm._BatchNorm)]
    model.eval()
    for norm in norms:
        norm.reset_running_stats()
        norm.momentum = None  # cumulative average instead of an exponential one
        norm.train()
    use_text = model.fusion != "none"
    seen = 0
    for batch in loader:
        text = embeddings(batch["text"]).to(device) if use_text else None
        model.decode(model.encode(batch["image"].to(device)), text)
        seen += 1
        if seen >= batches:
            break
    model.eval()
    return seen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--checkpoint", default="best.pt")
    parser.add_argument("--batches", type=int, default=200)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override data loading values")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.run / args.checkpoint, map_location=device, weights_only=True)
    config = with_defaults(checkpoint["config"])
    if args.root is not None:
        config["data"]["root"] = str(args.root)
    for item in args.set:
        dotted, _, value = item.partition("=")
        section, key = dotted.split(".")
        config[section][key] = type(config[section][key])(value) if config[section][key] is not None else int(value)
    config["data"]["all_objects"] = False  # only the images and instructions are needed
    seed_everything(config["train"]["seed"])

    embeddings = TextEmbeddings(config["data"]["root"], text_model_name(config["model"]))
    model = build_model(config["model"], checkpoint["text_dim"]).to(device)
    load_trainable_state(model, checkpoint["model"])

    used = recalibrate(model, make_loader(config, "train"), embeddings, device, args.batches)
    checkpoint["model"] = trainable_state(model)
    checkpoint["bn_recalibrated_batches"] = used
    out = args.run / f"{Path(args.checkpoint).stem}_bn.pt"
    torch.save(checkpoint, out)
    print(f"Recalibrated BatchNorm on {used} batches. Saved {out}")


if __name__ == "__main__":
    main()
