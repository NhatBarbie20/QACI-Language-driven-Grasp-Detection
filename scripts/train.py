"""Train one experiment.

    python scripts/train.py --config configs/film_cf.yaml
    python scripts/train.py --config configs/film.yaml --set train.epochs=10 train.seed=1

Writes to ``<output_dir>/<name>/``:
    config.yaml   the merged configuration
    log.jsonl     one line per epoch (losses and validation metrics)
    best.pt       checkpoint with the best validation success rate
    last.pt       checkpoint of the last epoch
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lgrasp.config import load_config  # noqa: E402
from lgrasp.engine import evaluate, make_loader, seed_everything, train_one_epoch  # noqa: E402
from lgrasp.models.grasp_net import build_model  # noqa: E402
from lgrasp.text import TextEmbeddings  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override config values")
    args = parser.parse_args()

    config = load_config(args.config, args.set)
    settings = config["train"]
    seed_everything(settings["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_dir = Path(settings["output_dir"]) / config["name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False)

    embeddings = TextEmbeddings(config["data"]["root"])
    train_loader = make_loader(config, "train")
    val_loader = make_loader(config, "val")
    model = build_model(config["model"], embeddings.dim).to(device)
    parameters = sum(p.numel() for p in model.parameters())
    print(f"{config['name']}: {parameters / 1e6:.2f}M parameters, {len(train_loader.dataset)} training samples, device {device}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["lr"], weight_decay=settings["weight_decay"])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=settings["lr"], total_steps=settings["epochs"] * len(train_loader), pct_start=0.05
    )
    scaler = torch.amp.GradScaler(enabled=settings["amp"] and device.type == "cuda")

    best = -1.0
    for epoch in range(1, settings["epochs"] + 1):
        started = time.time()
        losses = train_one_epoch(model, train_loader, embeddings, optimizer, scheduler, scaler, config, device)
        metrics, _ = evaluate(model, val_loader, embeddings, device)

        record = {"epoch": epoch, "seconds": round(time.time() - started, 1), "train": losses, "val": metrics}
        with open(run_dir / "log.jsonl", "a", encoding="utf-8") as file:
            file.write(json.dumps(record) + "\n")

        checkpoint = {"model": model.state_dict(), "config": config, "epoch": epoch, "val": metrics,
                      "text_dim": embeddings.dim}
        torch.save(checkpoint, run_dir / "last.pt")
        if metrics["success_rate"] > best:
            best = metrics["success_rate"]
            torch.save(checkpoint, run_dir / "best.pt")

        print(
            f"epoch {epoch:3d}  loss {losses['total']:.4f}"
            + (f"  cf {losses['counterfactual']:.4f}" if "counterfactual" in losses else "")
            + f"  val SR {metrics['success_rate']:.3f}  ignore {metrics['ignore_rate']:.3f}"
            + f"  paired {metrics['paired_sr']:.3f}  ({record['seconds']:.0f}s)"
        )

    print(f"Best validation success rate: {best:.3f}. Next: python scripts/evaluate.py --run {run_dir}")


if __name__ == "__main__":
    main()
