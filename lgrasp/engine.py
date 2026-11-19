"""Training and evaluation loops."""

from __future__ import annotations

import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from lgrasp.data.dataset import TARGET_KEYS, GraspAnythingPP, collate
from lgrasp.losses import counterfactual_loss, grasp_loss
from lgrasp.metrics import GraspMeter, decode_grasp, is_correct
from lgrasp.models.grasp_net import GraspNet
from lgrasp.text import TextEmbeddings


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loader(config: dict, split: str) -> DataLoader:
    """Data loader for a split, configured from the ``data`` and ``train`` sections."""
    data = config["data"]
    training = split == "train"
    dataset = GraspAnythingPP(
        data["root"],
        split,
        image_size=data["image_size"],
        augment=data["augment"],
        max_samples=data["max_train_samples"] if training else data["max_eval_samples"],
    )
    return DataLoader(
        dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=training,
        drop_last=training,
        num_workers=data["num_workers"],
        collate_fn=collate,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=data["num_workers"] > 0,
    )


def _embed(texts: list[str], embeddings: TextEmbeddings, device: torch.device) -> torch.Tensor:
    return embeddings(texts).to(device, non_blocking=True)


def train_one_epoch(
    model: GraspNet,
    loader: DataLoader,
    embeddings: TextEmbeddings,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    config: dict,
    device: torch.device,
) -> dict[str, float]:
    """Run one epoch and return the mean of every loss term."""
    model.train()
    cf_weight, cf_margin = config["loss"]["cf_weight"], config["loss"]["cf_margin"]
    use_text = model.fusion != "none"
    totals: dict[str, float] = {}
    steps = 0

    for batch in tqdm(loader, desc="train", leave=False):
        image = batch["image"].to(device, non_blocking=True)
        target = {key: batch[key].to(device, non_blocking=True) for key in TARGET_KEYS}

        with torch.autocast(device.type, enabled=scaler.is_enabled()):
            features = model.encode(image)
            prediction = model.decode(features, _embed(batch["text"], embeddings, device) if use_text else None)
            losses = grasp_loss(prediction, target)
            loss = losses["grasp"]

            # Second pass with an instruction for another object of the same image.
            swappable = [i for i, text in enumerate(batch["negative_text"]) if text]
            if cf_weight > 0 and use_text and swappable:
                swapped_text = _embed([batch["negative_text"][i] for i in swappable], embeddings, device)
                swapped = model.decode(features[swappable], swapped_text)
                losses["counterfactual"] = counterfactual_loss(
                    prediction["quality"][swappable], swapped["quality"], target["quality"][swappable], cf_margin
                )
                loss = loss + cf_weight * losses["counterfactual"]
            losses["total"] = loss

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        steps += 1
        for key, value in losses.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach())

    return {key: value / max(steps, 1) for key, value in totals.items()}


def _decode_batch(prediction: dict[str, torch.Tensor]) -> list[np.ndarray]:
    """Best grasp of every sample of a batch of network outputs."""
    maps = {key: value.float().squeeze(1).cpu().numpy() for key, value in prediction.items()}
    quality = 1.0 / (1.0 + np.exp(-maps["quality"]))
    return [
        decode_grasp(quality[i], maps["cos"][i], maps["sin"][i], maps["width"][i]) for i in range(len(quality))
    ]


@torch.no_grad()
def evaluate(
    model: GraspNet,
    loader: DataLoader,
    embeddings: TextEmbeddings,
    device: torch.device,
    collect: int = 0,
) -> tuple[dict[str, float], list[dict]]:
    """Evaluate on a split.

    Returns the metrics of :class:`lgrasp.metrics.GraspMeter` and, for the first
    ``collect`` samples that have a swapped instruction, the predictions needed
    for qualitative figures.
    """
    model.eval()
    use_text = model.fusion != "none"
    size = loader.dataset.image_size
    meter = GraspMeter()
    records: list[dict] = []

    for batch in tqdm(loader, desc=f"eval {loader.dataset.split}", leave=False):
        features = model.encode(batch["image"].to(device, non_blocking=True))
        grasps = _decode_batch(model.decode(features, _embed(batch["text"], embeddings, device) if use_text else None))

        swappable = [i for i, text in enumerate(batch["negative_text"]) if text]
        swapped_grasps: dict[int, np.ndarray] = {}
        if swappable:
            if use_text:
                swapped_text = _embed([batch["negative_text"][i] for i in swappable], embeddings, device)
                decoded = _decode_batch(model.decode(features[swappable], swapped_text))
                swapped_grasps = dict(zip(swappable, decoded))
            else:  # without language the prediction cannot change
                swapped_grasps = {i: grasps[i] for i in swappable}

        for i, grasp in enumerate(grasps):
            ground_truth = batch["grasps"][i].numpy()
            correct = is_correct(grasp, ground_truth, size)
            stuck = is_correct(swapped_grasps[i], ground_truth, size) if i in swapped_grasps else None
            meter.update(correct, stuck)
            if i in swapped_grasps and len(records) < collect:
                records.append(
                    {
                        "id": batch["id"][i],
                        "text": batch["text"][i],
                        "swapped_text": batch["negative_text"][i],
                        "ground_truth": ground_truth,
                        "prediction": grasp,
                        "swapped_prediction": swapped_grasps[i],
                        "correct": correct,
                        "stuck": bool(stuck),
                    }
                )

    return meter.compute(), records
