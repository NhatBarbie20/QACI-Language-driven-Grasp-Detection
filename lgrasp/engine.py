"""Training and evaluation loops."""

from __future__ import annotations

import contextlib
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from lgrasp.data.dataset import TARGET_KEYS, GraspAnythingPP, collate
from lgrasp.losses import counterfactual_loss, grasp_loss, grounding_loss
from lgrasp.metrics import GraspMeter, decode_grasp, is_correct
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
        all_objects=data["all_objects"],
        max_negative_overlap=data["max_negative_overlap"],
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


@contextlib.contextmanager
def frozen_running_stats(model: nn.Module):
    """Run a training-mode forward pass without updating BatchNorm running statistics.

    The swapped-instruction pass produces activations with a very different
    distribution from the matching pass. If both update the running averages,
    the stored statistics describe neither and the network breaks in evaluation
    mode. Inside this context BatchNorm still normalises with batch statistics
    (so gradients are unchanged) but leaves its running averages untouched; they
    then describe matching instructions only, which is the test-time input.
    """
    norms = [module for module in model.modules() if isinstance(module, nn.modules.batchnorm._BatchNorm)]
    momenta = [norm.momentum for norm in norms]
    for norm in norms:
        norm.momentum = 0.0
    try:
        yield
    finally:
        for norm, momentum in zip(norms, momenta):
            norm.momentum = momentum


def train_one_epoch(
    model: nn.Module,
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
    ground_weight = config["loss"]["ground_weight"]
    use_text = model.fusion != "none"
    totals: dict[str, float] = {}
    steps = 0

    for batch in tqdm(loader, desc="train", leave=False):
        image = batch["image"].to(device, non_blocking=True)
        target = {key: batch[key].to(device, non_blocking=True) for key in TARGET_KEYS}
        geometry_target = None
        if "geometry_quality" in batch:
            geometry_target = {key: batch[f"geometry_{key}"].to(device, non_blocking=True) for key in TARGET_KEYS}

        with torch.autocast(device.type, enabled=scaler.is_enabled()):
            features = model.encode(image)
            prediction = model.decode(features, _embed(batch["text"], embeddings, device) if use_text else None)
            losses = grasp_loss(prediction, target, geometry_target)
            loss = losses["grasp"]

            # Other objects of the image as negatives for the grounding factor.
            if ground_weight > 0 and geometry_target is not None and "grounding" in prediction:
                losses["grounding"] = grounding_loss(
                    prediction["grounding"], target["quality"], geometry_target["quality"]
                )
                loss = loss + ground_weight * losses["grounding"]

            # Second pass with an instruction for another object of the same image.
            swappable = [i for i, text in enumerate(batch["negative_text"]) if text]
            if cf_weight > 0 and use_text and swappable:
                swapped_text = _embed([batch["negative_text"][i] for i in swappable], embeddings, device)
                with frozen_running_stats(model):
                    swapped = model.decode(model.select(features, swappable), swapped_text)
                # The factorised model exposes its language-dependent factor; contrast that one.
                key = "grounding" if "grounding" in prediction else "quality"
                losses["counterfactual"] = counterfactual_loss(
                    prediction[key][swappable], swapped[key], target["quality"][swappable], cf_margin
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
    maps = {key: prediction[key].float().squeeze(1).cpu().numpy() for key in TARGET_KEYS}
    quality = 1.0 / (1.0 + np.exp(-maps["quality"]))
    return [
        decode_grasp(quality[i], maps["cos"][i], maps["sin"][i], maps["width"][i]) for i in range(len(quality))
    ]


GROUNDING_OVERRIDES = ("model", "none", "oracle")


def _override_grounding(prediction: dict[str, torch.Tensor], mode: str, region: torch.Tensor) -> dict[str, torch.Tensor]:
    """Replace the grounding factor of a factorised prediction (diagnostics only).

    ``"none"`` drops it, so the quality is the geometry factor alone: what the
    model does without language. ``"oracle"`` replaces it with the ground-truth
    grasp region of the named object: the best any grounding map could do.
    """
    geometry = prediction["geometry"].float()
    quality = geometry if mode == "none" else torch.where(region > 0, geometry, torch.full_like(geometry, -20.0))
    return {**prediction, "quality": quality}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    embeddings: TextEmbeddings,
    device: torch.device,
    collect: int = 0,
    grounding: str = "model",
) -> tuple[dict[str, float], list[dict]]:
    """Evaluate on a split.

    Returns the metrics of :class:`lgrasp.metrics.GraspMeter` and, for the first
    ``collect`` samples that have a swapped instruction, the predictions needed
    for qualitative figures.

    ``grounding`` other than ``"model"`` is a diagnostic for the factorised
    architecture, see :func:`_override_grounding`. The swapped-instruction
    metrics are not defined for ``"oracle"`` and are reported as zero.
    """
    if grounding not in GROUNDING_OVERRIDES:
        raise ValueError(f"Unknown grounding {grounding!r}; expected one of {GROUNDING_OVERRIDES}")
    if grounding != "model" and model.fusion != "factor":
        raise ValueError("The grounding override needs the factorised architecture")
    model.eval()
    use_text = model.fusion != "none"
    size = loader.dataset.image_size
    meter = GraspMeter()
    records: list[dict] = []

    for batch in tqdm(loader, desc=f"eval {loader.dataset.split}", leave=False):
        features = model.encode(batch["image"].to(device, non_blocking=True))
        prediction = model.decode(features, _embed(batch["text"], embeddings, device) if use_text else None)
        if grounding != "model":
            prediction = _override_grounding(prediction, grounding, batch["quality"].to(device))
        grasps = _decode_batch(prediction)

        swappable = [i for i, text in enumerate(batch["negative_text"]) if text] if grounding != "oracle" else []
        swapped_grasps: dict[int, np.ndarray] = {}
        if swappable:
            if use_text and grounding == "model":
                swapped_text = _embed([batch["negative_text"][i] for i in swappable], embeddings, device)
                decoded = _decode_batch(model.decode(model.select(features, swappable), swapped_text))
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
