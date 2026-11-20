"""Training objectives.

``grasp_loss``
    The supervised loss of generative grasp networks: one term per output map.

``counterfactual_loss``
    The proposed language-grounding term. Take an image with an instruction
    ``t+`` for object A and an instruction ``t-`` for a *different* object of the
    same image. The image is identical in both forward passes, so any difference
    between the two quality maps is caused by language alone. Inside the grasp
    region of A, the quality under ``t+`` must exceed the quality under ``t-``
    by a margin ``m``::

        L_cf = max(0, m - mean_R Q(I, t+) + mean_R Q(I, t-))

    A network that ignores the instruction produces the same map twice and pays
    the full margin, so the term directly penalises language-blind solutions.

``grounding_loss``
    Direct supervision of the grounding factor of the factorised model. The other
    annotated objects of the image are free negatives: the grounding map must be
    high on the grasp region of the named object and low on the grasp regions of
    every other object. Pixels without any annotation are ignored, because the
    grasp regions do not cover whole objects.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def grasp_loss(
    prediction: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    geometry_target: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """Per-map losses and their sum (key ``"grasp"``).

    Quality is a per-pixel binary classification (logits + BCE); the angle and
    width maps are regressed with a smooth L1 loss, as in GR-ConvNet.

    ``geometry_target`` holds the maps rendered from the grasps of *every* object
    of the image. When given (factorised model only), the language-independent
    outputs (geometry factor, angle, width) are supervised with it, while the
    final quality is still supervised with the grasps of the named object.
    """
    shared = geometry_target if geometry_target is not None else target
    losses = {
        "quality": F.binary_cross_entropy_with_logits(prediction["quality"].float(), target["quality"]),
        "cos": F.smooth_l1_loss(prediction["cos"], shared["cos"]),
        "sin": F.smooth_l1_loss(prediction["sin"], shared["sin"]),
        "width": F.smooth_l1_loss(prediction["width"], shared["width"]),
    }
    losses["grasp"] = losses["quality"] + losses["cos"] + losses["sin"] + losses["width"]
    if geometry_target is not None:
        losses["geometry"] = F.binary_cross_entropy_with_logits(
            prediction["geometry"].float(), geometry_target["quality"]
        )
        losses["grasp"] = losses["grasp"] + losses["geometry"]
    return losses


def grounding_loss(grounding_logits: torch.Tensor, region: torch.Tensor, scene_region: torch.Tensor) -> torch.Tensor:
    """Balanced binary cross-entropy of the grounding map on annotated pixels.

    Args:
        grounding_logits: ``[B, 1, H, W]`` logits of the grounding factor.
        region: ``[B, 1, H, W]``, positive inside the grasp region of the named object.
        scene_region: ``[B, 1, H, W]``, positive inside the grasp region of any object.

    Positives and negatives are averaged separately, so the loss does not depend
    on how many other objects an image has. Returns zero if the batch has no
    negative pixel (every image has a single annotated object).
    """
    logits = grounding_logits.float()
    positive = (region > 0).float()
    negative = (scene_region > 0).float() * (1 - positive)
    if negative.sum() == 0 or positive.sum() == 0:
        return logits.new_zeros(())
    loss = F.binary_cross_entropy_with_logits(logits, positive, reduction="none")
    return 0.5 * ((loss * positive).sum() / positive.sum() + (loss * negative).sum() / negative.sum())


def counterfactual_loss(
    quality_logits_pos: torch.Tensor,
    quality_logits_neg: torch.Tensor,
    region: torch.Tensor,
    margin: float = 0.5,
) -> torch.Tensor:
    """Margin loss between the quality maps of a matching and a swapped instruction.

    Args:
        quality_logits_pos: ``[B, 1, H, W]`` logits for the matching instruction.
        quality_logits_neg: ``[B, 1, H, W]`` logits for an instruction that refers
            to another object of the same image.
        region: ``[B, 1, H, W]`` binary mask of the ground-truth grasp region of
            the matching instruction.
        margin: required gap between the two mean qualities inside the region.

    Returns a scalar; zero if no sample has a non-empty region.
    """
    area = region.sum(dim=(1, 2, 3))
    valid = area > 0
    if not valid.any():
        return quality_logits_pos.new_zeros(())
    area = area.clamp(min=1)
    mean_pos = (torch.sigmoid(quality_logits_pos.float()) * region).sum(dim=(1, 2, 3)) / area
    mean_neg = (torch.sigmoid(quality_logits_neg.float()) * region).sum(dim=(1, 2, 3)) / area
    return F.relu(margin - mean_pos + mean_neg)[valid].mean()
