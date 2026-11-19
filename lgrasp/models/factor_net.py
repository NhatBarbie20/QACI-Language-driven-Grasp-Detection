"""Factorised language-driven grasp network: *how* to grasp x *where* to grasp.

The grasp quality of a pixel is the product of two maps::

    Q(p | I, t) = G(p | I) * S(p | I, t)

``G`` (geometry)   "a gripper can close here", together with the grasp angle and
                   width. It does not see the instruction, so it can be trained
                   on the grasps of every object of an image.
``S`` (grounding)  "this pixel belongs to the object the instruction names". It
                   starts from the patch-text similarity of a frozen CLIP model
                   (:mod:`lgrasp.models.clip_dense`).

The grounding branch never receives the text embedding itself. It only sees
*where* the instruction matches the image, plus image features. It therefore
cannot memorise the training vocabulary, and its ability to handle unseen
object names is inherited from CLIP.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .clip_dense import DEFAULT_CLIP, ClipDense, similarity_map
from .grasp_net import GraspNet, _conv_block

GROUNDING_MODES = ("raw", "refine")


class FactorNet(nn.Module):
    """Geometry network x CLIP-based grounding map.

    Args:
        clip_name: Hugging Face id of the frozen CLIP model (image and text side).
        clip_resolution: side of the image fed to CLIP.
        grounding: ``"raw"`` uses an affine function of the CLIP similarity;
            ``"refine"`` adds a small convolutional correction on top of it.
        channels, num_blocks, dropout: size of the geometry network.
    """

    fusion = "factor"
    frozen_prefixes = ("clip.",)  # not stored in checkpoints

    def __init__(
        self,
        clip_name: str = DEFAULT_CLIP,
        clip_resolution: int = 224,
        grounding: str = "refine",
        channels: int = 32,
        num_blocks: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        if grounding not in GROUNDING_MODES:
            raise ValueError(f"Unknown grounding {grounding!r}; expected one of {GROUNDING_MODES}")
        self.grounding = grounding
        self.geometry = GraspNet(fusion="none", channels=channels, num_blocks=num_blocks, dropout=dropout)
        self.clip = ClipDense(clip_name, clip_resolution)

        # S = scale * similarity + bias (+ learned correction). The similarity is
        # standardised per image, so these two scalars set a sensible threshold.
        self.sim_scale = nn.Parameter(torch.tensor(2.0))
        self.sim_bias = nn.Parameter(torch.tensor(-1.0))
        if grounding == "refine":
            self.reduce = _conv_block(4 * channels, 32, kernel=1, stride=1)
            self.refine = nn.Sequential(
                _conv_block(33, 64, kernel=3, stride=1),
                _conv_block(64, 64, kernel=3, stride=1),
                nn.Conv2d(64, 1, 3, padding=1),
            )
            nn.init.zeros_(self.refine[-1].weight)  # start exactly from the CLIP prior
            nn.init.zeros_(self.refine[-1].bias)

    def encode(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """Language-independent features: the geometry stem and the CLIP patch features."""
        return {"stem": self.geometry.encode(image), "patches": self.clip(image)}

    @staticmethod
    def select(features: dict[str, torch.Tensor], indices: list[int]) -> dict[str, torch.Tensor]:
        return {key: value[indices] for key, value in features.items()}

    def decode(self, features: dict[str, torch.Tensor], text: torch.Tensor) -> dict[str, torch.Tensor]:
        """Grasp maps ``[B, 1, H, W]``.

        ``quality`` is the logit of the product ``G * S``; ``geometry`` and
        ``grounding`` are the logits of the two factors.
        """
        stem = features["stem"]
        maps = self.geometry.decode(stem)
        geometry = maps["quality"]

        similarity = similarity_map(features["patches"], text)
        similarity = F.interpolate(similarity, size=stem.shape[-2:], mode="bilinear", align_corners=False)
        grounding = self.sim_scale * similarity + self.sim_bias
        if self.grounding == "refine":
            grounding = grounding + self.refine(torch.cat([self.reduce(stem), similarity.to(stem.dtype)], dim=1)).float()
        grounding = F.interpolate(grounding, size=geometry.shape[-2:], mode="bilinear", align_corners=False)

        product = (torch.sigmoid(geometry.float()) * torch.sigmoid(grounding.float())).clamp(1e-6, 1 - 1e-6)
        return {
            "quality": torch.logit(product),
            "cos": maps["cos"],
            "sin": maps["sin"],
            "width": maps["width"],
            "geometry": geometry,
            "grounding": grounding,
        }

    def forward(self, image: torch.Tensor, text: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.decode(self.encode(image), text)
