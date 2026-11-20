"""Frozen CLIP as a dense, open-vocabulary feature extractor.

CLIP is trained to align one global image vector with text, so its patch tokens
are not directly comparable with text embeddings. MaskCLIP (Zhou et al., ECCV
2022) showed that the *value* path of the last attention layer yields patch
features that live in the joint image-text space without any training. This
module implements that read-out and the resulting patch-text similarity map.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from lgrasp.data.dataset import IMAGE_MEAN, IMAGE_STD

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
DEFAULT_CLIP = "openai/clip-vit-base-patch16"
READOUTS = ("value", "qq")


class ClipDense(nn.Module):
    """Frozen CLIP image tower that returns one text-aligned feature per patch.

    Args:
        model_name: Hugging Face id of a CLIP ViT checkpoint.
        resolution: side of the image fed to CLIP. Inputs are resized to it; a
            larger value gives a finer patch grid (``resolution / patch_size``).
        readout: how the last attention layer is read. ``"value"`` keeps every
            patch to itself (MaskCLIP). ``"qq"`` lets a patch aggregate the patches
            with a similar query / key vector (the self-self attention of SCLIP),
            which smooths the map within an object.
    """

    def __init__(self, model_name: str = DEFAULT_CLIP, resolution: int = 224, readout: str = "value"):
        super().__init__()
        if readout not in READOUTS:
            raise ValueError(f"Unknown readout {readout!r}; expected one of {READOUTS}")
        self.readout = readout
        from transformers import CLIPModel  # heavy import, only needed here

        clip = CLIPModel.from_pretrained(model_name)
        self.vision = clip.vision_model
        self.projection = clip.visual_projection
        self.resolution = resolution
        self.native_resolution = clip.config.vision_config.image_size
        self.dim = clip.config.projection_dim
        self.requires_grad_(False)
        self.eval()

        # The dataset normalises with ImageNet statistics; CLIP expects its own.
        source_mean, source_std = torch.tensor(IMAGE_MEAN), torch.tensor(IMAGE_STD)
        target_mean, target_std = torch.tensor(CLIP_MEAN), torch.tensor(CLIP_STD)
        self.register_buffer("scale", (source_std / target_std).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("shift", ((source_mean - target_mean) / target_std).view(1, 3, 1, 1), persistent=False)

    def train(self, mode: bool = True) -> "ClipDense":
        return super().train(False)  # always in evaluation mode

    @torch.no_grad()
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """L2-normalised patch features ``[B, D, h, w]`` for ImageNet-normalised images."""
        image = image * self.scale + self.shift
        if image.shape[-1] != self.resolution:
            image = F.interpolate(image, size=(self.resolution, self.resolution), mode="bilinear", align_corners=False)

        hidden = self.vision(
            pixel_values=image,
            output_hidden_states=True,
            interpolate_pos_encoding=self.resolution != self.native_resolution,
        ).hidden_states[-2]  # input of the last transformer layer

        last = self.vision.encoder.layers[-1]
        attention = last.self_attn
        normed = last.layer_norm1(hidden)
        values = attention.v_proj(normed)
        if self.readout == "qq":
            values = self._self_self_attention(attention, normed, values)
        values = attention.out_proj(values)
        tokens = self.projection(self.vision.post_layernorm(values))[:, 1:]  # drop the class token

        side = int(tokens.shape[1] ** 0.5)
        features = F.normalize(tokens.float(), dim=-1)
        return features.transpose(1, 2).reshape(len(image), self.dim, side, side)

    @staticmethod
    def _self_self_attention(attention: nn.Module, normed: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """Mix ``values`` with query-query and key-key attention weights (per head)."""
        batch, tokens, _ = normed.shape
        heads = attention.num_heads

        def split(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.view(batch, tokens, heads, -1).transpose(1, 2)  # [B, heads, N, d]

        weights = 0
        for projection in (attention.q_proj, attention.k_proj):
            vectors = split(projection(normed))
            weights = weights + (vectors @ vectors.transpose(-1, -2) * attention.scale).softmax(dim=-1)
        mixed = (0.5 * weights) @ split(values)
        return mixed.transpose(1, 2).reshape(batch, tokens, -1)


def similarity_map(patch_features: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
    """Cosine similarity ``[B, 1, h, w]`` between patch features and one text vector per image.

    The map is standardised per image (zero mean, unit variance over patches):
    raw CLIP similarities have a large image-dependent offset, and only the
    relative pattern over the image carries the grounding signal.
    """
    similarity = torch.einsum("bdhw,bd->bhw", patch_features, F.normalize(text.float(), dim=-1))
    mean = similarity.mean(dim=(1, 2), keepdim=True)
    std = similarity.std(dim=(1, 2), keepdim=True).clamp(min=1e-6)
    return ((similarity - mean) / std).unsqueeze(1)
