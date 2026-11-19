"""Language-conditioned generative grasp network.

The vision path follows GR-ConvNet (Kumra et al., IROS 2020): a convolutional
stem that downsamples by 4, a stack of residual blocks, a transposed-convolution
decoder and four per-pixel heads (grasp quality, cos 2θ, sin 2θ, gripper width).

Language enters through one of three fusion modes:

``none``    the instruction is ignored (vision-only reference);
``concat``  the text vector is tiled over the feature map and concatenated once,
            before the residual blocks;
``film``    every residual block is modulated by the text with FiLM
            (Perez et al., AAAI 2018): ``y = (1 + γ(t)) * x + β(t)``.

The network is split into :meth:`GraspNet.encode` (image only) and
:meth:`GraspNet.decode` (image features + text), so that several instructions
can be evaluated on one image without recomputing the stem.
"""

from __future__ import annotations

import torch
from torch import nn

FUSION_MODES = ("none", "concat", "film")
OUTPUT_KEYS = ("quality", "cos", "sin", "width")


def _conv_block(in_channels: int, out_channels: int, kernel: int, stride: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, stride, padding=(kernel - 1) // 2, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


def _deconv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class FiLM(nn.Module):
    """Feature-wise linear modulation of a feature map by a text vector.

    The projection is zero-initialised, so the block starts as the identity and
    the network initially behaves like its vision-only counterpart.
    """

    def __init__(self, text_dim: int, channels: int):
        super().__init__()
        self.to_scale_shift = nn.Linear(text_dim, 2 * channels)
        nn.init.zeros_(self.to_scale_shift.weight)
        nn.init.zeros_(self.to_scale_shift.bias)

    def forward(self, features: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        scale, shift = self.to_scale_shift(text).chunk(2, dim=1)
        return features * (1 + scale[:, :, None, None]) + shift[:, :, None, None]


class ResidualBlock(nn.Module):
    """Two 3x3 convolutions with a skip connection and optional FiLM conditioning."""

    def __init__(self, channels: int, text_dim: int | None = None):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.film = FiLM(text_dim, channels) if text_dim else None
        self.act = nn.ReLU(inplace=True)

    def forward(self, features: torch.Tensor, text: torch.Tensor | None = None) -> torch.Tensor:
        out = self.act(self.bn1(self.conv1(features)))
        out = self.bn2(self.conv2(out))
        if self.film is not None:
            out = self.film(out, text)
        return self.act(out + features)


class GraspNet(nn.Module):
    """Predict dense grasp maps from an RGB image and an instruction embedding.

    Args:
        fusion: one of :data:`FUSION_MODES`.
        text_dim: size of the instruction embedding (e.g. 512 for CLIP ViT-B/32).
        channels: width of the first convolution; the bottleneck has ``4 * channels``.
        num_blocks: number of residual blocks.
        text_hidden: size of the learned projection of the text embedding.
        dropout: dropout probability before the output heads.
    """

    def __init__(
        self,
        fusion: str = "film",
        text_dim: int = 512,
        channels: int = 32,
        num_blocks: int = 5,
        text_hidden: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        if fusion not in FUSION_MODES:
            raise ValueError(f"Unknown fusion {fusion!r}; expected one of {FUSION_MODES}")
        self.fusion = fusion
        width = 4 * channels

        self.stem = nn.Sequential(
            _conv_block(3, channels, kernel=9, stride=1),
            _conv_block(channels, 2 * channels, kernel=4, stride=2),
            _conv_block(2 * channels, width, kernel=4, stride=2),
        )

        if fusion != "none":
            self.text_proj = nn.Sequential(
                nn.Linear(text_dim, text_hidden), nn.LayerNorm(text_hidden), nn.ReLU(inplace=True)
            )
        if fusion == "concat":
            self.merge = _conv_block(width + text_hidden, width, kernel=1, stride=1)

        block_text_dim = text_hidden if fusion == "film" else None
        self.blocks = nn.ModuleList(ResidualBlock(width, block_text_dim) for _ in range(num_blocks))

        self.decoder = nn.Sequential(
            _deconv_block(width, 2 * channels),
            _deconv_block(2 * channels, channels),
            _conv_block(channels, channels, kernel=9, stride=1),
            nn.Dropout2d(dropout),
        )
        self.heads = nn.ModuleDict({key: nn.Conv2d(channels, 1, 3, padding=1) for key in OUTPUT_KEYS})

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        """Language-independent image features ``[B, 4*channels, H/4, W/4]``."""
        return self.stem(image)

    @staticmethod
    def select(features: torch.Tensor, indices: list[int]) -> torch.Tensor:
        """Features of a subset of the batch."""
        return features[indices]

    def decode(self, features: torch.Tensor, text: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """Grasp maps ``[B, 1, H, W]`` for image features and instruction embeddings.

        ``quality`` holds logits; apply a sigmoid to obtain the grasp quality.
        """
        if self.fusion != "none":
            if text is None:
                raise ValueError(f"fusion={self.fusion!r} needs a text embedding")
            text = self.text_proj(text.to(features.dtype))
        if self.fusion == "concat":
            tiled = text[:, :, None, None].expand(-1, -1, *features.shape[-2:])
            features = self.merge(torch.cat([features, tiled], dim=1))
        for block in self.blocks:
            features = block(features, text if self.fusion == "film" else None)
        decoded = self.decoder(features)
        return {key: head(decoded) for key, head in self.heads.items()}

    def forward(self, image: torch.Tensor, text: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        return self.decode(self.encode(image), text)

