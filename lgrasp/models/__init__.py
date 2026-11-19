"""Model factory and checkpoint helpers."""

from __future__ import annotations

from torch import nn

ARCHITECTURES = ("grasp_net", "factor")


def text_model_name(config: dict) -> str:
    """CLIP checkpoint whose text encoder an architecture uses.

    The factorised model compares text with CLIP image patches, so both must come
    from the same checkpoint.
    """
    return config["clip"] if config["arch"] == "factor" else config["text_model"]


def build_model(config: dict, text_dim: int) -> nn.Module:
    """Create the network described by the ``model`` section of a config."""
    if config["arch"] == "grasp_net":
        from .grasp_net import GraspNet

        return GraspNet(
            fusion=config["fusion"],
            text_dim=text_dim,
            channels=config["channels"],
            num_blocks=config["num_blocks"],
            text_hidden=config["text_hidden"],
            dropout=config["dropout"],
        )
    if config["arch"] == "factor":
        from .factor_net import FactorNet

        return FactorNet(
            clip_name=config["clip"],
            clip_resolution=config["clip_resolution"],
            grounding=config["grounding"],
            channels=config["channels"],
            num_blocks=config["num_blocks"],
            dropout=config["dropout"],
        )
    raise ValueError(f"Unknown arch {config['arch']!r}; expected one of {ARCHITECTURES}")


def trainable_state(model: nn.Module) -> dict:
    """State dict without frozen pretrained parts (they are reloaded from their source)."""
    frozen = getattr(model, "frozen_prefixes", ())
    return {key: value for key, value in model.state_dict().items() if not (frozen and key.startswith(frozen))}


def load_trainable_state(model: nn.Module, state: dict) -> None:
    """Load a state saved by :func:`trainable_state`; everything else must already match."""
    missing, unexpected = model.load_state_dict(state, strict=False)
    frozen = getattr(model, "frozen_prefixes", ())
    missing = [key for key in missing if not (frozen and key.startswith(frozen))]
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch. Missing: {missing[:5]} Unexpected: {list(unexpected)[:5]}")
