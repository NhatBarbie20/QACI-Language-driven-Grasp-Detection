"""Experiment configuration.

Every experiment is ``configs/default.yaml`` plus a small file that overrides a
few values, plus optional ``section.key=value`` overrides from the command line.
The merged configuration is stored inside each checkpoint.
"""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


def _merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into a copy of ``base``; unknown keys are errors."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key not in merged:
            raise KeyError(f"Unknown config key {key!r}; valid keys: {sorted(merged)}")
        merged[key] = _merge(merged[key], value) if isinstance(merged[key], dict) else value
    return merged


def with_defaults(config: dict) -> dict:
    """Complete a stored config with the current defaults (for older checkpoints)."""
    with open(DEFAULT_CONFIG, encoding="utf-8") as file:
        return _merge(yaml.safe_load(file), config)


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> dict:
    """Load the default config, apply the experiment file, then ``a.b=value`` overrides."""
    with open(DEFAULT_CONFIG, encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if path is not None:
        with open(path, encoding="utf-8") as file:
            config = _merge(config, yaml.safe_load(file) or {})
    for item in overrides or []:
        dotted, _, raw = item.partition("=")
        nested: dict = {}
        cursor = nested
        *parents, leaf = dotted.split(".")
        for parent in parents:
            cursor = cursor.setdefault(parent, {})
        cursor[leaf] = yaml.safe_load(raw)
        config = _merge(config, nested)
    return config
