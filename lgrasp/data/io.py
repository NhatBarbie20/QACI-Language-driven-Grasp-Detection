"""Safe readers for the raw Grasp-Anything++ files.

The dataset stores text in pickle files and labels in ``torch.save`` files. Both
formats can execute code when loaded naively, so they are read here with
loaders that only accept plain data.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch
from PIL import Image

LABEL_COLUMNS = ("quality", "x", "y", "w", "h", "theta_deg")


class _PlainDataUnpickler(pickle.Unpickler):
    """Unpickler that refuses every class or function reference."""

    def find_class(self, module: str, name: str):
        raise pickle.UnpicklingError(f"refusing to load {module}.{name} from a dataset file")


def load_pickle(path: str | Path):
    """Load a pickle that contains only built-in containers, strings and numbers."""
    with open(path, "rb") as file:
        return _PlainDataUnpickler(file).load()


def load_scene_description(path: str | Path) -> tuple[str, list[str]]:
    """Return ``(scene sentence, object names)`` of a scene description file."""
    description, objects = load_pickle(path)
    return str(description), [str(name) for name in objects]


def load_instruction(path: str | Path) -> str:
    """Return the grasp instruction of one object part, e.g. ``"Hold key at its ring."``."""
    return str(load_pickle(path)).strip()


def load_grasp_label(path: str | Path) -> np.ndarray:
    """Return a float32 array ``[N, 6]`` with columns :data:`LABEL_COLUMNS`.

    Coordinates are in pixels of the original 416x416 image and the angle is in
    degrees.
    """
    tensor = torch.load(path, map_location="cpu", weights_only=True)
    return np.asarray(tensor, dtype=np.float32).reshape(-1, len(LABEL_COLUMNS))


def load_image(path: str | Path) -> Image.Image:
    """Load an image as RGB."""
    with Image.open(path) as image:
        return image.convert("RGB")
