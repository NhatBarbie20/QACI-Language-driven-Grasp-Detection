"""PyTorch dataset for language-driven grasp detection on Grasp-Anything++."""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .grasp import LABEL_IMAGE_SIZE, flip_grasps, label_to_grasps, render_grasp_maps
from .io import load_grasp_label, load_image
from .manifest import load_manifest

SPLITS = ("train", "val", "test_seen", "test_unseen")
IMAGE_MEAN = (0.485, 0.456, 0.406)  # ImageNet statistics
IMAGE_STD = (0.229, 0.224, 0.225)
TARGET_KEYS = ("quality", "cos", "sin", "width")


class GraspAnythingPP(Dataset):
    """One item = one annotated object part of one image.

    Each item is a dict with

    ``image``        float tensor ``[3, S, S]``, ImageNet-normalised;
    ``quality``, ``cos``, ``sin``, ``width``   target maps ``[1, S, S]``;
    ``text``         the grasp instruction, e.g. "Hold key at its ring.";
    ``negative_text``  an instruction for a *different object* of the same image,
                     or ``""`` when the image has a single annotated object;
    ``grasps``       ground-truth rectangles ``[N, 5]`` at resolution ``S``
                     (see :mod:`lgrasp.data.grasp`), used for evaluation;
    ``id``           ``<scene>_<obj>_<part>``.

    The negative prompt is drawn at random for the training split and is fixed
    for evaluation splits, so evaluation is deterministic.

    Args:
        root: dataset folder that contains ``manifest.json``.
        split: one of :data:`SPLITS`.
        image_size: side of the network input in pixels.
        augment: apply random horizontal / vertical flips (training only). The
            instructions never mention left or right, so flips keep them valid.
        all_objects: additionally return ``geometry_quality``, ``geometry_cos``,
            ``geometry_sin`` and ``geometry_width``: the target maps rendered from
            the grasps of *every* annotated object of the image (training only).
            They supervise outputs that do not depend on the instruction.
        max_samples: keep only the first N samples (for smoke tests).
    """

    def __init__(self, root: str | Path, split: str, image_size: int = 224, augment: bool = False,
                 all_objects: bool = False, max_samples: int | None = None):
        if split not in SPLITS:
            raise ValueError(f"Unknown split {split!r}; expected one of {SPLITS}")
        self.root = Path(root)
        self.split = split
        self.image_size = image_size
        self.scale = image_size / LABEL_IMAGE_SIZE
        self.training = split == "train"
        self.augment = augment and self.training
        self.all_objects = all_objects and self.training

        manifest = load_manifest(self.root)
        self.scenes: dict[str, dict] = manifest["scenes"]
        self.samples: list[dict] = [s for s in manifest["samples"] if s["split"] == split]
        if not self.samples:
            raise RuntimeError(f"No samples for split {split!r} in {self.root}")
        if max_samples is not None:
            self.samples = self.samples[:max_samples]

        # Instructions of every object in a scene, including objects outside the
        # split: they are only ever used as alternative prompts.
        self._scene_instructions: dict[str, dict[int, list[str]]] = defaultdict(lambda: defaultdict(list))
        # One label file per distinct set of grasps in a scene (parts often share labels).
        scene_labels: dict[str, dict[tuple[int, int], str]] = defaultdict(dict)
        for sample in manifest["samples"]:
            self._scene_instructions[sample["scene"]][sample["obj"]].append(sample["instruction"])
            scene_labels[sample["scene"]].setdefault((sample["obj"], sample["label_group"]), sample["id"])
        self._scene_label_ids = {scene: sorted(ids.values()) for scene, ids in scene_labels.items()}

        self._mean = torch.tensor(IMAGE_MEAN).view(3, 1, 1)
        self._std = torch.tensor(IMAGE_STD).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.samples)

    def has_negative(self, index: int) -> bool:
        """Whether the image of this sample contains another annotated object."""
        sample = self.samples[index]
        return len(self._scene_instructions[sample["scene"]]) > 1

    def _negative_text(self, sample: dict) -> str:
        others = [
            text
            for obj, texts in sorted(self._scene_instructions[sample["scene"]].items())
            if obj != sample["obj"]
            for text in texts
        ]
        if not others:
            return ""
        return random.choice(others) if self.training else others[0]

    def load_image(self, index: int) -> Image.Image:
        """The resized RGB image of a sample, before normalisation."""
        image = load_image(self.root / "image" / f"{self.samples[index]['scene']}.jpg")
        if image.size != (self.image_size, self.image_size):
            image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        return image

    def _grasps_of(self, sample_id: str) -> np.ndarray:
        return label_to_grasps(load_grasp_label(self.root / "grasp_label_positive" / f"{sample_id}.pt"), self.scale)

    def load_grasps(self, index: int) -> np.ndarray:
        """Ground-truth grasps ``[N, 5]`` of a sample at the dataset resolution."""
        return self._grasps_of(self.samples[index]["id"])

    def load_scene_grasps(self, index: int) -> np.ndarray:
        """Grasps ``[N, 5]`` of every annotated object in the image of a sample."""
        ids = self._scene_label_ids[self.samples[index]["scene"]]
        return np.concatenate([self._grasps_of(sample_id) for sample_id in ids])

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        image = self.load_image(index)
        grasps = self.load_grasps(index)
        scene_grasps = self.load_scene_grasps(index) if self.all_objects else None
        if self.augment:
            horizontal, vertical = random.random() < 0.5, random.random() < 0.5
            if horizontal:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
            if vertical:
                image = image.transpose(Image.FLIP_TOP_BOTTOM)
            grasps = flip_grasps(grasps, self.image_size, horizontal, vertical)
            if scene_grasps is not None:
                scene_grasps = flip_grasps(scene_grasps, self.image_size, horizontal, vertical)
        pixels = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0).permute(2, 0, 1)
        maps = render_grasp_maps(grasps, self.image_size)

        item = {key: torch.from_numpy(maps[key]).unsqueeze(0) for key in TARGET_KEYS}
        if scene_grasps is not None:
            scene_maps = render_grasp_maps(scene_grasps, self.image_size)
            item.update({f"geometry_{key}": torch.from_numpy(scene_maps[key]).unsqueeze(0) for key in TARGET_KEYS})
        item.update(
            image=(pixels - self._mean) / self._std,
            text=sample["instruction"],
            negative_text=self._negative_text(sample),
            grasps=torch.from_numpy(grasps),
            id=sample["id"],
        )
        return item


def collate(batch: list[dict]) -> dict:
    """Stack fixed-size tensors and keep variable-length fields as lists."""
    keys = ["image", *TARGET_KEYS]
    if f"geometry_{TARGET_KEYS[0]}" in batch[0]:
        keys += [f"geometry_{key}" for key in TARGET_KEYS]
    stacked = {key: torch.stack([item[key] for item in batch]) for key in keys}
    for key in ("text", "negative_text", "grasps", "id"):
        stacked[key] = [item[key] for item in batch]
    return stacked
