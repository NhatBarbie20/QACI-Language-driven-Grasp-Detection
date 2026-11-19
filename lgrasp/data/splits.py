"""Official Grasp-Anything++ splits and the naming scheme of dataset files.

The split files in ``splits/grasp_anything_pp`` are the lists published with the
LGD code release (https://github.com/Fsoft-AIC/LGD, ``split/grasp-anything++``),
converted from pickle to JSON. Each entry is an *object id*:

    <scene>_<object index>        e.g. 3f73...570f_0

where ``<scene>`` is the 64-character hash that names the image. Labels are
stored per object *part*, as ``<scene>_<object index>_<part index>``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple

REPO_ROOT = Path(__file__).resolve().parents[2]
SPLIT_DIR = REPO_ROOT / "splits" / "grasp_anything_pp"
OFFICIAL_SPLITS = ("train_seen", "test_seen", "test_unseen")
SCENE_ID_LENGTH = 64


class PartId(NamedTuple):
    """Identifier of one annotated object part."""

    scene: str
    obj: int
    part: int

    @property
    def object_id(self) -> str:
        return f"{self.scene}_{self.obj}"

    def __str__(self) -> str:
        return f"{self.scene}_{self.obj}_{self.part}"


def parse_part_id(stem: str) -> PartId:
    """Parse ``<scene>_<obj>_<part>`` (a label file name without its extension)."""
    scene, obj, part = stem.rsplit("_", 2)
    if len(scene) != SCENE_ID_LENGTH:
        raise ValueError(f"Unexpected scene id in {stem!r}")
    return PartId(scene, int(obj), int(part))


def scene_of(object_id: str) -> str:
    """Scene hash of an object id ``<scene>_<obj>``."""
    return object_id.rsplit("_", 1)[0]


def load_split(name: str, split_dir: Path = SPLIT_DIR) -> list[str]:
    """Return the object ids of an official split."""
    if name not in OFFICIAL_SPLITS:
        raise ValueError(f"Unknown split {name!r}; expected one of {OFFICIAL_SPLITS}")
    with open(split_dir / f"{name}.json", encoding="utf-8") as file:
        return json.load(file)


def load_all_splits(split_dir: Path = SPLIT_DIR) -> dict[str, list[str]]:
    return {name: load_split(name, split_dir) for name in OFFICIAL_SPLITS}


def scenes_in_splits(splits: dict[str, list[str]]) -> set[str]:
    """Every scene referenced by at least one split."""
    return {scene_of(object_id) for ids in splits.values() for object_id in ids}
