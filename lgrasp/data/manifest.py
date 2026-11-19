"""Build the sample index (``manifest.json``) from the raw dataset files.

Reading tens of thousands of tiny pickle files at every start-up is slow,
especially on Colab with Google Drive. The manifest is built once and holds all
text plus the split of every sample, so training only opens images and labels.

One *sample* is one annotated object part: an image, an instruction such as
"Hold key at its ring." and the positive grasp rectangles of that part.

Splits
------
``train`` / ``val``        from the official ``train_seen`` list (``val`` is a fixed
                           5% of its scenes, chosen by scene hash);
``test_seen`` / ``test_unseen``  the official test lists, untouched;
``None``                   objects of a downloaded scene that no split lists. They
                           are never used as samples, only as alternative prompts.

Scenes that occur in both ``train_seen`` and a test list are removed from
training so that no test image is seen during training.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

from tqdm import tqdm

from .grasp import valid_label_rows
from .io import load_grasp_label, load_instruction, load_scene_description
from .splits import parse_part_id, scene_of

MANIFEST_NAME = "manifest.json"
VAL_FRACTION_DENOMINATOR = 20  # 1 scene in 20 goes to validation


@dataclass(frozen=True)
class Sample:
    id: str  # <scene>_<obj>_<part>
    scene: str
    obj: int
    part: int
    split: str | None
    instruction: str
    object_name: str
    num_grasps: int  # valid positive rectangles
    label_group: int  # parts of one object with the same number share identical labels


def _is_val_scene(scene: str) -> bool:
    return int(scene[:8], 16) % VAL_FRACTION_DENOMINATOR == 0


def assign_splits(splits: dict[str, list[str]]) -> tuple[dict[str, str], set[str]]:
    """Map every listed object id to ``train`` / ``val`` / ``test_seen`` / ``test_unseen``.

    Returns the mapping and the set of training scenes dropped because they also
    appear in a test split.
    """
    test_scenes = {scene_of(i) for name in ("test_seen", "test_unseen") for i in splits[name]}
    assignment: dict[str, str] = {}
    leaked: set[str] = set()
    for object_id in splits["train_seen"]:
        scene = scene_of(object_id)
        if scene in test_scenes:
            leaked.add(scene)
            continue
        assignment[object_id] = "val" if _is_val_scene(scene) else "train"
    for name in ("test_seen", "test_unseen"):
        for object_id in splits[name]:
            assignment[object_id] = name
    return assignment, leaked


def build_manifest(root: Path, splits: dict[str, list[str]], workers: int = 16) -> tuple[dict, dict]:
    """Scan ``root`` and return ``(manifest, statistics)``.

    A part becomes a sample only if its image, instruction and label all exist
    and the label has at least one valid rectangle. Everything that is dropped is
    counted in the statistics.
    """
    assignment, leaked_scenes = assign_splits(splits)
    images = {name[:-4] for name in os.listdir(root / "image") if name.endswith(".jpg")}
    labels = {name[:-3] for name in os.listdir(root / "grasp_label_positive") if name.endswith(".pt")}
    instruction_stems = sorted(
        name[:-4] for name in os.listdir(root / "grasp_instructions") if name.endswith(".pkl")
    )

    scenes: dict[str, dict] = {}
    for scene in sorted(images):
        path = root / "scene_description" / f"{scene}.pkl"
        if path.exists():
            description, objects = load_scene_description(path)
            scenes[scene] = {"description": description, "objects": objects}

    stats = Counter()
    label_digests: dict[str, dict[str, int]] = defaultdict(dict)  # object id -> digest -> group
    samples: list[Sample] = []
    readable: list[str] = []
    for stem in instruction_stems:
        scene = parse_part_id(stem).scene
        stats["parts_total"] += 1
        if scene not in images:
            stats["parts_without_image"] += 1
        elif scene not in scenes:
            stats["parts_without_scene_description"] += 1
        elif stem not in labels:
            stats["parts_without_label"] += 1
        else:
            readable.append(stem)

    def read_part(stem: str):
        return (
            load_grasp_label(root / "grasp_label_positive" / f"{stem}.pt"),
            load_instruction(root / "grasp_instructions" / f"{stem}.pkl"),
        )

    # Reading ~100k tiny files is I/O bound, so threads help; map() keeps the order.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        contents = list(tqdm(pool.map(read_part, readable), total=len(readable), desc="indexing parts"))

    for stem, (label, instruction) in zip(readable, contents):
        part = parse_part_id(stem)
        valid = valid_label_rows(label)
        stats["label_rows_total"] += len(label)
        stats["label_rows_invalid"] += int((~valid).sum())
        if not valid.any():
            stats["parts_without_valid_grasp"] += 1
            continue
        if not instruction:
            stats["parts_with_empty_instruction"] += 1
            continue

        groups = label_digests[part.object_id]
        group = groups.setdefault(hashlib.md5(label.tobytes()).hexdigest(), len(groups))
        object_names = scenes[part.scene]["objects"]
        samples.append(
            Sample(
                id=stem,
                scene=part.scene,
                obj=part.obj,
                part=part.part,
                split=assignment.get(part.object_id),
                instruction=instruction,
                object_name=object_names[part.obj] if part.obj < len(object_names) else "",
                num_grasps=int(valid.sum()),
                label_group=group,
            )
        )

    stats["samples"] = len(samples)
    stats["train_scenes_dropped_for_test_overlap"] = len(leaked_scenes)
    for sample in samples:
        stats[f"samples_{sample.split or 'unlisted'}"] += 1

    manifest = {
        "version": 1,
        "label_image_size": 416,
        "scenes": scenes,
        "samples": [asdict(sample) for sample in samples],
    }
    return manifest, dict(stats)


def save_manifest(manifest: dict, root: Path) -> Path:
    path = root / MANIFEST_NAME
    with open(path, "w", encoding="utf-8") as file:
        json.dump(manifest, file)
    return path


def load_manifest(root: Path) -> dict:
    path = Path(root) / MANIFEST_NAME
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: python scripts/verify_data.py --root {root}")
    with open(path, encoding="utf-8") as file:
        return json.load(file)
