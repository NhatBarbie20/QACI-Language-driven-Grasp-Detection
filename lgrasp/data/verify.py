"""Integrity and consistency checks for a downloaded dataset.

Byte-level integrity (CRC-32 of every file) is already enforced while
downloading. This module checks that the dataset is *usable and complete*:

* every scene of the official splits has an image that decodes to 416x416 RGB;
* every file the remote archives hold for the selected scenes is on disk;
* how many listed objects actually have a language annotation (the released
  archives do not cover every object of the split lists);
* label values are sane, and how many rows had to be discarded;
* which scenes are shared between splits;
* how often different parts of an object carry identical grasp labels.

Errors make the dataset unfit for training. Warnings are properties of the
released data that the experiments must account for.
"""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

from PIL import Image
from tqdm import tqdm

from .download import INDEX_DIR
from .grasp import LABEL_IMAGE_SIZE
from .manifest import build_manifest
from .splits import scene_of


def _check_image(path: Path) -> str | None:
    """Return a problem description, or None if the image is fine."""
    try:
        with Image.open(path) as image:
            image.load()
            if image.size != (LABEL_IMAGE_SIZE, LABEL_IMAGE_SIZE):
                return f"{path.name}: unexpected size {image.size}"
            if image.mode != "RGB":
                return f"{path.name}: unexpected mode {image.mode}"
    except Exception as error:  # any decoding failure is a broken file
        return f"{path.name}: cannot decode ({error})"
    return None


def _check_against_archive_index(root: Path, errors: list[str]) -> dict:
    """Compare the files on disk with the member lists saved while downloading.

    The index holds every archive member that belongs to the selected scenes, so
    a file that is listed there but absent (or of a different size) on disk means
    the download is incomplete.
    """
    summary = {}
    for index_path in sorted((root / INDEX_DIR).glob("*.json")):
        component = index_path.stem
        with open(index_path, encoding="utf-8") as file:
            entries = json.load(file)["entries"]
        directory = root / component
        on_disk = {item.name: item.stat().st_size for item in os.scandir(directory)} if directory.is_dir() else {}
        missing = sum(1 for e in entries if on_disk.get(PurePosixPath(e["name"]).name) != e["size"])
        summary[component] = {"in_archive_for_selected_scenes": len(entries), "missing_or_wrong_size": missing}
        if missing:
            errors.append(f"{component}: {missing} files are missing or truncated, re-run the download")
    if not summary:
        errors.append("no archive index found under _index/: was this folder created by download_data.py?")
    return summary


def verify_dataset(root: Path, splits: dict[str, list[str]], workers: int = 8) -> tuple[dict, dict]:
    """Check ``root`` against the official ``splits``.

    Returns ``(report, manifest)``. ``report["errors"]`` is empty for a usable dataset.
    """
    errors: list[str] = []
    warnings: list[str] = []
    report: dict = {"root": str(root), "errors": errors, "warnings": warnings}

    # 1. Files on disk.
    counts = {}
    for folder in ("image", "scene_description", "grasp_instructions", "grasp_label_positive"):
        directory = root / folder
        if not directory.is_dir():
            errors.append(f"missing folder: {folder}")
            counts[folder] = 0
            continue
        counts[folder] = sum(1 for _ in os.scandir(directory))
        leftovers = [item.name for item in os.scandir(directory) if item.name.endswith(".part")]
        if leftovers:
            errors.append(f"{folder}: {len(leftovers)} unfinished '.part' files, re-run the download")
    report["files"] = counts
    if errors:
        return report, {}
    report["archive_completeness"] = _check_against_archive_index(root, errors)

    # 2. Images decode and have the expected geometry.
    image_paths = [Path(item.path) for item in os.scandir(root / "image")]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        problems = [p for p in tqdm(pool.map(_check_image, image_paths), total=len(image_paths), desc="decoding images") if p]
    report["images"] = {"checked": len(image_paths), "broken": len(problems)}
    errors.extend(problems[:20])

    # 3. Index every part; this reads all instructions and labels.
    manifest, stats = build_manifest(root, splits)
    report["parts"] = stats
    if stats.get("label_rows_total"):
        invalid_share = stats.get("label_rows_invalid", 0) / stats["label_rows_total"]
        report["parts"]["label_rows_invalid_share"] = round(invalid_share, 4)
        if invalid_share > 0:
            warnings.append(f"{invalid_share:.1%} of label rows are degenerate and are ignored")

    # 4. Coverage of the official splits.
    images = {path.stem for path in image_paths}
    usable_objects = {f"{s['scene']}_{s['obj']}" for s in manifest["samples"]}
    coverage = {}
    for name, object_ids in splits.items():
        scenes = {scene_of(i) for i in object_ids}
        missing_images = scenes - images
        missing_objects = [i for i in object_ids if i not in usable_objects]
        coverage[name] = {
            "objects": len(object_ids),
            "scenes": len(scenes),
            "scenes_without_image": len(missing_images),
            "objects_without_usable_part": len(missing_objects),
        }
        if missing_images:
            errors.append(f"{name}: {len(missing_images)} scenes have no image on disk")
        if missing_objects:
            share = len(missing_objects) / len(object_ids)
            warnings.append(
                f"{name}: {len(missing_objects)} listed objects ({share:.1%}) have no language annotation "
                "in the released Grasp-Anything++ archives and cannot be used"
            )
    report["split_coverage"] = coverage

    # 5. Scenes shared between official splits.
    split_scenes = {name: {scene_of(i) for i in ids} for name, ids in splits.items()}
    names = list(split_scenes)
    overlap = {
        f"{a}&{b}": len(split_scenes[a] & split_scenes[b])
        for index, a in enumerate(names)
        for b in names[index + 1 :]
    }
    report["scene_overlap_between_official_splits"] = overlap
    if any(overlap.values()):
        warnings.append(f"official splits share scenes {overlap}; shared scenes are removed from training")

    # 6. Do different parts of an object have different grasps?
    groups_per_object: dict[str, set[int]] = defaultdict(set)
    parts_per_object: Counter = Counter()
    for sample in manifest["samples"]:
        object_id = f"{sample['scene']}_{sample['obj']}"
        groups_per_object[object_id].add(sample["label_group"])
        parts_per_object[object_id] += 1
    multi_part = [o for o, n in parts_per_object.items() if n > 1]
    identical = sum(1 for o in multi_part if len(groups_per_object[o]) == 1)
    report["part_labels"] = {
        "objects": len(parts_per_object),
        "objects_with_several_parts": len(multi_part),
        "of_which_all_parts_share_identical_grasps": identical,
    }
    if multi_part and identical / len(multi_part) > 0.5:
        warnings.append(
            f"{identical / len(multi_part):.0%} of multi-part objects have the same grasps for every part: "
            "labels are effectively object-level"
        )

    # 7. Scenes that allow a counterfactual prompt (two or more annotated objects).
    objects_per_scene: dict[str, set[int]] = defaultdict(set)
    for sample in manifest["samples"]:
        objects_per_scene[sample["scene"]].add(sample["obj"])
    report["scenes"] = {
        "with_samples": len(objects_per_scene),
        "with_two_or_more_objects": sum(1 for objs in objects_per_scene.values() if len(objs) > 1),
    }
    return report, manifest
