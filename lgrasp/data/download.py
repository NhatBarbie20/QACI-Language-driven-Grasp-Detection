"""Download the part of Grasp-Anything++ that the official splits use.

The full dataset is about 80 GB (65 GB of images). The official LGD splits only
reference ~12k of the 1M scenes, so this module pulls exactly those scenes out
of the remote archives, member by member, and checks every file's CRC-32.

Resulting layout (same folder names as the upstream archives)::

    <root>/image/<scene>.jpg
    <root>/scene_description/<scene>.pkl
    <root>/grasp_instructions/<scene>_<obj>_<part>.pkl
    <root>/grasp_label_positive/<scene>_<obj>_<part>.pt
    <root>/part_mask/<scene>_<obj>_<part>.npy        (optional)
    <root>/grasp_label_negative/<scene>_<obj>_<part>.pt  (optional)

All annotations of a selected scene are downloaded, including objects that are
not listed in a split, because they provide the alternative prompts used for
counterfactual training.
"""

from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable

from tqdm import tqdm

from .remote_zip import RemoteBlob, RemoteZip, ZipEntry
from .splits import SCENE_ID_LENGTH

GRASP_ANYTHING_REPO = "airvlab/Grasp-Anything"
GRASP_ANYTHING_PP_REPO = "airvlab/Grasp-Anything-pp"
INDEX_DIR = "_index"


@dataclass(frozen=True)
class Component:
    """One kind of dataset file and the remote archive that holds it."""

    name: str  # also the output folder name
    repo: str
    files: tuple[str, ...]  # archive parts, in order


COMPONENTS: dict[str, Component] = {
    component.name: component
    for component in (
        Component("image", GRASP_ANYTHING_REPO, ("image_part_aa", "image_part_ab")),
        Component("scene_description", GRASP_ANYTHING_REPO, ("scene_description.zip",)),
        Component("grasp_instructions", GRASP_ANYTHING_PP_REPO, ("grasp_instructions.zip",)),
        Component("grasp_label_positive", GRASP_ANYTHING_PP_REPO, ("grasp_label_positive.zip",)),
        Component("part_mask", GRASP_ANYTHING_PP_REPO, ("part_mask.zip",)),
        Component("grasp_label_negative", GRASP_ANYTHING_PP_REPO, ("grasp_label_negative.zip",)),
    )
}
DEFAULT_COMPONENTS = ("image", "scene_description", "grasp_instructions", "grasp_label_positive")


@dataclass
class DownloadResult:
    component: str
    expected: int
    downloaded: int
    skipped: int
    failed: list[str]

    @property
    def ok(self) -> bool:
        return not self.failed


def _scene_of_member(name: str) -> str | None:
    """Scene hash of an archive member, or None for directories and stray files."""
    base = PurePosixPath(name).name
    if name.endswith("/") or len(base) < SCENE_ID_LENGTH:
        return None
    return base[:SCENE_ID_LENGTH]


def _selection_key(scenes: Iterable[str]) -> str:
    """Fingerprint of a scene selection, used to validate a cached index."""
    digest = hashlib.sha256()
    for scene in sorted(scenes):
        digest.update(scene.encode())
    return digest.hexdigest()


def _load_or_build_index(
    archive: RemoteZip, component: Component, scenes: set[str], root: Path
) -> list[ZipEntry]:
    """Return the archive members that belong to ``scenes``.

    Scanning a central directory means reading up to ~600 MB, so the filtered
    result is cached under ``<root>/_index`` and reused while the selection is
    unchanged.
    """
    index_path = root / INDEX_DIR / f"{component.name}.json"
    key = _selection_key(scenes)
    if index_path.exists():
        with open(index_path, encoding="utf-8") as file:
            cached = json.load(file)
        if cached.get("selection") == key:
            return [ZipEntry.from_dict(item) for item in cached["entries"]]

    with tqdm(unit="B", unit_scale=True, desc=f"{component.name}: reading archive index") as bar:
        entries, total = archive.scan(lambda name: _scene_of_member(name) in scenes, bar.update)
    entries.sort(key=lambda entry: entry.offset)

    index_path.parent.mkdir(parents=True, exist_ok=True)
    with open(index_path, "w", encoding="utf-8") as file:
        json.dump(
            {"selection": key, "archive_members": total, "entries": [entry.to_dict() for entry in entries]},
            file,
        )
    return entries


def _write_atomic(path: Path, data: bytes) -> None:
    """Write through a temporary file so an interrupted run never leaves a partial file."""
    temporary = path.with_name(path.name + ".part")
    with open(temporary, "wb") as file:
        file.write(data)
    os.replace(temporary, path)


def download_component(
    component: Component,
    scenes: set[str],
    root: Path,
    workers: int = 32,
    token: str | None = None,
) -> DownloadResult:
    """Download every file of ``component`` that belongs to ``scenes``.

    Files already on disk with the expected size are kept, so the command can be
    re-run to resume or repair a download.
    """
    archive = RemoteZip([RemoteBlob(component.repo, filename, token=token) for filename in component.files])
    entries = _load_or_build_index(archive, component, scenes, root)

    out_dir = root / component.name
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = {item.name: item.stat().st_size for item in os.scandir(out_dir) if item.is_file()}
    pending = [
        entry for entry in entries if existing.get(PurePosixPath(entry.name).name) != entry.size
    ]

    def fetch(entry: ZipEntry) -> None:
        _write_atomic(out_dir / PurePosixPath(entry.name).name, archive.fetch(entry))

    failed: list[str] = []
    if pending:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(fetch, entry): entry for entry in pending}
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"{component.name}: downloading"):
                if future.exception() is not None:
                    failed.append(f"{futures[future].name}: {future.exception()}")

    return DownloadResult(
        component=component.name,
        expected=len(entries),
        downloaded=len(pending) - len(failed),
        skipped=len(entries) - len(pending),
        failed=failed,
    )


def download_subset(
    scenes: set[str],
    root: Path,
    components: Iterable[str] = DEFAULT_COMPONENTS,
    workers: int = 32,
    token: str | None = None,
) -> list[DownloadResult]:
    """Download the requested components for ``scenes`` into ``root``."""
    root.mkdir(parents=True, exist_ok=True)
    results = []
    for name in components:
        if name not in COMPONENTS:
            raise ValueError(f"Unknown component {name!r}; choose from {sorted(COMPONENTS)}")
        results.append(download_component(COMPONENTS[name], scenes, root, workers, token))
    return results
