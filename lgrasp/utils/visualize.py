"""Plotting helpers for grasps, target maps and dataset samples."""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from lgrasp.data.grasp import grasp_corners  # noqa: E402

GROUND_TRUTH_COLOR = "#2ecc71"
PREDICTION_COLOR = "#e74c3c"


def draw_grasps(ax, grasps: np.ndarray, color: str = GROUND_TRUTH_COLOR, linewidth: float = 1.5) -> None:
    """Draw grasp rectangles ``[N, 5]`` on a matplotlib axis.

    The two sides that correspond to the gripper jaws are drawn thicker, so the
    orientation of the grasp is visible.
    """
    for corners in grasp_corners(np.asarray(grasps)):
        closed = np.vstack([corners, corners[:1]])
        ax.plot(closed[:, 0], closed[:, 1], color=color, linewidth=linewidth * 0.6)
        for start, end in ((1, 2), (3, 0)):  # the sides perpendicular to the gripper axis
            ax.plot(corners[[start, end], 0], corners[[start, end], 1], color=color, linewidth=linewidth * 1.8)


def show_image(ax, image, title: str = "", wrap: int = 34) -> None:
    ax.imshow(image)
    ax.set_title("\n".join(textwrap.wrap(title, wrap)), fontsize=8)
    ax.axis("off")


def save_sample_grid(dataset, indices: Sequence[int], path: str | Path, columns: int = 4) -> Path:
    """Save a figure with, for each sample, the image with its ground-truth
    grasps and the quality target map next to it."""
    rows = int(np.ceil(len(indices) / columns))
    figure, axes = plt.subplots(rows, columns * 2, figsize=(columns * 4.4, rows * 2.6), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")

    for position, index in enumerate(indices):
        item = dataset[index]
        row, column = divmod(position, columns)
        image_ax, map_ax = axes[row, column * 2], axes[row, column * 2 + 1]

        show_image(image_ax, dataset.load_image(index), item["text"])
        draw_grasps(image_ax, item["grasps"].numpy())
        image_ax.set_xlim(0, dataset.image_size)
        image_ax.set_ylim(dataset.image_size, 0)

        map_ax.imshow(item["quality"][0].numpy(), cmap="magma", vmin=0, vmax=1)
        map_ax.set_title(f"quality target ({len(item['grasps'])} grasps)", fontsize=8)

    figure.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=130)
    plt.close(figure)
    return path


def save_prediction_grid(dataset, records: Sequence[dict], path: str | Path) -> Path:
    """Save qualitative results: one column per sample, two rows.

    Top row: prediction for the matching instruction. Bottom row: prediction for
    an instruction that names another object of the same image. Ground truth of
    the matching instruction is green, predictions are red.
    """
    index_of = {sample["id"]: index for index, sample in enumerate(dataset.samples)}
    figure, axes = plt.subplots(2, len(records), figsize=(len(records) * 2.3, 5.2), squeeze=False)
    for column, record in enumerate(records):
        image = dataset.load_image(index_of[record["id"]])
        rows = (
            (record["text"], record["prediction"], "correct" if record["correct"] else "wrong"),
            (record["swapped_text"], record["swapped_prediction"], "still on first object" if record["stuck"] else "moved"),
        )
        for ax, (text, prediction, verdict) in zip(axes[:, column], rows):
            show_image(ax, image, f"{text}\n[{verdict}]", wrap=28)
            draw_grasps(ax, record["ground_truth"], GROUND_TRUTH_COLOR, linewidth=0.8)
            draw_grasps(ax, prediction[None], PREDICTION_COLOR, linewidth=2.0)
            ax.set_xlim(0, dataset.image_size)
            ax.set_ylim(dataset.image_size, 0)
    figure.tight_layout()
    path = Path(path)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path
