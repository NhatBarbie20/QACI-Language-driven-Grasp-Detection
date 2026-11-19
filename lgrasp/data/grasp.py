"""Grasp rectangle geometry and the dense map encoding used for training.

A grasp is stored as a row ``(x, y, angle, length, width)``:

* ``(x, y)``  centre of the rectangle in pixels (x to the right, y down);
* ``angle``   orientation of the gripper axis in radians, in ``[-pi/2, pi/2)``,
  positive = anticlockwise as seen on the displayed image;
* ``length``  gripper opening, the side of the rectangle along the gripper axis
  (``w`` in the dataset);
* ``width``   jaw size, the side perpendicular to the axis (``h`` in the dataset).

The conventions follow GR-ConvNet / LGD so that results are comparable: the
dataset angle is in degrees with the opposite sign, and the training target
marks the central third of each rectangle.
"""

from __future__ import annotations

import numpy as np
from skimage.draw import polygon

LABEL_IMAGE_SIZE = 416  # side of the images the labels refer to, in pixels
MIN_SIDE_PIXELS = 2.0  # rectangles thinner than this (at 416 px) are annotation noise
GRASP_COLUMNS = ("x", "y", "angle", "length", "width")


def wrap_angle(angle: np.ndarray | float) -> np.ndarray | float:
    """Wrap an angle to ``[-pi/2, pi/2)``; a grasp is symmetric under a half turn."""
    return (angle + np.pi / 2) % np.pi - np.pi / 2


def valid_label_rows(label: np.ndarray, image_size: int = LABEL_IMAGE_SIZE) -> np.ndarray:
    """Boolean mask of label rows that describe a usable rectangle.

    A small share of the automatically generated labels is degenerate (sub-pixel
    size, centre outside the image, non-finite values); these rows are dropped.
    """
    x, y, w, h = label[:, 1], label[:, 2], label[:, 3], label[:, 4]
    return (
        np.isfinite(label).all(axis=1)
        & (w >= MIN_SIDE_PIXELS)
        & (h >= MIN_SIDE_PIXELS)
        & (x >= 0)
        & (x < image_size)
        & (y >= 0)
        & (y < image_size)
    )


def label_to_grasps(label: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Convert dataset label rows ``[N, 6]`` to grasps ``[M, 5]``, dropping invalid rows.

    ``scale`` resizes the rectangles, e.g. ``224 / 416`` for a 224 px input.
    """
    label = label[valid_label_rows(label)]
    grasps = np.empty((len(label), 5), dtype=np.float32)
    grasps[:, 0] = label[:, 1] * scale
    grasps[:, 1] = label[:, 2] * scale
    grasps[:, 2] = wrap_angle(-np.deg2rad(label[:, 5]))
    grasps[:, 3] = label[:, 3] * scale
    grasps[:, 4] = label[:, 4] * scale
    return grasps


def grasp_corners(grasps: np.ndarray) -> np.ndarray:
    """Corner points ``[N, 4, 2]`` (x, y) of grasps ``[N, 5]``, in drawing order."""
    grasps = np.asarray(grasps, dtype=np.float64).reshape(-1, 5)
    x, y, angle, length, width = grasps.T
    # Unit vector of the gripper axis and of its normal, in image coordinates (y down).
    axis = np.stack([np.cos(angle), -np.sin(angle)], axis=1) * (length / 2)[:, None]
    normal = np.stack([np.sin(angle), np.cos(angle)], axis=1) * (width / 2)[:, None]
    centre = np.stack([x, y], axis=1)
    return np.stack(
        [centre - axis - normal, centre + axis - normal, centre + axis + normal, centre - axis + normal],
        axis=1,
    )


def rasterize(corners: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    """Pixel indices ``(rows, cols)`` covered by one rectangle ``[4, 2]`` on a ``size`` x ``size`` grid."""
    return polygon(corners[:, 1], corners[:, 0], shape=(size, size))


def render_grasp_maps(grasps: np.ndarray, size: int) -> dict[str, np.ndarray]:
    """Encode grasps as the dense training target of generative grasp networks.

    Returns float32 maps of shape ``[size, size]``:

    * ``quality``  1 inside the central third of any rectangle, else 0;
    * ``cos``, ``sin``  ``cos(2*angle)`` and ``sin(2*angle)`` (0-angle outside);
    * ``width``  gripper opening divided by ``size / 2`` and clipped to ``[0, 1]``.
    """
    quality = np.zeros((size, size), dtype=np.float32)
    angle = np.zeros((size, size), dtype=np.float32)
    opening = np.zeros((size, size), dtype=np.float32)

    centre_thirds = np.array(grasps, dtype=np.float32).reshape(-1, 5)
    centre_thirds[:, 3] /= 3.0
    for grasp, corners in zip(grasps, grasp_corners(centre_thirds)):
        rows, cols = rasterize(corners, size)
        quality[rows, cols] = 1.0
        angle[rows, cols] = grasp[2]
        opening[rows, cols] = grasp[3]

    return {
        "quality": quality,
        "cos": np.cos(2 * angle).astype(np.float32),
        "sin": np.sin(2 * angle).astype(np.float32),
        "width": np.clip(opening / (size / 2), 0.0, 1.0),
    }


def flip_grasps(grasps: np.ndarray, size: int, horizontal: bool = False, vertical: bool = False) -> np.ndarray:
    """Mirror grasps ``[N, 5]`` together with an image of side ``size``.

    Either mirror reverses the sense of rotation, so the angle changes sign.
    """
    grasps = grasps.copy()
    if horizontal:
        grasps[:, 0] = size - grasps[:, 0]
        grasps[:, 2] = wrap_angle(-grasps[:, 2])
    if vertical:
        grasps[:, 1] = size - grasps[:, 1]
        grasps[:, 2] = wrap_angle(-grasps[:, 2])
    return grasps
