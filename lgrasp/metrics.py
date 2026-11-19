"""Grasp decoding and evaluation metrics.

Success rate
    The standard rectangle metric (Jiang et al., ICRA 2011), as used by
    GR-ConvNet and LGD: the predicted grasp is correct if it has an IoU above
    25% with at least one ground-truth rectangle whose orientation differs by
    less than 30 degrees.

Language metrics (proposed)
    Success rate cannot tell whether a model *used* the instruction: grasping
    the most salient object is often correct by chance. For every test sample
    whose image contains another annotated object, the model is queried a second
    time with an instruction for that other object.

    ``ignore_rate``  among samples solved with the matching instruction, the share
                     where the prediction under the swapped instruction still
                     lands on the original object. 1.0 for a vision-only model.
    ``paired_sr``    share of samples that are solved with the matching instruction
                     *and* move away under the swapped one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter

from lgrasp.data.grasp import grasp_corners, rasterize, wrap_angle

IOU_THRESHOLD = 0.25
ANGLE_THRESHOLD = np.deg2rad(30)


def decode_grasp(quality: np.ndarray, cos: np.ndarray, sin: np.ndarray, width: np.ndarray) -> np.ndarray:
    """Turn dense maps ``[H, W]`` into the single best grasp ``(x, y, angle, length, width)``.

    ``quality`` must already be a probability. The maps are smoothed as in
    GR-ConvNet before the best pixel is selected. The jaw size is set to half the
    opening, the convention of the reference implementation.
    """
    size = quality.shape[0]
    quality = gaussian_filter(quality, sigma=2.0)
    angle = gaussian_filter(0.5 * np.arctan2(sin, cos), sigma=1.0)
    width = gaussian_filter(width, sigma=1.0)

    row, col = np.unravel_index(int(np.argmax(quality)), quality.shape)
    length = max(float(width[row, col]) * size / 2, 1.0)
    return np.array([col, row, angle[row, col], length, length / 2], dtype=np.float32)


def rectangle_iou(first: np.ndarray, second: np.ndarray, size: int) -> float:
    """IoU of two grasp rectangles, by rasterisation on a padded canvas."""
    pad = size // 2
    canvas_size = size + 2 * pad
    canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
    for grasp in (first, second):
        rows, cols = rasterize(grasp_corners(grasp)[0] + pad, canvas_size)
        canvas[rows, cols] += 1
    union = int((canvas > 0).sum())
    return float((canvas == 2).sum()) / union if union else 0.0


def is_correct(prediction: np.ndarray, ground_truth: np.ndarray, size: int) -> bool:
    """Whether ``prediction`` matches any ground-truth grasp ``[N, 5]`` under the rectangle metric."""
    for target in ground_truth:
        if abs(wrap_angle(prediction[2] - target[2])) > ANGLE_THRESHOLD:
            continue
        if rectangle_iou(prediction, target, size) > IOU_THRESHOLD:
            return True
    return False


@dataclass
class GraspMeter:
    """Accumulates per-sample outcomes and reports the metrics."""

    samples: int = 0
    correct: int = 0
    swappable: int = 0  # samples that have a swapped instruction
    swappable_correct: int = 0  # ... and are solved with the matching instruction
    stuck: int = 0  # ... and the swapped prediction still hits the original object

    def update(self, correct: bool, stuck: bool | None = None) -> None:
        """Record one sample. ``stuck`` is None when the image has no other object."""
        self.samples += 1
        self.correct += int(correct)
        if stuck is not None:
            self.swappable += 1
            if correct:
                self.swappable_correct += 1
                self.stuck += int(stuck)

    def compute(self) -> dict[str, float]:
        return {
            "success_rate": self.correct / max(self.samples, 1),
            "ignore_rate": self.stuck / max(self.swappable_correct, 1),
            "paired_sr": (self.swappable_correct - self.stuck) / max(self.swappable, 1),
            "samples": self.samples,
            "swappable_samples": self.swappable,
        }


def harmonic_mean(seen: float, unseen: float) -> float:
    """The ``H`` score reported by LGD for seen / unseen success rates."""
    return 2 * seen * unseen / (seen + unseen) if seen + unseen > 0 else 0.0
