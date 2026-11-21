"""Draw the method overview figure of the paper.

    python scripts/make_method_figure.py        # writes paper/figures/method.pdf (and .png)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

BLUE, ORANGE, GREEN, GREY = "#dbe9f6", "#fde5cc", "#dff0d8", "#eeeeee"
EDGE = "#444444"


def box(ax, x, y, w, h, text, color, fontsize=7.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", facecolor=color, edgecolor=EDGE,
                                linewidth=0.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize)


def arrow(ax, start, end, color=EDGE, linestyle="-"):
    ax.add_patch(
        FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=7, color=color, linewidth=0.9,
                        linestyle=linestyle, shrinkA=0, shrinkB=0)
    )


def main() -> None:
    figure, ax = plt.subplots(figsize=(3.35, 2.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.1)
    ax.axis("off")

    # Inputs.
    box(ax, 0.05, 3.85, 1.45, 0.9, "image $I$", GREY)
    box(ax, 0.05, 0.3, 2.55, 0.9, "$t$: “grasp the knife”", GREEN, fontsize=6.2)

    # Geometry factor: trained, never sees the instruction.
    box(ax, 2.3, 4.75, 2.9, 1.1, "geometry net\n(trained)", BLUE, fontsize=6.5)
    box(ax, 5.95, 4.75, 1.8, 1.1, r"$G$, $\theta$, $w$", BLUE)
    arrow(ax, (1.5, 4.55), (2.3, 5.2))
    arrow(ax, (5.2, 5.3), (5.95, 5.3))

    # Grounding factor: frozen CLIP similarity and a text-blind refinement.
    box(ax, 2.3, 2.7, 2.9, 1.1, "CLIP image enc.\n(frozen) patches", ORANGE, fontsize=6.5)
    box(ax, 3.1, 0.3, 2.1, 0.9, "CLIP text enc.\n(frozen)", ORANGE, fontsize=6.5)
    box(ax, 5.75, 1.5, 1.3, 1.0, "cosine\nmap $c$", GREY, fontsize=6.5)
    box(ax, 7.45, 1.5, 1.05, 1.0, "refine", BLUE, fontsize=6.5)
    box(ax, 8.9, 1.5, 1.0, 1.0, "$S$", ORANGE)
    arrow(ax, (1.5, 4.05), (2.3, 3.4))
    arrow(ax, (2.6, 0.75), (3.1, 0.75))
    arrow(ax, (5.2, 3.0), (6.0, 2.5))
    arrow(ax, (5.2, 0.9), (6.0, 1.5))
    arrow(ax, (7.05, 2.0), (7.45, 2.0))
    arrow(ax, (8.5, 2.0), (8.9, 2.0))
    # The refinement also reads geometry features, but never the text embedding.
    arrow(ax, (4.6, 4.75), (7.9, 2.5), color="#888888", linestyle=":")

    # Product.
    box(ax, 8.5, 3.4, 1.45, 1.0, r"$Q=G\cdot S$", GREEN, fontsize=6.8)
    arrow(ax, (7.75, 5.1), (8.95, 4.4))
    arrow(ax, (9.4, 2.5), (9.4, 3.4))

    # Supervision.
    ax.text(7.9, 5.65, "all objects' grasps", fontsize=5.6, color="#2e7d32", ha="left", va="center")
    ax.text(8.4, 3.9, "target grasps", fontsize=5.6, color="#c62828", ha="right", va="center")

    figure.subplots_adjust(left=0, right=1, top=1, bottom=0)
    out = Path("paper/figures")
    out.mkdir(parents=True, exist_ok=True)
    figure.savefig(out / "method.pdf")
    figure.savefig(out / "method.png", dpi=300)
    print(f"Saved {out / 'method.pdf'}")


if __name__ == "__main__":
    main()
