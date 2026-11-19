"""Draw the method overview figure of the paper.

    python scripts/make_method_figure.py        # writes paper/figures/method.pdf (and .png)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

BLUE, ORANGE, GREEN, RED, GREY = "#dbe9f6", "#fde5cc", "#dff0d8", "#f8d7da", "#eeeeee"
EDGE = "#444444"


def box(ax, x, y, w, h, text, color, fontsize=7.5, style="round,pad=0.02"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style, facecolor=color, edgecolor=EDGE, linewidth=0.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize)


def arrow(ax, start, end, color=EDGE, style="-|>", linestyle="-"):
    ax.add_patch(
        FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=7, color=color, linewidth=0.9,
                        linestyle=linestyle, shrinkA=0, shrinkB=0)
    )


def main() -> None:
    figure, ax = plt.subplots(figsize=(6.9, 2.15))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.1)
    ax.axis("off")

    # Vision path.
    box(ax, 0.05, 1.25, 0.9, 0.6, "image $I$", GREY)
    box(ax, 1.3, 1.25, 1.1, 0.6, "conv stem\n(shared)", BLUE)
    box(ax, 2.9, 1.25, 1.9, 0.6, "5 residual blocks\n+ FiLM", BLUE)
    box(ax, 5.3, 1.25, 1.1, 0.6, "decoder\n+ heads", BLUE)
    arrow(ax, (0.95, 1.55), (1.3, 1.55))
    arrow(ax, (2.4, 1.55), (2.9, 1.55))
    arrow(ax, (4.8, 1.55), (5.3, 1.55))

    # Language path.
    box(ax, 0.05, 2.45, 2.35, 0.5, "$t^{+}$: “Grasp the knife”", GREEN)
    box(ax, 0.05, 0.15, 2.35, 0.5, "$t^{-}$: “Grasp the fork”", RED)
    box(ax, 2.9, 2.45, 1.9, 0.5, "frozen CLIP text encoder", ORANGE)
    box(ax, 2.9, 0.15, 1.9, 0.5, "frozen CLIP text encoder", ORANGE)
    arrow(ax, (2.4, 2.7), (2.9, 2.7))
    arrow(ax, (2.4, 0.4), (2.9, 0.4))
    arrow(ax, (3.85, 2.45), (3.85, 1.85), color="#2e7d32")
    arrow(ax, (3.85, 0.65), (3.85, 1.25), color="#c62828", linestyle="--")
    ax.text(3.95, 2.13, r"$\gamma,\beta$", fontsize=7.5, color="#2e7d32")
    ax.text(3.95, 0.88, r"$\gamma,\beta$", fontsize=7.5, color="#c62828")

    # Outputs.
    box(ax, 6.9, 2.2, 1.35, 0.75, "$Q^{+}$, $\\cos2\\theta$,\n$\\sin2\\theta$, $W$", GREEN)
    box(ax, 6.9, 0.15, 1.35, 0.75, "$Q^{-}$", RED)
    arrow(ax, (6.4, 1.7), (6.9, 2.45), color="#2e7d32")
    arrow(ax, (6.4, 1.4), (6.9, 0.6), color="#c62828", linestyle="--")

    # Losses.
    box(ax, 8.75, 2.2, 1.2, 0.75, r"$\mathcal{L}_{grasp}$" + "\n(vs. ground truth)", GREY)
    box(ax, 8.75, 1.05, 1.2, 0.95, r"$\mathcal{L}_{cf}$" + "\n$Q^{+}>Q^{-}+m$\non region $R$", GREY)
    arrow(ax, (8.25, 2.57), (8.75, 2.57))
    arrow(ax, (8.25, 2.3), (8.75, 1.75), color="#2e7d32")
    arrow(ax, (8.25, 0.55), (8.75, 1.3), color="#c62828", linestyle="--")

    figure.subplots_adjust(left=0, right=1, top=1, bottom=0)
    out = Path("paper/figures")
    out.mkdir(parents=True, exist_ok=True)
    figure.savefig(out / "method.pdf")
    figure.savefig(out / "method.png", dpi=200)
    print(f"Saved {out / 'method.pdf'}")


if __name__ == "__main__":
    main()
