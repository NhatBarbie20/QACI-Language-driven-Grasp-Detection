"""How well does frozen CLIP localise the object named in an instruction?

No training is involved. For every sample whose image has a second annotated
object, the patch-text similarity map is computed for the matching instruction
and for the instruction of the other object. Two numbers are reported:

pair accuracy   share of samples where the mean similarity inside the target
                grasp region is higher for the matching instruction than for the
                swapped one (chance = 0.5);
pointing        share of samples where the similarity peak of the matching
                instruction falls within 20 px of a ground-truth grasp centre.

    python scripts/probe_similarity.py --split test_unseen --resolution 448
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lgrasp.data.dataset import SPLITS, GraspAnythingPP  # noqa: E402
from lgrasp.models.clip_dense import DEFAULT_CLIP, ClipDense, similarity_map  # noqa: E402
from lgrasp.text import encode_texts  # noqa: E402
from lgrasp.utils.visualize import draw_grasps, show_image  # noqa: E402


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("data/grasp_anything_pp"))
    parser.add_argument("--split", default="test_unseen", choices=SPLITS)
    parser.add_argument("--clip", default=DEFAULT_CLIP)
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--figure", type=int, default=8, help="number of examples to draw")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = GraspAnythingPP(args.root, args.split)
    indices = [i for i in range(len(dataset)) if dataset.has_negative(i)]
    indices = indices[:: max(1, len(indices) // args.max_samples)][: args.max_samples]
    clip = ClipDense(args.clip, args.resolution).to(device)

    items = [dataset[i] for i in indices]
    texts = sorted({t for item in items for t in (item["text"], item["negative_text"])})
    text_vectors = dict(zip(texts, encode_texts(texts, args.clip, device=device)))

    size = dataset.image_size
    pair_hits = pointing_hits = 0
    drawn = []
    for index, item in zip(indices, items):
        features = clip(item["image"][None].to(device))
        maps = {}
        for key in ("text", "negative_text"):
            coarse = similarity_map(features, text_vectors[item[key]][None].to(device))
            maps[key] = F.interpolate(coarse, size=(size, size), mode="bilinear", align_corners=False)[0, 0].cpu()

        region = item["quality"][0] > 0
        pair_hits += int(maps["text"][region].mean() > maps["negative_text"][region].mean())
        peak = maps["text"].flatten().argmax()
        peak_xy = torch.tensor([peak % size, peak // size], dtype=torch.float32)
        pointing_hits += int((item["grasps"][:, :2] - peak_xy).norm(dim=1).min() < 20)
        if len(drawn) < args.figure:
            drawn.append((index, item, maps))

    total = len(indices)
    grid = features.shape[-1]
    print(f"{args.split} | {args.clip} @ {args.resolution}px (grid {grid}x{grid}) | {total} samples")
    print(f"pair accuracy {pair_hits / total:.3f} (chance 0.500) | pointing within 20px {pointing_hits / total:.3f}")

    figure, axes = plt.subplots(3, len(drawn), figsize=(len(drawn) * 2.2, 6.9), squeeze=False)
    for column, (index, item, maps) in enumerate(drawn):
        image = dataset.load_image(index)
        show_image(axes[0, column], image, item["text"], wrap=26)
        draw_grasps(axes[0, column], item["grasps"].numpy())
        for row, key in ((1, "text"), (2, "negative_text")):
            show_image(axes[row, column], image, item[key] if row == 2 else "similarity, matching", wrap=26)
            axes[row, column].imshow(maps[key], cmap="jet", alpha=0.55, vmin=-2, vmax=3)
    figure.tight_layout()
    out = Path("outputs") / f"similarity_{args.split}_{args.resolution}.png"
    out.parent.mkdir(exist_ok=True)
    figure.savefig(out, dpi=120)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
