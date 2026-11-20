"""Training-free comparison of *sources* of a grounding map (companion of probe_grounding.py).

probe_grounding.py varies how frozen CLIP patch features are scored. This script
keeps the scoring simple and varies where the signal comes from:

patch     cosine between CLIP patch features and the text (the reference);
window    CLIP *global* image embeddings of overlapping crops. Every pixel gets
          the mean cosine of the crops that cover it. CLIP was trained on whole
          images, so crop embeddings are better aligned with text than patches;
box       the same global embedding, computed on the box around the ground-truth
          grasps of each annotated object. Uses ground-truth object locations:
          an upper bound for "propose regions, then rank them with CLIP";
clipseg   the logits of a frozen CLIPSeg model (optional, --clipseg).

Metrics are those of probe_grounding.py (object / pair / point); ``box`` has no
map, so only object and pair are defined for it.

    python scripts/probe_grounding_sources.py --splits val test_unseen --clipseg
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lgrasp.data.dataset import SPLITS, GraspAnythingPP  # noqa: E402
from lgrasp.data.manifest import load_manifest  # noqa: E402
from lgrasp.models.clip_dense import DEFAULT_CLIP, ClipDense  # noqa: E402
from lgrasp.text import encode_texts  # noqa: E402
from probe_grounding import NOUN_TEMPLATE, collect  # noqa: E402

WINDOWS = ((112, 37), (160, 32))  # (side, stride) of the crops, in pixels of the 224 px image
CLIPSEG = "CIDAS/clipseg-rd64-refined"


class ClipGlobal(ClipDense):
    """Global CLIP embedding of a batch of ImageNet-normalised images (any size)."""

    @torch.no_grad()
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        image = image * self.scale + self.shift
        side = self.native_resolution
        if image.shape[-2:] != (side, side):
            image = F.interpolate(image, size=(side, side), mode="bicubic", align_corners=False)
        pooled = self.vision(pixel_values=image).pooler_output
        return F.normalize(self.projection(pooled).float(), dim=-1)


def window_boxes(size: int) -> list[tuple[int, int, int, int]]:
    boxes = []
    for side, stride in WINDOWS:
        starts = list(range(0, size - side + 1, stride))
        if starts[-1] != size - side:
            starts.append(size - side)
        boxes += [(x, y, x + side, y + side) for y in starts for x in starts]
    return boxes


def region_box(region: torch.Tensor, size: int, margin: float = 0.25, minimum: int = 48) -> tuple[int, int, int, int]:
    """Square-ish box around a binary region, enlarged by ``margin``."""
    rows, cols = torch.nonzero(region, as_tuple=True)
    x0, x1, y0, y1 = cols.min().item(), cols.max().item() + 1, rows.min().item(), rows.max().item() + 1
    side = max(x1 - x0, y1 - y0, minimum) * (1 + margin)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    box = [cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2]
    return tuple(int(min(max(round(value), 0), size)) for value in box)


def crops(image: torch.Tensor, boxes: list[tuple[int, int, int, int]], side: int) -> torch.Tensor:
    return torch.cat([
        F.interpolate(image[:, :, y0:y1, x0:x1], size=(side, side), mode="bicubic", align_corners=False)
        for x0, y0, x1, y1 in boxes
    ])


def map_metrics(matching: torch.Tensor, mismatching: torch.Tensor, item: dict, size: int) -> np.ndarray:
    inside = matching[item["region"]].mean()
    object_hit = all(inside > matching[region].mean() for region in item["other_regions"])
    pair_hit = inside > mismatching[item["region"]].mean()
    peak = int(matching.flatten().argmax())
    peak_xy = torch.tensor([peak % size, peak // size], dtype=torch.float32)
    point_hit = (item["centres"] - peak_xy).norm(dim=1).min() < 20
    return np.array([bool(object_hit), bool(pair_hit), bool(point_hit)], dtype=float)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("data/grasp_anything_pp"))
    parser.add_argument("--splits", nargs="+", default=["val", "test_unseen"], choices=SPLITS)
    parser.add_argument("--clip", default=DEFAULT_CLIP)
    parser.add_argument("--clipseg", action="store_true", help="also evaluate a frozen CLIPSeg model")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--out", type=Path, default=Path("outputs/probe_grounding_sources.json"))
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    manifest = load_manifest(args.root)
    names = {sample["instruction"]: sample["object_name"] for sample in manifest["samples"]}
    data = {split: GraspAnythingPP(args.root, split) for split in args.splits}
    records = {split: collect(dataset, names, args.max_samples) for split, dataset in data.items()}

    nouns = sorted({name for items in records.values() for item in items for name in (item["noun"], item["negative_noun"])})
    vectors = dict(zip(nouns, encode_texts([NOUN_TEMPLATE.format(name) for name in nouns], args.clip, device=device).to(device)))

    patch_model = ClipDense(args.clip, 224).to(device)
    global_model = ClipGlobal(args.clip, 224).to(device)
    native = global_model.native_resolution
    segmenter = processor = None
    if args.clipseg:
        from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor

        processor = CLIPSegProcessor.from_pretrained(CLIPSEG)
        segmenter = CLIPSegForImageSegmentation.from_pretrained(CLIPSEG).to(device).eval()

    results = []
    for split, items in records.items():
        dataset = data[split]
        size = dataset.image_size
        boxes = window_boxes(size)
        cover = torch.zeros(len(boxes), size, size)
        for index, (x0, y0, x1, y1) in enumerate(boxes):
            cover[index, y0:y1, x0:x1] = 1
        cover = cover / cover.sum(dim=0).clamp(min=1)

        sources = ["patch", "window", "patch+window", "box"] + (["clipseg"] if segmenter else [])
        hits = {source: np.zeros(3) for source in sources}
        scene, cache = None, {}
        for item in items:
            if item["scene"] != scene:
                scene = item["scene"]
                image = dataset[item["index"]]["image"][None].to(device)
                cache = {
                    "image": image,
                    "patch": patch_model(image)[0],
                    "window": global_model(crops(image, boxes, native)),
                    "segmentation": {},
                }
            query, swapped = vectors[item["noun"]], vectors[item["negative_noun"]]

            patch = torch.einsum("dhw,kd->khw", cache["patch"], torch.stack([query, swapped]))
            patch = F.interpolate(patch[None], size=(size, size), mode="bilinear", align_corners=False)[0].cpu()
            window = torch.einsum("nk,nhw->khw", (cache["window"] @ torch.stack([query, swapped]).T).cpu(), cover)
            hits["patch"] += map_metrics(patch[0], patch[1], item, size)
            hits["window"] += map_metrics(window[0], window[1], item, size)

            def standardise(maps: torch.Tensor) -> torch.Tensor:
                return (maps - maps.mean(dim=(1, 2), keepdim=True)) / maps.std(dim=(1, 2), keepdim=True).clamp(min=1e-6)

            both = standardise(patch) + standardise(window)
            hits["patch+window"] += map_metrics(both[0], both[1], item, size)

            # Global embedding of the box around every annotated object.
            object_boxes = [region_box(region, size) for region in [item["region"], *item["other_regions"]]]
            scores = global_model(crops(cache["image"], object_boxes, native)) @ torch.stack([query, swapped]).T
            object_hit = bool((scores[0, 0] > scores[1:, 0]).all())
            pair_hit = bool(scores[0, 0] > scores[0, 1])
            hits["box"] += np.array([object_hit, pair_hit, 0.0])

            if segmenter:
                maps = []
                for name in (item["noun"], item["negative_noun"]):
                    if name not in cache["segmentation"]:
                        inputs = processor(text=[name], images=[dataset.load_image(item["index"])], padding=True,
                                           return_tensors="pt").to(device)
                        logits = segmenter(**inputs).logits.reshape(1, 1, *inputs["pixel_values"].shape[-2:])
                        cache["segmentation"][name] = F.interpolate(
                            logits.float(), size=(size, size), mode="bilinear", align_corners=False)[0, 0].cpu()
                    maps.append(cache["segmentation"][name])
                hits["clipseg"] += map_metrics(maps[0], maps[1], item, size)

        for source, counts in hits.items():
            rates = counts / len(items)
            results.append({"split": split, "source": source, "samples": len(items),
                            "object": float(rates[0]), "pair": float(rates[1]),
                            "point": None if source == "box" else float(rates[2])})
            point = "  -  " if source == "box" else f"{rates[2]:.3f}"
            print(f"{split:<12}{source:<14}object {rates[0]:.3f}  pair {rates[1]:.3f}  point {point}  ({len(items)} samples)",
                  flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
