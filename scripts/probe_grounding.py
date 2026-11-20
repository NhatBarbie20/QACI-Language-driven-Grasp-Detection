"""Training-free comparison of ways to ground an instruction with frozen CLIP.

Every variant turns (image, instruction) into a score map; no weight is trained.
The variants differ in

read-out     how patch features are taken from CLIP (``value`` | ``qq``) and at
             which input resolution;
text         what is encoded: the full instruction (``sentence``) or only the
             object it names, as "a photo of a {object}" (``noun``);
score        ``abs``    cosine between patch and text;
             ``bank``   log-probability of the text in a softmax over the text and
                        a bank of distractors: the object names of the training
                        split plus a few background words. Uses nothing from the
                        evaluated image;
             ``scene``  the same, with the other annotated objects of the image
                        added to the bank. This is available during training only
                        and is reported as a reference.

Only samples whose image has a second annotated object are used. Metrics:

object    the mean score inside the grasp region of the named object is higher
          than inside the region of every other annotated object (same prompt).
          This is what a product ``geometry x grounding`` needs.
pair      the mean score inside the region of the named object is higher for the
          matching instruction than for the instruction of another object.
point     the peak of the map lies within 20 px of a ground-truth grasp centre.

    python scripts/probe_grounding.py --splits val test_unseen --resolutions 224 448
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lgrasp.data.dataset import SPLITS, GraspAnythingPP  # noqa: E402
from lgrasp.data.grasp import render_grasp_maps  # noqa: E402
from lgrasp.data.manifest import load_manifest  # noqa: E402
from lgrasp.models.clip_dense import DEFAULT_CLIP, READOUTS, ClipDense  # noqa: E402
from lgrasp.text import NOUN_TEMPLATE, encode_texts  # noqa: E402

BACKGROUND = ("background", "table", "floor", "wall", "surface")
# (text, score) combinations that are evaluated.
VARIANTS = (("sentence", "abs"), ("noun", "abs"), ("noun", "bank"), ("noun", "scene"))


def object_of(sample_id: str) -> int:
    """Object index of a sample id ``<scene>_<obj>_<part>``."""
    return int(sample_id.rsplit("_", 2)[1])


def collect(dataset: GraspAnythingPP, names: dict[str, str], max_samples: int | None) -> list[dict]:
    """Everything the probe needs about each multi-object sample, without the image."""
    indices = [i for i in range(len(dataset)) if dataset.has_negative(i)]
    if max_samples is not None:
        indices = indices[:: max(1, len(indices) // max_samples)][:max_samples]
    size = dataset.image_size
    records = []
    for index in indices:
        sample = dataset.samples[index]
        grasps = dataset.load_grasps(index)
        others: dict[int, np.ndarray] = {}  # object index -> grasp region
        for label_id in dataset._scene_label_ids[sample["scene"]]:
            obj = object_of(label_id)
            if obj != sample["obj"]:
                region = render_grasp_maps(dataset._grasps_of(label_id), size)["quality"] > 0
                others[obj] = others[obj] | region if obj in others else region
        scene_objects = dataset.scenes[sample["scene"]]["objects"]
        negative = dataset._negative_text(sample)
        records.append(
            {
                "index": index,
                "scene": sample["scene"],
                "sentence": sample["instruction"],
                "noun": sample["object_name"],
                "negative_sentence": negative,
                "negative_noun": names[negative],
                "scene_nouns": [name for name in scene_objects if name != sample["object_name"]],
                "region": torch.from_numpy(render_grasp_maps(grasps, size)["quality"] > 0),
                "other_regions": [torch.from_numpy(region) for region in others.values()],
                "centres": torch.from_numpy(grasps[:, :2]),
            }
        )
    return records


def score_maps(features: torch.Tensor, query: torch.Tensor, bank: torch.Tensor | None, temperature: float) -> torch.Tensor:
    """Score map ``[h, w]`` of one image for one query vector.

    Without a bank this is the cosine similarity. With a bank ``[K, D]`` it is the
    log-probability of the query among ``{query} + bank`` at temperature ``temperature``.
    """
    cosine = torch.einsum("dhw,d->hw", features, query)
    if bank is None:
        return cosine
    logits = torch.cat([cosine[None], torch.einsum("dhw,kd->khw", features, bank)]) / temperature
    return logits[0] - torch.logsumexp(logits, dim=0)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("data/grasp_anything_pp"))
    parser.add_argument("--splits", nargs="+", default=["val", "test_unseen"], choices=SPLITS)
    parser.add_argument("--clip", default=DEFAULT_CLIP)
    parser.add_argument("--readouts", nargs="+", default=list(READOUTS), choices=READOUTS)
    parser.add_argument("--resolutions", nargs="+", type=int, default=[224, 448])
    parser.add_argument("--temperatures", nargs="+", type=float, default=[0.01, 0.05])
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--out", type=Path, default=Path("outputs/probe_grounding.json"))
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    manifest = load_manifest(args.root)
    names = {sample["instruction"]: sample["object_name"] for sample in manifest["samples"]}
    train_nouns = sorted({sample["object_name"] for sample in manifest["samples"] if sample["split"] == "train"})

    data = {split: GraspAnythingPP(args.root, split) for split in args.splits}
    records = {split: collect(dataset, names, args.max_samples) for split, dataset in data.items()}
    for split, items in records.items():
        nouns = {item["noun"] for item in items}
        print(f"{split}: {len(items)} samples, {len({item['scene'] for item in items})} images, "
              f"{len(nouns - set(train_nouns))}/{len(nouns)} object names not in the training split")
    print(f"distractor bank: {len(train_nouns)} training object names + {len(BACKGROUND)} background words")

    # Encode every text once.
    texts = set(BACKGROUND)
    texts.update(NOUN_TEMPLATE.format(name) for name in train_nouns)
    for items in records.values():
        for item in items:
            texts.update((item["sentence"], item["negative_sentence"]))
            texts.update(NOUN_TEMPLATE.format(name) for name in (item["noun"], item["negative_noun"], *item["scene_nouns"]))
    texts = sorted(texts)
    vectors = dict(zip(texts, encode_texts(texts, args.clip, device=device).to(device)))

    def noun_vector(name: str) -> torch.Tensor:
        return vectors[NOUN_TEMPLATE.format(name)]

    bank_names = train_nouns
    bank_vectors = torch.stack([noun_vector(name) for name in bank_names] + [vectors[word] for word in BACKGROUND])
    bank_index = {name: position for position, name in enumerate(bank_names)}

    def bank_without(name: str, extra: list[str] = ()) -> torch.Tensor:
        """The bank without the queried name, plus the given extra names."""
        keep = torch.ones(len(bank_vectors), dtype=torch.bool)
        if name in bank_index:
            keep[bank_index[name]] = False
        added = [noun_vector(other) for other in extra if other not in bank_index and other != name]
        return torch.cat([bank_vectors[keep], torch.stack(added)]) if added else bank_vectors[keep]

    results = []
    for readout, resolution in itertools.product(args.readouts, args.resolutions):
        clip = ClipDense(args.clip, resolution, readout).to(device)
        for split, items in records.items():
            dataset = data[split]
            size = dataset.image_size
            configs = [(text, score, t) for text, score in VARIANTS
                       for t in (args.temperatures if score != "abs" else [None])]
            hits = {config: np.zeros(3) for config in configs}
            cache: dict[str, torch.Tensor] = {}
            for item in items:
                if item["scene"] not in cache:
                    cache.clear()  # samples of one image are consecutive
                    cache[item["scene"]] = clip(dataset[item["index"]]["image"][None].to(device))[0]
                features = cache[item["scene"]]
                for config in configs:
                    text, score, temperature = config
                    if text == "sentence":
                        query, swapped = vectors[item["sentence"]], vectors[item["negative_sentence"]]
                    else:
                        query, swapped = noun_vector(item["noun"]), noun_vector(item["negative_noun"])
                    bank = swapped_bank = None
                    if score != "abs":
                        extra = item["scene_nouns"] if score == "scene" else []
                        bank = bank_without(item["noun"], extra)
                        # for the swapped prompt the named object becomes a distractor
                        swapped_extra = [item["noun"], *item["scene_nouns"]] if score == "scene" else []
                        swapped_bank = bank_without(item["negative_noun"], swapped_extra)
                    maps = torch.stack([
                        score_maps(features, query, bank, temperature),
                        score_maps(features, swapped, swapped_bank, temperature),
                    ])
                    maps = F.interpolate(maps[None], size=(size, size), mode="bilinear", align_corners=False)[0].cpu()
                    matching, mismatching = maps
                    inside = matching[item["region"]].mean()
                    object_hit = all(inside > matching[region].mean() for region in item["other_regions"])
                    pair_hit = inside > mismatching[item["region"]].mean()
                    peak = int(matching.flatten().argmax())
                    peak_xy = torch.tensor([peak % size, peak // size], dtype=torch.float32)
                    point_hit = (item["centres"] - peak_xy).norm(dim=1).min() < 20
                    hits[config] += np.array([bool(object_hit), bool(pair_hit), bool(point_hit)], dtype=float)
            for (text, score, temperature), counts in hits.items():
                rates = counts / len(items)
                results.append({
                    "split": split, "readout": readout, "resolution": resolution, "text": text, "score": score,
                    "temperature": temperature, "samples": len(items),
                    "object": float(rates[0]), "pair": float(rates[1]), "point": float(rates[2]),
                })
                tau = f"{temperature:.2f}" if temperature is not None else "  - "
                print(f"{split:<12}{readout:<6}{resolution:<5}{text:<9}{score:<6}tau {tau}  "
                      f"object {rates[0]:.3f}  pair {rates[1]:.3f}  point {rates[2]:.3f}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
