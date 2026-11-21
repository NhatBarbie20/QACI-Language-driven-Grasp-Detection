# Language-driven Grasp Detection

A language-driven grasp detector for [Grasp-Anything++](https://huggingface.co/datasets/airvlab/Grasp-Anything-pp)
that writes the grasp quality of a pixel as a product of two maps:

```
Q(p | image, instruction) = G(p | image) * S(p | image, instruction)
```

* **G (geometry, "how")** is a small GR-ConvNet-style network (1.9M parameters). It never sees the
  instruction, so it is trained on the grasps of *every* annotated object of the image.
* **S (grounding, "where")** is the patch-text similarity of a frozen CLIP ViT-B/16, sharpened by a
  small convolutional refinement that sees image features but never the text embedding. The
  vocabulary of the detector is therefore inherited from CLIP instead of being learned from the
  129 object names of the training split.

Networks that fuse the text embedding into the visual features lose about a quarter of their
success rate on object names that were not seen in training. The factorised model does not.

## Results

Success rate (SR) and paired success rate (PSR, see [Metrics](#metrics)) on the test splits, mean
and standard deviation over runs with different seeds. 30 epochs, 224x224 images, one T4 GPU.

| Model | Runs | SR seen | SR unseen | PSR seen | PSR unseen |
|---|---|---|---|---|---|
| Vision only (`vision_only`) | 2 | 0.317 ± 0.011 | 0.314 ± 0.006 | 0 | 0 |
| Concat fusion (`concat`) | 3 | 0.543 ± 0.007 | 0.343 ± 0.035 | 0.438 ± 0.024 | 0.191 ± 0.029 |
| FiLM fusion (`film`) | 4 | 0.527 ± 0.017 | 0.384 ± 0.043 | 0.432 ± 0.007 | 0.223 ± 0.025 |
| **Factorised G x S (`factor_all`)** | 3 | **0.545 ± 0.017** | **0.540 ± 0.045** | **0.447 ± 0.012** | **0.331 ± 0.038** |

Diagnostics and variants of the factorised model:

| Variant | Runs | SR seen | SR unseen |
|---|---|---|---|
| G alone, grounding dropped at test time (`--grounding none`) | 3 | 0.390 ± 0.025 | 0.430 ± 0.047 |
| Ground-truth grounding (`--grounding oracle`) | 3 | 0.773 ± 0.017 | 0.680 ± 0.048 |
| S without the refinement (`factor_raw`) | 1 | 0.437 | 0.399 |
| G supervised with the target grasps only (`factor`) | 1 | 0.379 | 0.377 |
| + counterfactual prompt loss (`factor_full`) | 3 | 0.503 ± 0.032 | 0.548 ± 0.044 |
| + grounding loss (`factor_ground`) | 3 | 0.558 ± 0.017 | 0.528 ± 0.033 |

Things that did not help, kept in the code for reference: the counterfactual prompt loss
(`loss.cf_weight`), which makes the FiLM network collapse on unseen objects (SR 0.21, one run) and
has no reliable effect on the factorised model; a direct loss on S (`loss.ground_weight`); and a
noun-only prompt (`model.prompt=noun`). The unseen SR varies by about 0.04 between seeds, so
smaller differences are not conclusive.

The factorised model additionally runs the frozen CLIP image encoder (86M parameters, not trained);
the baselines use only the CLIP text encoder.

### Label noise in Grasp-Anything++

12.8% of the object pairs of an image have byte-identical grasp labels, and a further 15.4% nearly
coincide. A prompt-swap test counts a model that follows the instruction as ignoring it on such a
pair. `data.max_negative_overlap` (and `evaluate.py --max-negative-overlap 0.1`) restricts the swap
to objects whose grasp region overlaps the target by at most that fraction. This keeps 793 of 996
swap pairs on the seen split and 257 of 467 on the unseen split. The PSR above is computed on these
pairs.

## Setup

```bash
pip install -r requirements.txt
```

## Data

The experiments use [Grasp-Anything++](https://huggingface.co/datasets/airvlab/Grasp-Anything-pp)
with the official splits of the [LGD](https://github.com/Fsoft-AIC/LGD) code release
(`splits/grasp_anything_pp/`, converted from pickle to JSON).

The full dataset is about 80 GB. The splits reference only ~12k of its 1M scenes, so the
download script reads those scenes directly out of the remote archives with HTTP range
requests (about 1 GB on disk, a few minutes) and checks the CRC-32 of every file.

```bash
python scripts/download_data.py        # download the scenes used by the splits
python scripts/verify_data.py          # check the data, write manifest.json and verify_report.json
python scripts/encode_text.py          # encode all instructions once with frozen CLIP
python scripts/visualize_samples.py    # optional: save a grid of samples with ground-truth grasps
```

`download_data.py` can be re-run at any time: it only fetches files that are missing.

### Splits

| Split | Source | Use |
|---|---|---|
| `train` | official `train_seen`, 95% of scenes | training |
| `val` | official `train_seen`, 5% of scenes (fixed by scene hash) | model selection |
| `test_seen` | official `test/seen` | evaluation |
| `test_unseen` | official `test/unseen` | evaluation |

Scenes that appear in both `train_seen` and a test list are removed from training. Only objects
with a language annotation in the public release are used: 26,820 training, 1,363 seen and 597
unseen test samples. The 25 object names of the unseen split do not occur in training.

## Experiments

| Config | Model | Role |
|---|---|---|
| `configs/vision_only.yaml` | geometry network, no text | reference without language |
| `configs/concat.yaml` | text embedding concatenated to the features | simple fusion |
| `configs/film.yaml` | text embedding through FiLM in every residual block | main baseline |
| `configs/factor_all.yaml` | factorised G x S, G trained on all objects | **proposed model** |
| `configs/factor.yaml`, `configs/factor_raw.yaml` | ablations of the factorised model | design choices |
| `configs/film_cf.yaml`, `configs/factor_full.yaml`, `configs/factor_ground.yaml` | extra losses | negative results |

```bash
python scripts/train.py --config configs/factor_all.yaml                        # train one experiment
python scripts/evaluate.py --run outputs/factor_all --figure 8                  # test metrics, qualitative figure
python scripts/evaluate.py --run outputs/factor_all --max-negative-overlap 0.1  # PSR on label-disjoint pairs
python scripts/evaluate.py --run outputs/factor_all --grounding none            # geometry factor alone
python scripts/evaluate.py --run outputs/factor_all --grounding oracle          # ground-truth grounding
bash scripts/run_experiments.sh                                                 # baselines and proposed model
```

Any config value can be overridden on the command line, e.g. `--set train.epochs=10 train.seed=1`.
The seeds of the table are 0, 1 and 2. Two runs with the same seed on different GPUs do not give
identical numbers, which is why FiLM has four runs.

`scripts/probe_grounding.py` and `scripts/probe_grounding_sources.py` measure, without any
training, how well different read-outs of frozen CLIP rank the region of the named object above
the other objects of the image.

### Metrics

* **SR** (success rate): the predicted rectangle has IoU > 25% with a ground-truth rectangle whose
  angle differs by less than 30 degrees.
* **PSR** (paired success rate): for images with a second annotated object, the model is queried
  again with an instruction for that object. A sample counts only if the prediction is correct for
  the original instruction *and* no longer matches the labels of the first object for the swapped
  one. A model that ignores language scores 0.
* **Ignore rate**: among the samples the model solved, the share where the prediction for the
  swapped instruction still lands on the first object. A vision-only model scores 1.

### Running on Kaggle (GPU T4)

Enable *Internet* and *GPU T4* in the notebook settings, then:

```bash
!git clone https://github.com/NhatBarbie20/QACI-Language-driven-Grasp-Detection.git
%cd QACI-Language-driven-Grasp-Detection
!pip install -q -r requirements.txt
!python scripts/download_data.py
!python scripts/verify_data.py
!python scripts/encode_text.py
!bash scripts/run_experiments.sh
```

## Code layout

```
configs/            default.yaml and one small file per experiment
lgrasp/
  data/
    remote_zip.py   read members of a remote ZIP archive with range requests
    download.py     download the dataset subset
    splits.py       official splits and file naming
    io.py           safe readers for the raw pickle / torch files
    grasp.py        grasp rectangle geometry and dense target maps
    manifest.py     build the sample index and assign splits
    verify.py       dataset integrity and consistency checks
    dataset.py      PyTorch dataset
  models/
    grasp_net.py    GR-ConvNet-style network with none / concat / FiLM text fusion
    clip_dense.py   patch features of a frozen CLIP image encoder
    factor_net.py   factorised model: geometry network x CLIP grounding map
  text.py           frozen CLIP text embeddings, computed once and cached
  losses.py         grasp loss, counterfactual prompt loss, grounding loss
  metrics.py        grasp decoding, success rate, language metrics
  engine.py         training and evaluation loops
  config.py         YAML configuration with overrides
  utils/
    visualize.py    plotting helpers
scripts/            command-line entry points
splits/             official split lists
```
