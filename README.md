# Language-driven Grasp Detection

Work in progress. A GR-ConvNet-style network conditioned on a frozen CLIP text embedding, trained with a
counterfactual prompt loss that penalises ignoring the instruction.

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

Scenes that appear in both `train_seen` and a test list are removed from training.

## Experiments

| Config | Text | Fusion | Counterfactual loss | Role |
|---|---|---|---|---|
| `configs/vision_only.yaml` | no | none | no | reference without language |
| `configs/concat.yaml` | CLIP | concatenation | no | simple fusion |
| `configs/film.yaml` | CLIP | FiLM | no | main baseline |
| `configs/film_cf.yaml` | CLIP | FiLM | yes | proposed method |

```bash
python scripts/train.py --config configs/film_cf.yaml      # train one experiment
python scripts/evaluate.py --run outputs/film_cf --figure 8  # test metrics and qualitative figure
bash scripts/run_experiments.sh                            # all four, then the results table
```

Any config value can be overridden on the command line, e.g. `--set train.epochs=10 train.seed=1`.

### Metrics

* **SR** (success rate): the predicted rectangle has IoU > 25% with a ground-truth rectangle whose
  angle differs by less than 30 degrees. **H** is the harmonic mean of the seen and unseen SR.
* **Ignore rate**: for images with a second annotated object, the model is queried again with an
  instruction for that object. Among the samples it solved, the ignore rate is the share where the
  new prediction still lands on the first object. A vision-only model scores 1.0. Lower is better.
* **Paired SR**: share of those images where the model is correct for the original instruction and
  moves away for the swapped one. Higher is better.

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
  text.py           frozen CLIP text embeddings, computed once and cached
  losses.py         grasp loss and counterfactual prompt loss
  metrics.py        grasp decoding, success rate, language metrics
  engine.py         training and evaluation loops
  config.py         YAML configuration with overrides
  utils/
    visualize.py    plotting helpers
scripts/            command-line entry points
splits/             official split lists
```
