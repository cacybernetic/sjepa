<div align="center">

<img src="assets/banner.png" width="640" alt="S-JEPA"/>

![](https://img.shields.io/badge/STATUS-stable-brightgreen)
![](https://img.shields.io/badge/Python-3.10-blue)
![](https://img.shields.io/badge/PyTorch-2.8.0-orange)
![](https://img.shields.io/badge/LICENSE-MIT-%2300557f)
![](https://img.shields.io/badge/latest-2026--06--25-green)

</div>

A clean reimplementation of **S-JEPA** (Soft Clustering Anchors for
Self-Supervised Speech Representation Learning). S-JEPA learns speech
representations with **no labels**: a JEPA-style encoder and predictor are
trained to match the **soft posteriors** of a Gaussian Mixture Model (GMM) at
masked frames, with a single KL divergence loss. Python is used for training;
the encoder is exported to ONNX for fast, language-agnostic inference.

**Table of Contents**

- [Description](#description)
- [Features](#features)
- [Project structure](#project-structure)
- [Installation](#installation)
  - [Quick install](#quick-install-without-cloning)
  - [Python — Linux](#python--linux)
  - [Python — Windows](#python--windows)
  - [ONNX (optional)](#onnx-optional)
- [Dataset format](#dataset-format)
- [Usage](#usage)
  - [1. Build an HDF5 dataset](#1-build-an-hdf5-dataset)
  - [2. Train](#2-train)
  - [3. Evaluate](#3-evaluate)
  - [4. Export to ONNX](#4-export-to-onnx)
  - [5. Run inference on an audio clip](#5-run-inference-on-an-audio-clip)
  - [6. Two-phase training](#6-two-phase-training)
- [Configuration files](#configuration-files)
- [To contribute](#to-contribute)
- [Licence](#licence)
- [Acknowledgments](#acknowledgments)
- [References](#references)
- [Contact](#contact)

---

## Description

Most modern speech encoders learn by predicting **hard** cluster IDs at masked
positions (HuBERT, WavLM). This collapses the natural ambiguity at sound
boundaries and forces a stop-and-restart pipeline to re-cluster the whole corpus
between iterations.

S-JEPA fixes both points in a single continuous training pass:

1. A CNN frontend turns the raw 16 kHz waveform into 20 ms frames.
2. A 6-layer Transformer encoder builds frame representations (`f_phi`).
3. A block mask hides about 65% of frames; a small predictor (`h_psi`) fills
   them in.
4. A cluster head (`g_omega`) maps frames to `K` cluster logits.
5. The training target is the **soft** posterior of a GMM, matched by KL:
   - **Phase 1**: a frozen GMM over 39-dim MFCC features (`K = 100`).
   - **Phase 2**: an online GMM over EMA-encoder features (`K = 500`), with an
     EMA target encoder and adaptive layer selection.

After training, only the encoder `f_phi` is kept; the predictor, cluster head,
and GMM are discarded.

For a friendly, step-by-step explanation of the ideas, read
[`docs/en_concepts.md`](docs/en_concepts.md) (English) or
[`docs/fr_concepts.md`](docs/fr_concepts.md) (French).

## Features

- **Single KL loss** between the GMM soft posteriors and the predictor softmax.
- **Two-phase training** as one continuous run: frozen MFCC GMM, then online
  encoder GMM with EMA target and adaptive layer selection.
- **Reads any audio** (`.wav`, `.mp3`, `.flac`, `.ogg`, ...) from a folder, a
  `.zip`, or a `.tar` archive, **recursively** and **without unpacking**.
- **Dataset cleaning** with a JSON cache (drop corrupt or empty files once).
- **HDF5 build** for fast, ready-to-train data.
- **Gradient accumulation** (with a final flush), gradient clipping, AdamW,
  warmup + cosine schedule.
- **Full checkpointing** (model, optimizer, scheduler, GMM, EMA) with rotation,
  deterministic **resume**, **best** and **last** weights.
- **Per-epoch history** CSV and train-vs-validation plots (overfitting check).
- **Geeky terminal output**: loguru logging into files plus two tqdm bars
  (epoch and step).
- **ONNX export** and a **standalone** inference script (copy-paste anywhere).
- **Ready configs** for CPU, NVIDIA CUDA, and AMD ROCm.

## Project structure

```
.
├── README.md
├── Makefile                   # install (CPU/CUDA/ROCm), test
├── pyproject.toml             # package metadata + CLI entry points
├── assets/                    # logo and banner (SVG sources + PNG renders)
├── docs/
│   ├── en_concepts.md         # beginner guide (English)
│   └── fr_concepts.md         # beginner guide (French)
├── cpu/configs/
│   ├── hdf5.yaml
│   ├── train.yaml
│   ├── eval.yaml
│   └── export.yaml
├── gpu/configs/               # same configs, device: cuda (CUDA and ROCm)
│   ├── hdf5.yaml
│   ├── train.yaml
│   ├── eval.yaml
│   └── export.yaml
├── src/
│   └── sjepa/
│       ├── model.py           # the full S-JEPA model (encoder, predictor, head)
│       ├── config.py          # SJEPAConfig (model hyperparameters)
│       ├── gmm.py             # diagonal GMM, fitter, online GMM
│       ├── gmm_builder.py     # build or load the phase GMMs
│       ├── targets.py         # phase-aware soft target builders
│       ├── lossfn.py          # KL divergence objective
│       ├── optimizers.py      # optimizer with parameter groups
│       ├── lr_shedulers.py    # warmup + cosine scheduler
│       ├── metrics/           # kl, top1 agreement, predictor entropy, effective rank
│       ├── step.py            # forward pass and loss for one batch
│       ├── trainer.py         # the epoch loop (the engine)
│       ├── assembly.py        # wire everything into a ready Trainer
│       ├── data_module.py     # train / val / test data loaders
│       ├── checkpointing.py   # save, rotate, resume; best.pt and last.pt
│       ├── rundir.py          # runs/<name>/train, train2, ... folders
│       ├── history.py         # per-epoch history CSV
│       ├── plotting.py        # train vs val history plots
│       ├── progress.py        # epoch and step tqdm bars
│       ├── summary.py         # torchinfo model summary
│       ├── config_schema.py   # YAML to dataclasses
│       ├── logging.py         # loguru setup, colors, tqdm-safe sink
│       ├── onnx_export.py     # encoder to ONNX
│       ├── dataset/
│       │   ├── sources.py     # find audio in folder / zip / tar (recursive)
│       │   ├── readers.py     # read bytes from one referenced file
│       │   ├── audio.py       # decode, mono, resample, crop
│       │   ├── features.py    # 39-dim MFCC features (Phase 1 GMM input)
│       │   ├── filtering.py   # drop bad files, JSON cache
│       │   ├── augment.py     # denoising augmentation (noise / mix)
│       │   ├── dataset.py     # AudioDataset + collate
│       │   └── hdf5.py        # build and read a ready-to-train HDF5 file
│       ├── modules/
│       │   ├── feature_extractor.py  # CNN frontend
│       │   ├── positional_encoding.py
│       │   ├── attention.py
│       │   ├── transformer.py
│       │   ├── encoder.py     # the speech encoder f_phi
│       │   ├── predictor.py   # the predictor h_psi
│       │   ├── cluster_head.py# the cluster head g_omega
│       │   ├── masking.py     # block mask + padding mask
│       │   ├── ema.py         # EMA encoder + switched decay (Phase 2)
│       │   ├── losses.py      # KL divergence loss
│       │   ├── normalization.py
│       │   └── gradient_scaling.py
│       └── entrypoints/
│           ├── train.py       # trainsjepa: training loop
│           ├── buildds.py     # buildh5ds: build an HDF5 dataset
│           ├── evaluate.py    # evalsjepa: full test-set evaluation
│           ├── exportmodel.py # exportw: ONNX export
│           └── inference.py   # runinfer: standalone ONNX inference
└── tests/                     # unit and integration tests
```

---

## Installation

### Quick install (without cloning)

You can install the package directly from GitHub using either `pip` or `uv`.
This gives you immediate access to all CLI tools (`trainsjepa`, `buildh5ds`,
`evalsjepa`, `exportw`, `runinfer`) without downloading the full repository.

**With pip** (works in any Python environment, no extra tools needed):

```bash
pip install git+https://github.com/cacybernetic/sjepa
```

**With uv** (faster, after installing `uv`):

```bash
uv pip install git+https://github.com/cacybernetic/sjepa
```

After installation, you can run the commands directly (see [Usage](#usage)) —
just make sure you have the required configuration YAML files (download them from
the [cpu/configs/](cpu/configs/) folder if needed).

> **Note for contributors**: if you plan to modify the code or contribute,
> please follow the full local installation instructions below.

### Python — Linux

**1. Install `uv` (fast Python package manager)**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**2. Clone the repository**

```bash
git clone https://github.com/mokira3d48/sjepa
cd sjepa
```

**3. Create a virtual environment with Python 3.10**

```bash
uv venv --python 3.10
source .venv/bin/activate
```

**4. Install PyTorch for your hardware, then the package**

The `Makefile` picks the right PyTorch build for your machine and installs the
project (editable), registering the command-line tools. Each target installs
both `torch` and `torchaudio` from the matching index, so the two always agree.

```bash
make install        # CPU only
make cuda_install   # NVIDIA CUDA
make rocm_install   # AMD ROCm
```

> **Important — always pick the build that matches your hardware.** `torch` and
> `torchaudio` ship per-hardware wheels (CPU, CUDA, ROCm). If you let `pip`
> install them from the default index, you may get a CUDA build on a machine
> with no GPU. You will then see this error at import time:
> ```
> OSError: libcudart.so.13: cannot open shared object file: No such file or directory
> ```
> This means `torchaudio` was built for CUDA but the CUDA runtime is missing. To
> fix it, reinstall both packages from the right index:
> ```bash
> pip uninstall -y torch torchaudio
> # CPU only (no GPU):
> pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cpu
> # NVIDIA CUDA 12.4 (check your driver with nvidia-smi):
> pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu124
> ```
> The `make install` / `make cuda_install` / `make rocm_install` targets already
> do this for you.

Then run the tests to check everything works:

```bash
make test
```

> **Note — headless server (no display):** the plotting uses the non-interactive
> "Agg" backend, so it works without a screen. To decode some audio formats you
> may also need the system codecs:
> ```bash
> sudo apt-get install libsndfile1 ffmpeg
> ```

### Python — Windows

1. Download and install Python 3.10 from [python.org](https://www.python.org/downloads/).
2. Open a command prompt inside the project folder.
3. Install `uv`:
   ```bash
   pip install uv
   ```
4. Create the virtual environment:
   ```bash
   uv venv --python 3.10
   .venv\Scripts\activate
   ```
5. Install PyTorch for your hardware first, then the package:
   ```bash
   # CPU only (no GPU):
   uv pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cpu
   # or NVIDIA CUDA 12.4 (check your driver with nvidia-smi):
   uv pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu124
   uv pip install -e .
   ```

### ONNX (optional)

Only needed if you want to export the encoder and run the standalone inference
script. Skip this section if you only train and evaluate.

```bash
uv pip install -e ".[onnx]"
```

This adds `onnx` and `onnxruntime` so `exportw` and `runinfer` can run.

---

## Dataset format

A dataset is a **folder**, a **`.zip`**, or a **`.tar`** archive that holds audio
files. Files may sit in the root or in sub-folders; they are found
**recursively** and read straight from the archive without unpacking it.

Before training, each dataset is scanned once. Bad files (corrupt, empty,
unreadable) are dropped, and the good ones are saved to a JSON cache next to the
dataset, so the next run does not scan again:

```
data/
  train.zip
  train.cache.json   <- created automatically
  test.zip
  test.cache.json
```

The validation set is a fraction (`val_prob`, default `0.5`) of the **test** set.
The final evaluation runs on the whole test set.

---

## Usage

Every tool reads one YAML config with `-c` / `--config`. Ready-made configs live
in `cpu/configs/` and `gpu/configs/` (the GPU configs work for both NVIDIA CUDA
and AMD ROCm).

| Command       | Job                                        | Example                                  |
|---------------|--------------------------------------------|------------------------------------------|
| `trainsjepa`  | Train the model                            | `trainsjepa -c cpu/configs/train.yaml`   |
| `buildh5ds`   | Build a ready-to-train HDF5 dataset        | `buildh5ds -c cpu/configs/hdf5.yaml`     |
| `evalsjepa`   | Evaluate on the full test set              | `evalsjepa -c cpu/configs/eval.yaml`     |
| `exportw`     | Export the encoder to ONNX                 | `exportw -c cpu/configs/export.yaml`     |
| `runinfer`  | Standalone ONNX inference on one clip      | `runinfer -c cpu/configs/export.yaml --audio clip.wav` |

### 1. Build an HDF5 dataset

Optional. Decode every clip once and store the waveforms (and optional augmented
copies) so training skips on-the-fly decoding.

```bash
buildh5ds -c cpu/configs/hdf5.yaml
```

Then set `dataset.use_hdf5: true` in the training config to read from the HDF5
files.

### 2. Train

```bash
trainsjepa -c cpu/configs/train.yaml
```

Each run writes into `runs/<run_name>/train` (then `train2`, `train3`, ...):

```
runs/sjepa_base/train/
  history.csv          # train vs val metrics per epoch
  config_used.yaml     # the exact config used
  weights/
    best.pt            # best validation score
    last.pt            # last epoch
  checkpoints/
    epoch_000.pth      # full state (model, optimizer, scheduler, GMM, EMA)
  plotes/
    history_kl.jpg     # train vs val curves (overfitting check)
  logs/
    train_2026-06-25_19-55-06.log
```

To continue an interrupted run, set `checkpoint.resume: true`. When a usable
checkpoint exists, training reuses the highest-numbered run folder and continues
from the last checkpoint.

### 3. Evaluate

Point `init_weights` at the weight file to evaluate, then run:

```bash
evalsjepa -c cpu/configs/eval.yaml
```

The metrics are printed and written to `runs/<run_name>/eval/results.csv`.

### 4. Export to ONNX

```bash
exportw -c cpu/configs/export.yaml
```

Only the encoder is exported. The output path is the `onnx_path` field in the
config.

### 5. Run inference on an audio clip

`runinfer` is fully self-contained: it imports only `numpy`, `soundfile`,
`onnxruntime`, and `pyyaml`, so you can copy it into another project.

```bash
runinfer -c cpu/configs/export.yaml --audio data/sample.wav
```

It loads the ONNX encoder, reads the clip as mono 16 kHz audio, and prints the
frame features. These features are what you feed to a small task head (speech
recognition, emotion, ...).

### 6. Two-phase training

The paper runs both phases as **one continuous trajectory**. The trainer does
this in a single run: start in Phase 1 (frozen MFCC GMM) and let it switch to
Phase 2 (online encoder GMM, `K = 500`) at a chosen epoch.

```yaml
train:
  phase: 1
  phase2_start_epoch: 50   # switch to the online encoder GMM mid-run
  masked_only_epoch: 75    # then drop the visible loss + turn augmentation off
gmm:
  num_clusters: 100        # K in Phase 1
  num_clusters_phase2: 500 # K after the transition
  auto_layer: true         # pick the GMM input layer by effective rank
```

At `phase2_start_epoch` the cluster head is rebuilt for `K = 500` (its optimizer
state swapped in place while the encoder/predictor moments are kept), an EMA
target encoder is created from the current encoder, and an online GMM is seeded
over the EMA features at the active layer. The active layer is then tracked by
effective rank. At `masked_only_epoch` the loss becomes masked-only and the
denoising augmentation is turned off, matching the paper's Phase 2 transition.

The learning rate is **warm-restarted at the transition** (`scheduler.rewarm_on_phase2`,
on by default): a single whole-run cosine would otherwise leave Phase 2 — the
phase that does the heavy lifting — training on its decayed tail near zero. With
the restart, the LR warms back up at `phase2_start_epoch` and decays again over
the Phase 2 epochs down to `scheduler.min_ratio` of the peak (keep `min_ratio`
above 0, e.g. `0.1`, so it never reaches zero).

> **Order the two switches correctly.** The paper turns the loss masked-only
> *partway through Phase 2*, so set `masked_only_epoch` **after**
> `phase2_start_epoch` (e.g. transition at 50, masked-only at 75). If
> `masked_only_epoch` lands before the transition, the visible loss is dropped
> while still in Phase 1, which is not the intended schedule. Use `-1` to
> disable either switch.

> **`ema_layer` must be a valid layer index.** It is the encoder layer the
> online GMM reads. A `tiny` model has only 2 layers (indices `0, 1`), so use
> `ema_layer: 1`, not `2`. With `auto_layer: true` the active layer is then
> re-selected automatically by effective rank.

Prefer two separate runs instead? Leave `phase2_start_epoch: -1` and start a
fresh run directly in Phase 2:

```yaml
train:
  phase: 2
gmm:
  online: true
  num_clusters: 500
init_weights: runs/sjepa_base/train/weights/best.pt
```

A ready single-run example lives in `cpu/configs/train_twophase.yaml`.

#### Reading the training curves

The two phases optimize **different targets** (MFCC GMM with `K = 100` in
Phase 1, encoder GMM with `K = 500` in Phase 2), so the KL is **not directly
comparable across the transition** — judge each phase by its own trend. A
healthy run looks like this:

| Stage | `val_kl` | `val_top1` | `val_entropy_bits` |
|-------|----------|------------|--------------------|
| Phase 1 | falls, then **plateaus** | low, flat | mid |
| Transition epoch | **spikes up** (new head + new `K`) | jumps | `≈ log2(K)` (uniform) |
| Phase 2 | **falls back below the Phase 1 plateau** | climbs | **decreases** |

What to watch for:

- **Phase 1 plateau is expected** — it is exactly the ceiling Phase 2 exists to
  break. Schedule `phase2_start_epoch` once `val_kl` flattens.
- **The spike at the transition epoch is normal**: the `K = 500` cluster head is
  freshly initialized and the targets change, so the predictor starts near a
  uniform distribution (`entropy_bits ≈ log2(K)`). It should recover within a
  few epochs.
- **Healthy Phase 2** = `val_kl` trending down past the Phase 1 best, `val_top1`
  rising, and `val_entropy_bits` decreasing while staying **well above 0**.
  Entropy collapsing toward 0 (one cluster) or `val_kl` frozen would signal a
  representational collapse — the online GMM re-seeds dead components to avoid
  this.
- **Give Phase 2 a generous epoch budget.** Phase 2 keeps improving even after
  the learning rate has reached its floor (`min_ratio`): the EMA encoder and the
  online GMM co-evolve with the encoder, so the targets keep sharpening and
  `val_kl` keeps falling on a low, steady rate. In practice it is still
  descending long after the transition — if `val_kl` is still going down at the
  last epoch, the run stopped early. Schedule the transition once Phase 1
  flattens and leave Phase 2 the larger share of epochs.

The metrics are logged each epoch and written to `history.csv` with matching
plots under `<run>/plotes/` (`history_kl.jpg`, `history_top1.jpg`,
`history_entropy_bits.jpg`, ...).

---

## Configuration files

| File                       | Used by      | Key fields                                            |
|----------------------------|--------------|-------------------------------------------------------|
| `cpu/configs/train.yaml`   | `trainsjepa` | `dataset`, `model.size`, `train`, `optimizer`, `gmm`  |
| `cpu/configs/train_twophase.yaml` | `trainsjepa` | single-run Phase 1 -> Phase 2 demo (`phase2_start_epoch`) |
| `cpu/configs/hdf5.yaml`    | `buildh5ds`  | `dataset.train_path`, `dataset.train_h5`, `augment`   |
| `cpu/configs/eval.yaml`    | `evalsjepa`  | `init_weights`, `dataset.test_path`, `gmm.num_clusters` |
| `cpu/configs/export.yaml`  | `exportw` / `runinfer` | `init_weights`, `onnx_path`, `audio`      |

The same files exist under `gpu/configs/` with `device: cuda` (used for both
NVIDIA CUDA and AMD ROCm), sized for a full-scale run (`model.size: base`, the
whole corpus, the paper's epoch budget). The `cpu/configs/` are for quick local
experiments on a CPU-only machine (`tiny`/`small`, capped `max_train_samples`).
The two sets are kept in sync: any change to a `cpu/` config is mirrored to its
`gpu/` counterpart. A few important keys:

```yaml
train:
  epochs: 10
  batch_size: 8
  grad_accum: 4            # effective batch = batch_size x grad_accum
  phase: 1                 # 1 = MFCC GMM, 2 = online encoder GMM
  use_visible_loss: true   # add the visible-frame KL (Phase 1 / early Phase 2)
  phase2_start_epoch: -1   # epoch to switch to Phase 2 in one run (-1 = off)
  masked_only_epoch: -1    # epoch to drop visible loss + augmentation (-1 = off)

gmm:
  num_clusters: 100        # K in Phase 1
  num_clusters_phase2: 500 # K after the in-run Phase 2 transition
  online: false            # true to start a run directly in Phase 2
  ema_layer: 2             # initial encoder layer used by the online GMM
  auto_layer: true         # pick the layer by effective rank
  erank_decay: 0.9         # smoothing of the per-check effective-rank score

scheduler:
  kind: cosine             # cosine | constant
  warmup_steps: 5000
  min_ratio: 0.1           # LR floor (fraction of peak); keep > 0 for Phase 2
  rewarm_on_phase2: true   # warm-restart the LR at phase2_start_epoch

best:
  metric: kl               # which metric chooses best.pt (kl, top1, entropy_bits)
```

---

## To contribute

Contributions are welcome! Please follow these steps:

1. Fork the repository and clone it locally.
2. Create a new branch for your feature: `git checkout -b feature/my-feature`
3. Commit your changes: `git commit -m 'Add a new feature'`
4. Push to the branch: `git push origin feature/my-feature`
5. Open a Pull Request.

## Licence

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file
for details.

## Acknowledgments

This project was built while studying the inner workings of S-JEPA. A big
thank-you to **Georgios Ioannides** and the co-authors of the S-JEPA paper, and
to the reference implementation
[**gioannides/s-jepa**](https://github.com/gioannides/s-jepa), which served as
the primary reference for the training recipe (soft GMM targets, online updates,
switched EMA decay, and adaptive layer selection).

If you find this project useful, please consider giving the original **s-jepa**
repository a star as a token of appreciation for the work that made it possible.

## References

The implementation is based on the following papers and resources:

### Method and objective

- **S-JEPA** — Ioannides, G., Kieback, A., Goldfeder, J., Pang, L., Chadha, A.,
  Elkins, A., LeCun, Y., & Shwartz-Ziv, R. (2026). *S-JEPA: Soft Clustering
  Anchors for Self-Supervised Speech Representation Learning*. The paper this
  repository reimplements (see `papers/sources/arXiv-2606.19398v1/`).
- **JEPA** — LeCun, Y. (2022). *A Path Towards Autonomous Machine Intelligence*.
  The encoder-predictor pattern with a learned mask token.
- **I-JEPA** — Assran, M., et al. (2023). *Self-Supervised Learning from Images
  with a Joint-Embedding Predictive Architecture*. CVPR 2023.
  [arXiv:2301.08243](https://arxiv.org/abs/2301.08243)

### Clustering and soft targets

- **GMM / EM** — Dempster, A. P., Laird, N. M., & Rubin, D. B. (1977).
  *Maximum Likelihood from Incomplete Data via the EM Algorithm*. JRSS B.
- **Reservoir sampling** — Vitter, J. S. (1985). *Random Sampling with a
  Reservoir*. ACM TOMS.
- **HuBERT** — Hsu, W.-N., et al. (2021). *HuBERT: Self-Supervised Speech
  Representation Learning by Masked Prediction of Hidden Units*. The hard-label
  recipe S-JEPA softens.
  [arXiv:2106.07447](https://arxiv.org/abs/2106.07447)

### Architecture and training

- **wav2vec 2.0** — Baevski, A., Zhou, H., Mohamed, A., & Auli, M. (2020).
  *wav2vec 2.0: A Framework for Self-Supervised Learning of Speech
  Representations*. NeurIPS 2020. CNN frontend and masking.
  [arXiv:2006.11477](https://arxiv.org/abs/2006.11477)
- **data2vec** — Baevski, A., et al. (2022). *data2vec: A General Framework for
  Self-Supervised Learning*. ICML 2022. EMA target encoder.
  [arXiv:2202.03555](https://arxiv.org/abs/2202.03555)
- **Effective rank / RankMe** — Garrido, Q., et al. (2023). *RankMe: Assessing
  the Downstream Performance of Pretrained Self-Supervised Representations by
  their Rank*. ICML 2023. Label-free layer selection signal.
  [arXiv:2210.02885](https://arxiv.org/abs/2210.02885)

### Educational reference

- **gioannides/s-jepa** — Ioannides, G. (2026). Reference implementation
  of the S-JEPA training recipe.

## Contact

For questions or suggestions:

- **Author**: Dr Mokira — arnoldmokira3d48@gmail.com
- **Maintainer**: Dr Mokira — arnoldmokira3d48@gmail.com
- **GitHub**: [mokira3d48/sjepa](https://github.com/cacybernetic/sjepa)
