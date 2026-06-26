# Changelog


The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added

- S-JEPA model implementation: the encoder `f_phi`, the predictor `h_psi`, and
  the cluster head `g_omega`.
- Diagonal-covariance GMM with a k-means + EM fitter and an online GMM, plus
  39-dimensional MFCC features for the Phase 1 targets.
- Soft-target training objective: a single KL divergence between the GMM
  posteriors and the predictor softmax at the selected frames.
- Block masking, EMA encoder with a periodically switched decay schedule, and
  effective-rank layer selection.
- Training engine with gradient accumulation, gradient clipping, a
  warmup-plus-cosine scheduler, checkpoint management, and resume.
- Best model saving and progress bars on training, validation, and evaluation.
- YAML configuration schema, data module, and an HDF5 dataset builder.
- Console entry points: `trainsjepa`, `evalsjepa`, `buildh5ds`, `exportw`, and
  `runs` (ONNX inference).
- Model summary in the train, evaluate, and export programs.
- ONNX model export plus preprocessing and postprocessing with numpy.
- Ready configuration files for CPU and for GPU (NVIDIA CUDA and AMD ROCm).
- Evaluation metrics building and plotting of the training/validation history.
- Logging system based on loguru.
- Unit tests and performance tests for the model and the training modules.
- Documentation: a detailed `README.md` and bilingual (English and French)
  concept guides under `docs/`.
- Single-run two-phase training (`PhaseScheduler`): a run starts in Phase 1 and
  switches to Phase 2 in the same trajectory at `train.phase2_start_epoch`.
- Phase 2 switch rebuilds the cluster head for `K = num_clusters_phase2`, swaps
  it into the optimizer while keeping the encoder and predictor moments, and
  installs the EMA encoder with the online GMM.
- Masked-only switch at `train.masked_only_epoch` (drops the visible loss and
  turns the denoising augmentation off).
- Learning-rate warm restart at the Phase 2 transition (`scheduler.rewarm_on_phase2`).
- Online GMM dead-cluster re-seeding to avoid vocabulary collapse.
- Train-side metrics (kl, top1, entropy_bits) so the plots overlay the train and
  the validation curve for every metric.
- Fast resume of a Phase 2 checkpoint (`phase2_scaffold`) without refitting the GMM.
- New configuration keys: `train.phase2_start_epoch`, `train.masked_only_epoch`,
  `gmm.num_clusters_phase2`, `gmm.erank_decay`, `scheduler.rewarm_on_phase2`.
- Example single-run two-phase config: `cpu/configs/train_twophase.yaml`.
- Packaging metadata in `pyproject.toml` (keywords and classifiers).
- Community files: `CONTRIBUTING.md` and `CODE_OF_CONDUCT.md`.


### Changed

- Validation averages the full objective loss (same definition as training) so
  the loss plot overlays comparable curves.
- Phase 2 online GMM is updated from every micro-batch of a gradient-accumulation
  window, not only the last one.
- Layer selection initializes its scores lazily and uses a dedicated
  `gmm.erank_decay`, decoupled from `gmm.param_decay`.
- Online GMM variance floor raised from 1e-6 to 1e-4.
- Defaults updated: `gmm.auto_layer` is now true and `scheduler.min_ratio` is now 0.1.
- History plotter is robust to non-numeric or empty cells.

### Fixed

- Training plateau: the Phase 2 machinery (online GMM, switched EMA, adaptive
  layer selection) is now actually engaged, and the learning rate no longer
  decays to near zero during Phase 2.
- Resume from a Phase 2 checkpoint no longer behaves like a return to Phase 1: it
  rebuilds the scaffolding only, restores the masked-only state, and continues
  from the saved epoch.


### Deprecated
<!-- - The old config key X is deprecated and will be removed in version 1.0. -->


### Security
<!-- - Updated library Y to fix a security issue. -->
