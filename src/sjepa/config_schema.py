"""Load the YAML training and evaluation configuration into dataclasses.

Keeping every setting in one typed object makes the rest of the code clear and
catches typos early. Each section of the YAML file maps to one small dataclass.
Missing keys fall back to safe defaults, so a short config still works.

The top object is `ExperimentConfig`. Use `load_experiment_config(path)` to read
a YAML file into it, and `config.to_dict()` to write the used config back out.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields

import yaml


def _section(data, key):
    """Return a sub-dict for a section, or an empty dict when missing."""
    value = data.get(key) if data else None
    return value if isinstance(value, dict) else {}


def _keep_known(cls, data):
    """Drop unknown keys so an extra YAML field does not crash the load."""
    names = {f.name for f in fields(cls)}
    return {key: value for key, value in data.items() if key in names}


@dataclass
class AugmentConfig:
    """Denoising augmentation settings."""

    enabled: bool = True
    p_noise: float = 0.25
    p_mix: float = 0.25
    snr_noise: tuple = (-5.0, 20.0)
    ratio_mix: tuple = (-5.0, 5.0)

    @classmethod
    def from_dict(cls, data):
        return cls(**_keep_known(cls, data))


@dataclass
class DatasetConfig:
    """Where the data lives and how to read it."""

    train_path: str = ""
    test_path: str = ""
    train_h5: str = "data/train.h5"
    test_h5: str = "data/test.h5"
    use_hdf5: bool = False
    validate: bool = True
    max_train_samples: object = None
    max_test_samples: object = None
    val_prob: float = 0.5
    # Keep the final test evaluation disjoint from the validation subset, so
    # the clips used to pick the best checkpoint never inflate the test score.
    val_disjoint: bool = True
    sample_rate: int = 16000
    max_seconds: float = 15.0
    # Fraction of overlap between consecutive windows tiled across each clip
    # (window length = max_seconds; hop = max_seconds * (1 - window_overlap)).
    # 0.5 = 50% overlap, 0.0 = back-to-back windows, capped below 1.0.
    window_overlap: float = 0.5
    num_workers: int = 4
    augment: AugmentConfig = field(default_factory=AugmentConfig)

    @classmethod
    def from_dict(cls, data):
        known = _keep_known(cls, data)
        known["augment"] = AugmentConfig.from_dict(_section(data, "augment"))
        return cls(**known)


@dataclass
class ModelConfig:
    """Model size and optional overrides."""

    size: str = "base"
    overrides: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data):
        known = _keep_known(cls, data)
        extra = {k: v for k, v in (data or {}).items()
                 if k not in ("size", "overrides")}
        known.setdefault("overrides", {})
        known["overrides"].update(_section(data, "overrides"))
        known["overrides"].update(extra)
        return cls(**known)


@dataclass
class MaskingConfig:
    """Block masking settings."""

    mask_ratio: float = 0.65
    mask_length: int = 10

    @classmethod
    def from_dict(cls, data):
        return cls(**_keep_known(cls, data))


@dataclass
class TrainLoopConfig:
    """Core training loop settings."""

    epochs: int = 10
    batch_size: int = 16
    grad_accum: int = 1
    grad_clip_norm: float = 1.0
    log_every: int = 16
    phase: int = 1
    use_visible_loss: bool = True
    # "fp32" or "bf16": bf16 runs the model forward under torch.autocast on
    # CUDA (the loss and the GMM targets stay in float32). No GradScaler is
    # needed for bf16.
    precision: str = "fp32"
    # Epoch at which a Phase 1 run switches to Phase 2 in the same trajectory
    # (K -> num_clusters_phase2, EMA encoder + online GMM). -1 disables it.
    phase2_start_epoch: int = -1
    # Epoch at which the loss becomes masked-only and augmentation is turned off
    # (the paper does this partway through Phase 2). -1 disables it.
    masked_only_epoch: int = -1

    @classmethod
    def from_dict(cls, data):
        return cls(**_keep_known(cls, data))


@dataclass
class OptimizerConfig:
    """Optimizer settings."""

    name: str = "adamw"
    lr: float = 1e-4
    weight_decay: float = 1e-3
    betas: tuple = (0.9, 0.99)

    @classmethod
    def from_dict(cls, data):
        return cls(**_keep_known(cls, data))


@dataclass
class SchedulerConfig:
    """Learning rate scheduler settings."""

    kind: str = "cosine"
    warmup_steps: int = 0
    min_ratio: float = 0.1
    # Warm-restart the learning rate at the Phase 1 -> Phase 2 transition so
    # Phase 2 does not train on the decayed tail of the whole-run cosine.
    rewarm_on_phase2: bool = True
    # Scale of the restarted Phase 2 window relative to the base rate. The
    # paper trains Phase 2 at 2.5e-5 against 1e-4 in Phase 1 (ratio 0.25);
    # restarting at the full base rate destabilizes the self-referential
    # Phase 2 targets.
    phase2_lr_ratio: float = 0.25

    @classmethod
    def from_dict(cls, data):
        return cls(**_keep_known(cls, data))


@dataclass
class GmmConfig:
    """GMM settings for both phases."""

    num_clusters: int = 100
    # K used once the run reaches Phase 2 (paper uses 500).
    num_clusters_phase2: int = 500
    kmeans_iters: int = 5
    em_iters: int = 20
    fit_frames: int = 50000
    path: object = None
    online: bool = False
    param_decay: float = 0.999
    ema_layer: int = 2
    auto_layer: bool = True
    layer_check_every: int = 10000
    # Smoothing of the per-layer effective-rank score, applied once per layer
    # check (not per step). Kept separate from the slow per-step `param_decay`
    # so the adaptive layer switch reacts on the right timescale.
    erank_decay: float = 0.9
    ema_decay_fast: float = 0.999
    ema_decay_slow: float = 0.9999
    ema_switch_every: int = 20000

    @classmethod
    def from_dict(cls, data):
        return cls(**_keep_known(cls, data))


@dataclass
class CheckpointConfig:
    """Checkpoint and resume settings."""

    dir: object = None
    max_checkpoint: int = 5
    resume: bool = False
    # Save a full in-epoch checkpoint every `ckpt_step` optimizer steps (train)
    # or processed batches (val/test). 0 = end-of-epoch checkpoints only.
    ckpt_step: int = 0

    @classmethod
    def from_dict(cls, data):
        return cls(**_keep_known(cls, data))


@dataclass
class BestConfig:
    """Best-model selection settings."""

    metric: str = "kl"

    @classmethod
    def from_dict(cls, data):
        return cls(**_keep_known(cls, data))


@dataclass
class ExperimentConfig:
    """The whole experiment configuration."""

    run_name: str = "sjepa"
    runs_root: str = "runs"
    device: str = "cpu"
    seed: int = 0
    init_weights: object = None
    onnx_path: str = "model.onnx"
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    masking: MaskingConfig = field(default_factory=MaskingConfig)
    train: TrainLoopConfig = field(default_factory=TrainLoopConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    gmm: GmmConfig = field(default_factory=GmmConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    best: BestConfig = field(default_factory=BestConfig)

    @classmethod
    def from_dict(cls, data):
        """Build the full config from a parsed YAML dict."""
        data = data or {}
        top = {k: data[k] for k in ("run_name", "runs_root", "device", "seed",
                                    "init_weights", "onnx_path") if k in data}
        return cls(
            dataset=DatasetConfig.from_dict(_section(data, "dataset")),
            model=ModelConfig.from_dict(_section(data, "model")),
            masking=MaskingConfig.from_dict(_section(data, "masking")),
            train=TrainLoopConfig.from_dict(_section(data, "train")),
            optimizer=OptimizerConfig.from_dict(_section(data, "optimizer")),
            scheduler=SchedulerConfig.from_dict(_section(data, "scheduler")),
            gmm=GmmConfig.from_dict(_section(data, "gmm")),
            checkpoint=CheckpointConfig.from_dict(_section(data, "checkpoint")),
            best=BestConfig.from_dict(_section(data, "best")),
            **top,
        )

    def to_dict(self):
        """Return a plain dict, used to write config_used.yaml."""
        return asdict(self)


def load_experiment_config(path):
    """Read a YAML file into an `ExperimentConfig`."""
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return ExperimentConfig.from_dict(data)


def save_used_config(config, path):
    """Write the used config to a YAML file for full traceability."""
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config.to_dict(), handle, sort_keys=False)
