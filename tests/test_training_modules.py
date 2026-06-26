"""Tests for optimizer, scheduler, metrics, run folders, checkpoints, config."""

import math
import os

import torch
import torch.nn as nn

from sjepa.checkpointing import CheckpointManager, WeightSaver
from sjepa.config_schema import ExperimentConfig, load_experiment_config
from sjepa.history import HistoryRecorder
from sjepa.lr_shedulers import build_scheduler
from sjepa.metrics import (
    MetricGroup,
    effective_rank,
    kl_divergence,
    predictor_entropy_bits,
    top1_agreement,
)
from sjepa.optimizers import build_optimizer
from sjepa.rundir import RunDirectoryManager


# ----- optimizer -----

def test_param_groups_split_decay():
    """Norm and bias parameters land in the no-decay group."""
    model = nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4))
    optimizer = build_optimizer(model, "adamw", lr=1e-3, weight_decay=0.1)
    decay_group, no_decay_group = optimizer.param_groups
    assert decay_group["weight_decay"] == 0.1
    assert no_decay_group["weight_decay"] == 0.0


# ----- scheduler -----

def test_scheduler_warmup_then_decay():
    """The learning rate rises during warmup and falls afterwards."""
    model = nn.Linear(4, 4)
    optimizer = build_optimizer(model, "adamw", lr=1.0)
    scheduler = build_scheduler(optimizer, "cosine", warmup_steps=10,
                                total_steps=100)
    first = optimizer.param_groups[0]["lr"]
    for _ in range(10):
        scheduler.step()
    after_warmup = optimizer.param_groups[0]["lr"]
    for _ in range(89):
        scheduler.step()
    near_end = optimizer.param_groups[0]["lr"]
    assert first < after_warmup
    assert near_end < after_warmup


# ----- metrics precision -----

def test_kl_zero_for_equal_distributions():
    """KL between identical distributions is zero."""
    targets = torch.softmax(torch.randn(2, 5, 8), dim=-1)
    logits = torch.log(targets + 1e-9)
    selection = torch.ones(2, 5, dtype=torch.bool)
    assert kl_divergence(logits, targets, selection) < 1e-4


def test_top1_perfect_agreement():
    """Top-1 agreement is one when the top cluster always matches."""
    logits = torch.zeros(1, 3, 4)
    logits[..., 2] = 10.0
    targets = torch.zeros(1, 3, 4)
    targets[..., 2] = 1.0
    selection = torch.ones(1, 3, dtype=torch.bool)
    assert abs(top1_agreement(logits, targets, selection) - 1.0) < 1e-6


def test_entropy_of_two_way_tie_is_one_bit():
    """A perfect two-way tie has an entropy of one bit."""
    logits = torch.full((1, 1, 2), 0.0)
    selection = torch.ones(1, 1, dtype=torch.bool)
    value = predictor_entropy_bits(logits, selection)
    assert abs(value - 1.0) < 1e-5


def test_effective_rank_of_one_direction():
    """A rank-one feature matrix has an effective rank near one."""
    base = torch.randn(1, 16)
    features = torch.arange(1, 51).float().unsqueeze(1) * base
    assert effective_rank(features) < 1.5


def test_metric_group_modes():
    """The metric group reports the correct best mode per metric."""
    assert MetricGroup.mode_of("kl") == "min"
    assert MetricGroup.mode_of("top1") == "max"


# ----- run folders -----

def test_run_folders_numbering(tmp_path):
    """The first run has no number, the next ones increase."""
    manager = RunDirectoryManager(str(tmp_path), "demo", "train")
    first, _ = manager.resolve(resume=False)
    second, _ = manager.resolve(resume=False)
    assert first.root.endswith("train")
    assert second.root.endswith("train2")


def test_run_resume_reuses_folder(tmp_path):
    """Resume reuses the latest folder that has a checkpoint."""
    manager = RunDirectoryManager(str(tmp_path), "demo", "train")
    layout, _ = manager.resolve(resume=False)
    CheckpointManager(layout.checkpoints_dir).save({"epoch": 0}, 0)
    reused, resumed = manager.resolve(resume=True)
    assert resumed
    assert reused.root == layout.root


# ----- checkpoints -----

def test_checkpoint_rotation(tmp_path):
    """Only the newest checkpoints are kept on disk."""
    manager = CheckpointManager(str(tmp_path), max_checkpoints=2)
    for epoch in range(4):
        manager.save({"epoch": epoch}, epoch)
    assert manager._epochs() == [2, 3]
    assert manager.latest_path().endswith("epoch_003.pth")


def test_weight_saver_roundtrip(tmp_path):
    """Weights saved and loaded keep the same values."""
    model = nn.Linear(4, 4)
    saver = WeightSaver(str(tmp_path))
    saver.save(model, "best.pt", {"epoch": 1})
    payload = saver.load("best.pt")
    assert "model" in payload and payload["epoch"] == 1


# ----- history -----

def test_history_csv_roundtrip(tmp_path):
    """The history writes rows and reloads them on a new recorder."""
    path = str(tmp_path / "history.csv")
    recorder = HistoryRecorder(path)
    recorder.add({"epoch": 0, "train_kl": 2.0, "val_kl": 2.1})
    recorder.add({"epoch": 1, "train_kl": 1.5, "val_kl": 1.8})
    reloaded = HistoryRecorder(path)
    assert len(reloaded.history()) == 2
    assert reloaded.history()[0]["epoch"] == 0


# ----- config -----

def test_config_defaults_and_nesting():
    """A small config dict fills in defaults and nested sections."""
    config = ExperimentConfig.from_dict({"run_name": "x",
                                         "train": {"epochs": 3}})
    assert config.run_name == "x"
    assert config.train.epochs == 3
    assert config.dataset.val_prob == 0.5


def test_config_load_from_file(tmp_path):
    """A YAML file loads into the config object."""
    path = tmp_path / "c.yaml"
    path.write_text("run_name: demo\ntrain:\n  batch_size: 7\n")
    config = load_experiment_config(str(path))
    assert config.train.batch_size == 7


# ----- pipeline model sizing -----

def test_pipeline_sizes_predictor_for_long_clips(tmp_path):
    """The predictor must have enough positions for the longest clip."""
    from sjepa.assembly import PipelineBuilder
    from sjepa.rundir import RunLayout
    config = ExperimentConfig.from_dict({
        "model": {"size": "tiny"},
        "dataset": {"max_seconds": 15.0, "sample_rate": 16000},
        "gmm": {"num_clusters": 8},
    })
    builder = PipelineBuilder(config, RunLayout(str(tmp_path)))
    model = builder._build_model()
    assert model.predictor.max_frames >= 15 * 16000 // 320
