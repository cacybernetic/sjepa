"""End-to-end pipeline test on a tiny synthetic dataset.

This builds a small dataset, runs one short training run, and checks that the
expected output files appear. It exercises the data module, the target builder,
the forward step, the trainer, checkpointing, and the history plots together.
"""

import io
import os
import zipfile

import numpy as np
import soundfile as sf
import torch
import yaml

from sjepa.entrypoints import train as train_entry


def _sine_zip(path, count, base=200):
    """Write a few short sine clips into a zip archive."""
    with zipfile.ZipFile(path, "w") as archive:
        for index in range(count):
            time = np.linspace(0, 0.6, int(16000 * 0.6), endpoint=False)
            wave = (0.2 * np.sin(2 * np.pi * (base + 20 * index) * time))
            buffer = io.BytesIO()
            sf.write(buffer, wave.astype("float32"), 16000, format="WAV")
            archive.writestr(f"clip_{index}.wav", buffer.getvalue())


def _tiny_config(data_dir):
    """Build a tiny training config dict for a fast CPU run."""
    return {
        "run_name": "pytest", "runs_root": "runs", "device": "cpu", "seed": 0,
        "dataset": {
            "use_hdf5": False, "validate": False,
            "train_path": os.path.join(data_dir, "train.zip"),
            "test_path": os.path.join(data_dir, "test.zip"),
            "val_prob": 0.5, "sample_rate": 16000, "max_seconds": 1.0,
            "num_workers": 0, "augment": {"enabled": False},
        },
        "model": {"size": "tiny"},
        "masking": {"mask_ratio": 0.6, "mask_length": 4},
        "train": {"epochs": 1, "batch_size": 3, "grad_accum": 2,
                  "log_every": 1, "phase": 1, "use_visible_loss": True},
        "optimizer": {"name": "adamw", "lr": 5.0e-4},
        "scheduler": {"kind": "cosine", "warmup_steps": 1},
        "gmm": {"num_clusters": 12, "kmeans_iters": 2, "em_iters": 3,
                "fit_frames": 1000},
        "checkpoint": {"max_checkpoint": 2, "resume": False},
        "best": {"metric": "kl"},
    }


def test_training_pipeline_outputs(tmp_path, monkeypatch):
    """A short run must create checkpoints, weights, history, and plots."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _sine_zip(str(data_dir / "train.zip"), 9)
    _sine_zip(str(data_dir / "test.zip"), 4, base=300)
    config = _tiny_config(str(data_dir))
    config_path = tmp_path / "train.yaml"
    config_path.write_text(yaml.safe_dump(config))
    monkeypatch.chdir(tmp_path)
    train_entry.run(str(config_path))
    run_dir = tmp_path / "runs" / "pytest" / "train"
    assert (run_dir / "weights" / "best.pt").exists()
    assert (run_dir / "weights" / "last.pt").exists()
    assert (run_dir / "history.csv").exists()
    assert any((run_dir / "checkpoints").glob("ckpt_e000_*.pth"))
    assert any((run_dir / "plotes").iterdir())


def test_in_epoch_checkpoint_and_resume(tmp_path, monkeypatch):
    """A small ckpt_step writes mid-epoch checkpoints and a resume finishes."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _sine_zip(str(data_dir / "train.zip"), 12)
    _sine_zip(str(data_dir / "test.zip"), 4, base=300)
    config = _tiny_config(str(data_dir))
    config["train"]["epochs"] = 2
    config["train"]["grad_accum"] = 1
    config["checkpoint"] = {"max_checkpoint": 20, "resume": False, "ckpt_step": 1}
    config_path = tmp_path / "train.yaml"
    config_path.write_text(yaml.safe_dump(config))
    monkeypatch.chdir(tmp_path)

    train_entry.run(str(config_path))
    ckpt_dir = tmp_path / "runs" / "pytest" / "train" / "checkpoints"

    # Mid-epoch ("train") checkpoints were written with the full resumable state.
    stages = []
    for path in ckpt_dir.glob("ckpt_*.pth"):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        stages.append(payload["cursor"]["stage"])
        assert "loaders" in payload and "train" in payload["loaders"]
        assert "ckpt_seq" in payload
    assert "train" in stages          # at least one mid-epoch checkpoint
    assert "done" in stages           # and the end-of-epoch one

    # A resume run reuses the folder and completes without error.
    config["checkpoint"]["resume"] = True
    config_path.write_text(yaml.safe_dump(config))
    train_entry.run(str(config_path))
    run_dir = tmp_path / "runs" / "pytest" / "train"
    assert (run_dir / "weights" / "last.pt").exists()


def test_raising_epochs_after_done_resumes_training(tmp_path, monkeypatch):
    """A finished run continues for the added epochs when `epochs` is raised.

    With in-epoch checkpointing on, a completed run's newest checkpoint is a
    "test" one written during the final evaluation. Raising `train.epochs` and
    resuming must train the added epochs (not report the run done), so the
    history ends with one row per epoch across the whole extended budget.
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _sine_zip(str(data_dir / "train.zip"), 12)
    _sine_zip(str(data_dir / "test.zip"), 4, base=300)
    config = _tiny_config(str(data_dir))
    config["train"]["epochs"] = 2
    config["train"]["grad_accum"] = 1
    config["checkpoint"] = {"max_checkpoint": 50, "resume": False, "ckpt_step": 1}
    config_path = tmp_path / "train.yaml"
    config_path.write_text(yaml.safe_dump(config))
    monkeypatch.chdir(tmp_path)

    train_entry.run(str(config_path))
    run_dir = tmp_path / "runs" / "pytest" / "train"
    assert _read_history_epochs(run_dir / "history.csv") == [0, 1]

    # The completed run left a "test" checkpoint as the newest on disk.
    latest = torch.load(
        sorted((run_dir / "checkpoints").glob("ckpt_*.pth"))[-1],
        map_location="cpu", weights_only=False)
    assert latest["cursor"]["stage"] == "test"

    # Raise the epoch budget and resume: the added epochs must be trained.
    config["train"]["epochs"] = 4
    config["checkpoint"]["resume"] = True
    config_path.write_text(yaml.safe_dump(config))
    train_entry.run(str(config_path))
    assert _read_history_epochs(run_dir / "history.csv") == [0, 1, 2, 3]


class _Boom(Exception):
    """Marker exception used to simulate a crash mid-epoch."""


def _read_history_epochs(csv_path):
    """Return the list of epoch numbers recorded in a history CSV."""
    import csv
    with open(csv_path, encoding="utf-8", newline="") as handle:
        return [int(row["epoch"]) for row in csv.DictReader(handle)]


def test_crash_mid_epoch_then_resume_finishes(tmp_path, monkeypatch):
    """A crash in the middle of an epoch resumes at the right batch and finishes.

    The first run is forced to crash right after its second in-epoch ("train")
    checkpoint, so the data loader is parked mid-epoch and no history row has been
    written yet. The resumed run must continue the same epoch (not restart it),
    complete every epoch, and leave the history with exactly one row per epoch
    (no duplicate from a replayed epoch).
    """
    from sjepa.trainer import Trainer

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _sine_zip(str(data_dir / "train.zip"), 12)
    _sine_zip(str(data_dir / "test.zip"), 4, base=300)
    config = _tiny_config(str(data_dir))
    config["train"]["epochs"] = 2
    config["train"]["batch_size"] = 3
    config["train"]["grad_accum"] = 1
    config["checkpoint"] = {"max_checkpoint": 50, "resume": False, "ckpt_step": 1}
    config_path = tmp_path / "train.yaml"
    config_path.write_text(yaml.safe_dump(config))
    monkeypatch.chdir(tmp_path)

    # Crash after the second mid-epoch train checkpoint (loader parked at batch 2
    # of the 4 train batches in epoch 0).
    real_save = Trainer._save_checkpoint
    seen = {"train": 0}

    def crashing_save(self, epoch, stage, extra=None):
        real_save(self, epoch, stage, extra)
        if stage == "train":
            seen["train"] += 1
            if seen["train"] == 2:
                raise _Boom()

    monkeypatch.setattr(Trainer, "_save_checkpoint", crashing_save)
    try:
        train_entry.run(str(config_path))
        raise AssertionError("the run was expected to crash mid-epoch")
    except _Boom:
        pass
    monkeypatch.setattr(Trainer, "_save_checkpoint", real_save)

    run_dir = tmp_path / "runs" / "pytest" / "train"
    # The crash happened before epoch 0 finished, so no history row exists yet.
    assert not (run_dir / "history.csv").exists()
    latest = torch.load(
        sorted((run_dir / "checkpoints").glob("ckpt_*.pth"))[-1],
        map_location="cpu", weights_only=False)
    assert latest["cursor"]["stage"] == "train"
    assert 0 < latest["loaders"]["train"]["batches_done"] < 4

    # Resume: must finish both epochs with exactly one history row each.
    config["checkpoint"]["resume"] = True
    config_path.write_text(yaml.safe_dump(config))
    train_entry.run(str(config_path))
    assert (run_dir / "weights" / "last.pt").exists()
    assert _read_history_epochs(run_dir / "history.csv") == [0, 1]
