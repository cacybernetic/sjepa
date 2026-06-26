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
    assert (run_dir / "checkpoints" / "epoch_000.pth").exists()
    assert any((run_dir / "plotes").iterdir())
