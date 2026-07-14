"""Save and restore the full training state.

A checkpoint holds the whole state: model weights, optimizer, scheduler, epoch,
global step, best score, the in-epoch cursor and data loader positions, and the
extra Phase 2 state (EMA encoder and online GMM). Checkpoints are written both at
the end of an epoch and, for in-epoch checkpointing, every `ckpt_step` steps. The
manager keeps only the newest few and deletes the oldest ones.

Each file is named `ckpt_e{epoch:03d}_s{global_step:09d}_n{seq:09d}.pth`. The
global step does not advance during validation or evaluation, so it cannot order
those in-epoch checkpoints on its own; a strictly increasing save sequence `seq`
gives a total chronological order over every checkpoint. Rotation and "latest"
both use `seq` as the key, while the epoch and step stay in the name for reading.

Two responsibilities live here, each in its own class:

  * `CheckpointManager`: write, rotate, and find checkpoints.
  * `WeightSaver`: write the plain model weights for best.pt and last.pt.
"""

from __future__ import annotations

import os
import re

import torch

from .logging import get_logger

_LOGGER = get_logger()
_CKPT_RE = re.compile(r"ckpt_e(\d+)_s(\d+)_n(\d+)\.pth")


def _atomic_save(payload, path):
    """Write with torch.save to a temp file, then rename atomically.

    A crash in the middle of `torch.save` straight to the final path leaves a
    truncated file that the next resume would pick as "latest" and fail on.
    `os.replace` is atomic on POSIX, so the final name only ever points to a
    complete file.
    """
    tmp_path = f"{path}.tmp"
    try:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def _seq_of(name):
    """Return the save sequence number in a checkpoint file name, or None."""
    match = _CKPT_RE.fullmatch(name)
    return int(match.group(3)) if match else None


def is_checkpoint_file(name):
    """Return True when a file name is a checkpoint the manager can load.

    Used by the run-folder resolver so a folder that only holds checkpoints in an
    older, unreadable naming scheme is not mistaken for a resumable run (which
    would silently restart from epoch 0 and overwrite it).
    """
    return _CKPT_RE.fullmatch(name) is not None


class CheckpointManager:
    """Write epoch checkpoints, keep the newest few, and find the latest."""

    def __init__(self, checkpoints_dir, max_checkpoints=5):
        if max_checkpoints <= 0:
            raise ValueError("max_checkpoints must be > 0")
        self.dir = checkpoints_dir
        self.max_checkpoints = max_checkpoints
        os.makedirs(self.dir, exist_ok=True)

    def _entries(self):
        """Return the checkpoint files on disk as (seq, name), sorted by seq."""
        entries = []
        for name in os.listdir(self.dir):
            seq = _seq_of(name)
            if seq is not None:
                entries.append((seq, name))
        return sorted(entries)

    def _path(self, epoch, global_step, seq):
        """Return the file path for one checkpoint."""
        return os.path.join(
            self.dir, f"ckpt_e{epoch:03d}_s{global_step:09d}_n{seq:09d}.pth")

    def _rotate(self):
        """Delete the oldest checkpoints beyond the keep limit."""
        entries = self._entries()
        extra = len(entries) - self.max_checkpoints
        for seq, name in entries[:max(0, extra)]:
            os.remove(os.path.join(self.dir, name))
            _LOGGER.info("Removed old checkpoint {} (seq {})", name, seq)

    def latest_seq(self):
        """Return the highest save sequence on disk, or -1 when empty."""
        entries = self._entries()
        return entries[-1][0] if entries else -1

    def save(self, payload, epoch, global_step=None, seq=None):
        """Write one checkpoint and rotate the old ones. Returns the path.

        `seq` is the strictly increasing save sequence that keys the file name
        and the rotation order. It is read from the payload (`ckpt_seq`) when not
        passed explicitly; `global_step` defaults the same way.
        """
        if global_step is None:
            global_step = int(payload.get("global_step", 0))
        if seq is None:
            seq = int(payload.get("ckpt_seq", 0))
        path = self._path(epoch, global_step, seq)
        _atomic_save(payload, path)
        _LOGGER.info("Saved checkpoint to {}", path)
        self._rotate()
        return path

    def latest_path(self):
        """Return the path of the newest checkpoint, or None when there is none."""
        entries = self._entries()
        if not entries:
            return None
        return os.path.join(self.dir, entries[-1][1])

    def paths_newest_first(self):
        """Return every checkpoint path, newest first (resume fallback order)."""
        return [os.path.join(self.dir, name)
                for _, name in reversed(self._entries())]

    def has_checkpoint(self):
        """Return True when at least one checkpoint exists on disk."""
        return bool(self._entries())

    @staticmethod
    def load(path, map_location="cpu"):
        """Read a checkpoint payload from disk.

        `weights_only=True`: the payload holds only tensors, containers, and
        primitives, and the restricted unpickler cannot execute code from a
        tampered file.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"checkpoint not found: {path}")
        return torch.load(path, map_location=map_location, weights_only=True)


class WeightSaver:
    """Write and read the plain model weights (best.pt and last.pt)."""

    def __init__(self, weights_dir):
        self.dir = weights_dir
        os.makedirs(self.dir, exist_ok=True)

    def _path(self, name):
        """Return the file path for a named weight file."""
        return os.path.join(self.dir, name)

    def save(self, model, name, extra=None):
        """Save the model state dict (plus optional extra fields)."""
        payload = {"model": model.state_dict()}
        if extra:
            payload.update(extra)
        path = self._path(name)
        _atomic_save(payload, path)
        _LOGGER.info("Saved weights to {}", path)
        return path

    def load(self, name, map_location="cpu"):
        """Load a weight payload by file name."""
        path = self._path(name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"weights not found: {path}")
        return torch.load(path, map_location=map_location, weights_only=True)
