"""Save and restore the full training state.

The spec asks for a checkpoint at the end of every epoch that holds the whole
state: model weights, optimizer, scheduler, epoch, step, best score, history,
and the extra Phase 2 state (EMA encoder and online GMM). It also asks to keep
only the newest few checkpoints and to delete the oldest ones.

Two responsibilities live here, each in its own class:

  * `CheckpointManager`: write, rotate, and find epoch checkpoints.
  * `WeightSaver`: write the plain model weights for best.pt and last.pt.
"""

from __future__ import annotations

import os
import re

import torch

from .logging import get_logger

_LOGGER = get_logger()
_EPOCH_RE = re.compile(r"epoch_(\d+)\.pth")


def _epoch_of(name):
    """Return the epoch number stored in a checkpoint file name, or None."""
    match = _EPOCH_RE.fullmatch(name)
    return int(match.group(1)) if match else None


class CheckpointManager:
    """Write epoch checkpoints, keep the newest few, and find the latest."""

    def __init__(self, checkpoints_dir, max_checkpoints=5):
        if max_checkpoints <= 0:
            raise ValueError("max_checkpoints must be > 0")
        self.dir = checkpoints_dir
        self.max_checkpoints = max_checkpoints
        os.makedirs(self.dir, exist_ok=True)

    def _epochs(self):
        """Return the sorted list of epoch numbers found on disk."""
        numbers = []
        for name in os.listdir(self.dir):
            epoch = _epoch_of(name)
            if epoch is not None:
                numbers.append(epoch)
        return sorted(numbers)

    def _path(self, epoch):
        """Return the file path for one epoch checkpoint."""
        return os.path.join(self.dir, f"epoch_{epoch:03d}.pth")

    def _rotate(self):
        """Delete the oldest checkpoints beyond the keep limit."""
        epochs = self._epochs()
        extra = len(epochs) - self.max_checkpoints
        for epoch in epochs[:max(0, extra)]:
            os.remove(self._path(epoch))
            _LOGGER.info("Removed old checkpoint epoch {}", epoch)

    def save(self, payload, epoch):
        """Write one checkpoint and rotate the old ones. Returns the path."""
        path = self._path(epoch)
        torch.save(payload, path)
        _LOGGER.info("Saved checkpoint to {}", path)
        self._rotate()
        return path

    def latest_path(self):
        """Return the path of the newest checkpoint, or None when there is none."""
        epochs = self._epochs()
        if not epochs:
            return None
        return self._path(epochs[-1])

    @staticmethod
    def load(path, map_location="cpu"):
        """Read a checkpoint payload from disk."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"checkpoint not found: {path}")
        return torch.load(path, map_location=map_location, weights_only=False)


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
        torch.save(payload, path)
        _LOGGER.info("Saved weights to {}", path)
        return path

    def load(self, name, map_location="cpu"):
        """Load a weight payload by file name."""
        path = self._path(name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"weights not found: {path}")
        return torch.load(path, map_location=map_location, weights_only=False)
