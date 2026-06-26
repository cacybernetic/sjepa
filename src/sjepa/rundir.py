"""Manage the run output folders under "runs/".

Every training or evaluation run writes its files into one folder. The spec
fixes the layout:

    runs/
      <run_name>/
        train/    train2/    train3/   ...   (training runs)
        eval/     eval2/     eval3/    ...   (evaluation runs)

The first run of each kind has no number ("train", "eval"). The next ones add a
number ("train2", "train3"). When resume is on and a reusable checkpoint exists,
we reuse the highest-numbered existing folder instead of making a new one.

Each run folder holds these sub-folders:
    weights/  checkpoints/  plotes/  logs/
plus history.csv and config_used.yaml.
"""

from __future__ import annotations

import os
import re


def _index_of(name, kind):
    """Return the run number from a folder name, or None when it does not match.

    "train" -> 1, "train2" -> 2, "eval10" -> 10. The first run has no number,
    which we read as number 1.
    """
    match = re.fullmatch(rf"{kind}(\d*)", name)
    if match is None:
        return None
    suffix = match.group(1)
    return 1 if suffix == "" else int(suffix)


def _name_for(kind, index):
    """Return the folder name for a run kind and number."""
    return kind if index == 1 else f"{kind}{index}"


class RunLayout:
    """Hold the paths of one run folder and make its sub-folders."""

    SUBDIRS = ("weights", "checkpoints", "plotes", "logs")

    def __init__(self, root):
        self.root = root

    def create(self):
        """Make the run folder and every sub-folder. Returns self."""
        os.makedirs(self.root, exist_ok=True)
        for name in self.SUBDIRS:
            os.makedirs(os.path.join(self.root, name), exist_ok=True)
        return self

    def path(self, *parts):
        """Join parts onto the run root path."""
        return os.path.join(self.root, *parts)

    @property
    def weights_dir(self):
        return self.path("weights")

    @property
    def checkpoints_dir(self):
        return self.path("checkpoints")

    @property
    def plots_dir(self):
        return self.path("plotes")

    @property
    def logs_dir(self):
        return self.path("logs")

    @property
    def history_csv(self):
        return self.path("history.csv")

    @property
    def config_used(self):
        return self.path("config_used.yaml")


class RunDirectoryManager:
    """Pick the right run folder for a new or resumed run."""

    def __init__(self, runs_root, run_name, kind="train"):
        if kind not in ("train", "eval"):
            raise ValueError("kind must be 'train' or 'eval'")
        self.base = os.path.join(runs_root, run_name)
        self.kind = kind

    def _existing(self):
        """Return existing run numbers of this kind, sorted ascending."""
        if not os.path.isdir(self.base):
            return []
        numbers = []
        for name in os.listdir(self.base):
            index = _index_of(name, self.kind)
            if index is not None and os.path.isdir(os.path.join(self.base, name)):
                numbers.append(index)
        return sorted(numbers)

    def _has_checkpoint(self, index):
        """Return True when a run folder holds at least one checkpoint file."""
        layout = RunLayout(os.path.join(self.base, _name_for(self.kind, index)))
        folder = layout.checkpoints_dir
        if not os.path.isdir(folder):
            return False
        return any(name.endswith(".pth") for name in os.listdir(folder))

    def _resume_index(self):
        """Return the highest existing run number that has a checkpoint."""
        for index in reversed(self._existing()):
            if self._has_checkpoint(index):
                return index
        return None

    def _next_index(self):
        """Return the number for a brand new run folder."""
        existing = self._existing()
        if not existing:
            return 1
        return existing[-1] + 1

    def resolve(self, resume=False):
        """Return the `RunLayout` to use, creating its folders.

        Args:
            resume: when True, reuse the highest run folder that has a usable
                checkpoint. When no such folder exists, a new one is made.

        Returns:
            A pair (layout, resumed) where resumed is True when an old folder is
            reused.
        """
        if resume:
            index = self._resume_index()
            if index is not None:
                root = os.path.join(self.base, _name_for(self.kind, index))
                return RunLayout(root).create(), True
        root = os.path.join(self.base, _name_for(self.kind, self._next_index()))
        return RunLayout(root).create(), False
