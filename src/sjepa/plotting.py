"""Draw training history curves so we can spot overfitting.

After each epoch we plot the train and validation curves for every tracked
metric on the same figure. When the validation curve climbs while the train
curve keeps falling, the model is starting to overfit.

We use the non-interactive "Agg" backend so the plots are written to files even
on a machine with no screen.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .logging import get_logger  # noqa: E402

_LOGGER = get_logger()

# Progress bar style characters from the spec, reused here for tick clarity.
_TRAIN_COLOR = "#1f77b4"
_VAL_COLOR = "#d62728"


class HistoryPlotter:
    """Plot one figure per metric from the training history."""

    def __init__(self, out_dir):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

    @staticmethod
    def _metric_keys(history):
        """Return the metric base names that have train or val series."""
        keys = set()
        for row in history:
            for name in row:
                if name.startswith("train_"):
                    keys.add(name[len("train_"):])
                elif name.startswith("val_"):
                    keys.add(name[len("val_"):])
        return sorted(keys)

    @staticmethod
    def _series(history, prefix, metric):
        """Return (epochs, values) for one train or val metric series."""
        epochs, values = [], []
        for row in history:
            key = f"{prefix}_{metric}"
            if key in row and row[key] is not None:
                epochs.append(row["epoch"])
                values.append(row[key])
        return epochs, values

    def _plot_one(self, history, metric):
        """Draw and save the train vs val figure for one metric."""
        figure, axis = plt.subplots(figsize=(8, 5))
        train_x, train_y = self._series(history, "train", metric)
        val_x, val_y = self._series(history, "val", metric)
        if train_x:
            axis.plot(train_x, train_y, label="train", color=_TRAIN_COLOR)
        if val_x:
            axis.plot(val_x, val_y, label="val", color=_VAL_COLOR)
        axis.set_xlabel("epoch")
        axis.set_ylabel(metric)
        axis.set_title(f"training history: {metric}")
        axis.grid(True, alpha=0.3)
        axis.legend()
        path = os.path.join(self.out_dir, f"history_{metric}.jpg")
        figure.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(figure)
        return path

    def plot(self, history):
        """Plot every metric found in the history. Returns the file paths."""
        if not history:
            return []
        paths = []
        for metric in self._metric_keys(history):
            paths.append(self._plot_one(history, metric))
        _LOGGER.info("Saved {} history plots to {}", len(paths), self.out_dir)
        return paths
