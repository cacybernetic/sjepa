"""Geeky but clean terminal progress bars built on tqdm.

The spec asks for two bars:

  * a big "epoch" bar that stays on screen (leave=True) and shows the total
    training progress, the elapsed and remaining time, the average epoch
    duration, the best score, and the current learning rate;
  * a small "step" bar that disappears when done (leave=False) and shows the
    metrics of the current stage (train or validation).

The fill character is the full block and the background is the light block, as
the spec requires. No emoji and no special keys are used.
"""

from __future__ import annotations

import time

from tqdm import tqdm

# tqdm reads the "ascii" string from empty to full. Light block is the
# background, full block is the fill, matching the spec preview.
_BAR_ASCII = " >="
_BLOCKS = "░█"


def _format_seconds(seconds):
    """Return a short H:MM:SS string for a duration in seconds."""
    seconds = int(seconds)
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours:d}:{minutes:02d}:{secs:02d}"


class EpochProgress:
    """The big bar that tracks the whole training across epochs."""

    def __init__(self, total_epochs, start_epoch=0):
        self.bar = tqdm(
            total=total_epochs, initial=start_epoch, leave=True,
            desc="TRAINING", ascii=_BLOCKS, dynamic_ncols=True,
            bar_format="{desc}: {percentage:3.0f}%|{bar}| "
                       "{n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}",
        )
        self.start_time = time.time()
        self.start_epoch = start_epoch

    def update(self, best_value, lr, avg_epoch_seconds):
        """Advance by one epoch and refresh the prefix information."""
        elapsed = time.time() - self.start_time
        best_text = "n/a" if best_value is None else f"{best_value:.4f}"
        self.bar.set_postfix_str(
            f"best={best_text} lr={lr:.2e} "
            f"avg_epoch={_format_seconds(avg_epoch_seconds)} "
            f"elapsed={_format_seconds(elapsed)}")
        self.bar.update(1)

    def close(self):
        """Close the bar."""
        self.bar.close()


class StepProgress:
    """The small bar that tracks one stage (train or validation)."""

    def __init__(self, total_steps, stage, epoch, num_epochs):
        desc = f"{stage} e{epoch}/{num_epochs}"
        self.bar = tqdm(
            total=total_steps, leave=False, desc=desc, ascii=_BLOCKS,
            dynamic_ncols=True,
            bar_format="    {desc}: {percentage:3.0f}%|{bar}| "
                       "{n_fmt}/{total_fmt} [{elapsed}<{remaining}, "
                       "{rate_fmt}]{postfix}",
        )

    def update(self, metrics):
        """Advance by one step and show the current metric values."""
        if metrics:
            text = " ".join(f"{key}={value:.3f}" for key, value in metrics.items())
            self.bar.set_postfix_str(text)
        self.bar.update(1)

    def close(self):
        """Close the bar."""
        self.bar.close()
