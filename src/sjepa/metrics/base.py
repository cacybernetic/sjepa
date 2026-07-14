"""Small helpers shared by every metric.

A metric collects values during validation and returns one average at the end.
`AverageMeter` keeps a running mean so we never hold every value in memory.
"""

from __future__ import annotations

import torch


class AverageMeter:
    """Keep a running weighted average of a scalar value.

    Values and weights may be Python floats or 0-dim tensors. Tensor updates
    accumulate on their device with no host sync; the single sync happens in
    `average()` (called at the logging cadence and at the end of a pass), not
    on every batch.
    """

    def __init__(self):
        self.total = 0.0
        self.count = 0.0

    def reset(self):
        """Forget every value seen so far."""
        self.total = 0.0
        self.count = 0.0

    def update(self, value, weight=1.0):
        """Add one value with an optional weight (for example a frame count)."""
        if isinstance(value, torch.Tensor) or isinstance(weight, torch.Tensor):
            # No comparison against zero here: that would sync the device.
            # A zero weight simply contributes nothing to either sum.
            self.total = self.total + value * weight
            self.count = self.count + weight
            return
        if weight <= 0:
            return
        self.total += float(value) * weight
        self.count += weight

    def average(self):
        """Return the running average as a float, or 0.0 when empty."""
        count = float(self.count)
        if count <= 0:
            return 0.0
        return float(self.total) / count

    def state_dict(self):
        """Return the running totals (as plain floats) for checkpoints."""
        return {"total": float(self.total), "count": float(self.count)}

    def load_state_dict(self, state):
        """Restore the running totals from a checkpoint."""
        self.total = float(state["total"])
        self.count = float(state["count"])
