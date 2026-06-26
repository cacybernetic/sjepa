"""Small helpers shared by every metric.

A metric collects values during validation and returns one average at the end.
`AverageMeter` keeps a running mean so we never hold every value in memory.
"""

from __future__ import annotations


class AverageMeter:
    """Keep a running weighted average of a scalar value."""

    def __init__(self):
        self.total = 0.0
        self.count = 0.0

    def reset(self):
        """Forget every value seen so far."""
        self.total = 0.0
        self.count = 0.0

    def update(self, value, weight=1.0):
        """Add one value with an optional weight (for example a frame count)."""
        if weight <= 0:
            return
        self.total += float(value) * weight
        self.count += weight

    def average(self):
        """Return the running average, or 0.0 when nothing was added."""
        if self.count <= 0:
            return 0.0
        return self.total / self.count
