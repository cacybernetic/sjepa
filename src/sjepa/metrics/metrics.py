"""Metric classes that collect values over a whole validation pass.

Each metric wraps one function from `functional.py` with an `AverageMeter`. The
training loop calls `update` on every batch and `compute` at the end. A
`MetricGroup` runs several metrics at once and returns a name-to-value dict.

The metric names are stable so they can be used as the "best model" criterion
in the config (for example "kl" with mode "min").
"""

from __future__ import annotations

from .base import AverageMeter
from .functional import (
    kl_divergence,
    predictor_entropy_bits,
    top1_agreement,
)


class _SelectionMetric:
    """A metric computed on the masked frames of each batch."""

    name = "metric"
    mode = "min"

    def __init__(self):
        self.meter = AverageMeter()

    def reset(self):
        """Clear the running average."""
        self.meter.reset()

    @staticmethod
    def _count(selection):
        """Return how many frames are selected in the batch."""
        return int(selection.sum())

    def compute(self):
        """Return the average value over the whole pass."""
        return self.meter.average()

    def state_dict(self):
        """Return the meter state so a partial pass can be resumed."""
        return self.meter.state_dict()

    def load_state_dict(self, state):
        """Restore the meter state from a checkpoint."""
        self.meter.load_state_dict(state)


class KlMetric(_SelectionMetric):
    """Mean KL divergence at masked frames (lower is better)."""

    name = "kl"
    mode = "min"

    def update(self, logits, targets, selection):
        """Add the KL value of one batch."""
        value = kl_divergence(logits, targets, selection)
        self.meter.update(value, self._count(selection))


class Top1Metric(_SelectionMetric):
    """Top-1 cluster agreement at masked frames (higher is better)."""

    name = "top1"
    mode = "max"

    def update(self, logits, targets, selection):
        """Add the agreement value of one batch."""
        value = top1_agreement(logits, targets, selection)
        self.meter.update(value, self._count(selection))


class EntropyMetric(_SelectionMetric):
    """Mean predictor entropy in bits at masked frames (analysis only)."""

    name = "entropy_bits"
    mode = "min"

    def update(self, logits, targets, selection):
        """Add the entropy value of one batch."""
        value = predictor_entropy_bits(logits, selection)
        self.meter.update(value, self._count(selection))


_METRIC_TYPES = {
    KlMetric.name: KlMetric,
    Top1Metric.name: Top1Metric,
    EntropyMetric.name: EntropyMetric,
}


class MetricGroup:
    """Run several metrics together over a validation pass."""

    def __init__(self, names=None):
        names = names or list(_METRIC_TYPES.keys())
        self.metrics = [self._build(name) for name in names]

    @staticmethod
    def _build(name):
        """Build one metric by name."""
        if name not in _METRIC_TYPES:
            raise ValueError(f"unknown metric '{name}'")
        return _METRIC_TYPES[name]()

    def reset(self):
        """Reset every metric for a new pass."""
        for metric in self.metrics:
            metric.reset()

    def update(self, logits, targets, selection):
        """Update every metric with one batch."""
        for metric in self.metrics:
            metric.update(logits, targets, selection)

    def compute(self):
        """Return a dict that maps each metric name to its value."""
        return {metric.name: metric.compute() for metric in self.metrics}

    def state_dict(self):
        """Return every metric's running state, keyed by metric name."""
        return {metric.name: metric.state_dict() for metric in self.metrics}

    def load_state_dict(self, state):
        """Restore every metric whose name is present in the saved state."""
        for metric in self.metrics:
            if metric.name in state:
                metric.load_state_dict(state[metric.name])

    @staticmethod
    def mode_of(name):
        """Return 'min' or 'max' for a metric name (how 'best' is judged)."""
        if name not in _METRIC_TYPES:
            raise ValueError(f"unknown metric '{name}'")
        return _METRIC_TYPES[name].mode
