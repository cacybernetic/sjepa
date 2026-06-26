"""Validation and analysis metrics for S-JEPA.

The metrics work with soft GMM targets and the predictor output. They are used
during validation to track progress and to pick the best model.
"""

from .base import AverageMeter
from .functional import (
    effective_rank,
    kl_divergence,
    predictor_entropy_bits,
    top1_agreement,
)
from .metrics import (
    EntropyMetric,
    KlMetric,
    MetricGroup,
    Top1Metric,
)

__all__ = [
    "AverageMeter",
    "effective_rank",
    "kl_divergence",
    "predictor_entropy_bits",
    "top1_agreement",
    "EntropyMetric",
    "KlMetric",
    "MetricGroup",
    "Top1Metric",
]
