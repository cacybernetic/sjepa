"""Pure functions for the S-JEPA metrics.

These functions take tensors and return plain Python floats. They are easy to
test on their own and are reused by the metric classes.

The metrics fit a self-supervised model with soft targets:

  * `kl_divergence`: distance between the target and the prediction (lower is
    better). This is the main validation criterion.
  * `top1_agreement`: fraction of frames where the predicted top cluster equals
    the target top cluster (higher is better).
  * `predictor_entropy_bits`: average uncertainty of the prediction in bits.
  * `effective_rank`: spread of the feature spectrum (richness of features).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _select(values, selection):
    """Keep only the selected rows of a (B, T, K) tensor, return (M, K)."""
    num_clusters = values.shape[-1]
    flat = values.reshape(-1, num_clusters)
    keep = selection.reshape(-1)
    return flat[keep]


def kl_divergence(logits, targets, selection):
    """Return the mean KL(target || softmax(logits)) over selected frames.

    Returns a 0-dim tensor (or 0.0 when nothing is selected) so callers can
    accumulate without forcing a host/device sync on every batch.
    """
    chosen_logits = _select(logits, selection)
    chosen_targets = _select(targets, selection)
    if chosen_logits.shape[0] == 0:
        return 0.0
    log_pred = F.log_softmax(chosen_logits.float(), dim=-1)
    return F.kl_div(log_pred, chosen_targets.float(), reduction="batchmean")


def top1_agreement(logits, targets, selection):
    """Return the fraction where the predicted and target top cluster match."""
    chosen_logits = _select(logits, selection)
    chosen_targets = _select(targets, selection)
    if chosen_logits.shape[0] == 0:
        return 0.0
    pred = chosen_logits.argmax(dim=-1)
    gold = chosen_targets.argmax(dim=-1)
    return (pred == gold).float().mean()


def predictor_entropy_bits(logits, selection):
    """Return the mean per-frame entropy of the prediction, in bits."""
    chosen = _select(logits, selection)
    if chosen.shape[0] == 0:
        return 0.0
    log_prob = F.log_softmax(chosen.float(), dim=-1)
    prob = log_prob.exp()
    entropy_nats = -(prob * log_prob).sum(dim=-1)
    return entropy_nats.mean() / math.log(2.0)


def effective_rank(features, max_rows=2000):
    """Return the effective rank of a (N, D) feature matrix.

    The effective rank is the exponential of the entropy of the normalized
    singular value spectrum. It is high when many directions are used.
    """
    flat = features.reshape(-1, features.shape[-1]).float()
    if flat.shape[0] > max_rows:
        index = torch.randperm(flat.shape[0])[:max_rows]
        flat = flat[index]
    centered = flat - flat.mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(centered)
    spectrum = singular / singular.sum().clamp(min=1e-12)
    spectrum = spectrum[spectrum > 1e-10]
    entropy = -(spectrum * spectrum.log()).sum()
    return float(entropy.exp())
