"""Learning rate schedulers for stable, production-quality training.

A warmup phase raises the learning rate from a small value to the base value.
After warmup the rate follows a cosine curve down to a small floor, or stays
constant, depending on the chosen kind.

We build a `torch.optim.lr_scheduler.LambdaLR` so resume is easy: the scheduler
state is saved and restored with the optimizer.
"""

from __future__ import annotations

import math

from torch.optim.lr_scheduler import LambdaLR

from .logging import get_logger, log_hparams

_LOGGER = get_logger()


def _warmup_factor(step, warmup_steps):
    """Return the linear warmup factor for an early step."""
    if warmup_steps <= 0:
        return 1.0
    return min(1.0, float(step + 1) / float(warmup_steps))


def _cosine_factor(step, warmup_steps, total_steps, min_ratio):
    """Return the cosine decay factor after warmup."""
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_ratio + (1.0 - min_ratio) * cosine


class _LambdaBuilder:
    """Build the step-to-factor function for one scheduler kind.

    When `phase2_start_step` is set, the schedule is restarted at that step: the
    learning rate warms up again and then decays over the remaining steps. This
    gives Phase 2 a real learning-rate budget instead of the decayed tail of the
    single whole-run cosine, which otherwise leaves the most important phase
    training at a near-zero rate.

    The restarted window is scaled by `phase2_lr_ratio`: the paper trains
    Phase 2 at 2.5e-5 against 1e-4 in Phase 1 (ratio 0.25). Restarting to the
    full base rate would hit the self-referential Phase 2 targets (online GMM
    over EMA features) with 4x the intended step size, right after the cluster
    head was rebuilt.
    """

    def __init__(self, kind, warmup_steps, total_steps, min_ratio,
                 phase2_start_step=None, phase2_lr_ratio=1.0):
        self.kind = kind
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_ratio = min_ratio
        self.phase2_start_step = phase2_start_step
        self.phase2_lr_ratio = phase2_lr_ratio

    def _factor(self, step, start, warmup_steps):
        """Warmup + (cosine|constant) factor for a window starting at `start`."""
        local = step - start
        if local < warmup_steps:
            return _warmup_factor(local, warmup_steps)
        if self.kind == "constant":
            return 1.0
        return _cosine_factor(step, start + warmup_steps, self.total_steps,
                              self.min_ratio)

    def __call__(self, step):
        """Return the multiplier applied to the base learning rate at a step."""
        if (self.phase2_start_step is not None
                and step >= self.phase2_start_step):
            factor = self._factor(step, self.phase2_start_step,
                                  self.warmup_steps)
            return factor * self.phase2_lr_ratio
        return self._factor(step, 0, self.warmup_steps)


def build_scheduler(optimizer, kind="cosine", warmup_steps=0, total_steps=1,
                    min_ratio=0.0, phase2_start_step=None,
                    phase2_lr_ratio=1.0):
    """Build a learning rate scheduler.

    Args:
        optimizer: the optimizer to schedule.
        kind: "cosine" or "constant".
        warmup_steps: number of warmup steps.
        total_steps: total number of optimizer steps over the whole run.
        min_ratio: the floor as a fraction of the base learning rate.
        phase2_start_step: optimizer step at which Phase 2 begins. When given,
            the learning rate warms up again from that step (a warm restart) so
            Phase 2 is not stuck on the decayed tail of the Phase 1 cosine.
        phase2_lr_ratio: multiplier applied to the whole Phase 2 window (the
            paper trains Phase 2 at a quarter of the Phase 1 rate).

    Returns:
        A `LambdaLR` scheduler.
    """
    if kind not in ("cosine", "constant"):
        raise ValueError(f"unknown scheduler '{kind}'")
    if not 0.0 < phase2_lr_ratio <= 1.0:
        raise ValueError("phase2_lr_ratio must be in (0, 1]")
    builder = _LambdaBuilder(kind, warmup_steps, total_steps, min_ratio,
                             phase2_start_step, phase2_lr_ratio)
    log_hparams("scheduler", {"kind": kind, "warmup_steps": warmup_steps,
                              "total_steps": total_steps,
                              "min_ratio": min_ratio,
                              "phase2_start_step": phase2_start_step,
                              "phase2_lr_ratio": phase2_lr_ratio},
                color="green")
    return LambdaLR(optimizer, lr_lambda=builder)
