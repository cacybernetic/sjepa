"""EMA encoder and its switched decay schedule (phase 2).

The EMA encoder is a slow copy of the online encoder. Its weights follow an
exponential moving average of the online weights. In phase 2 it feeds the
online GMM with clean features. It is never trained by gradients.

The paper flips the decay rate between a fast value and a slow value on a fixed
cadence. The fast rate lets the EMA encoder follow recent changes; the slow
rate holds the target stable so the online encoder can learn against it.

Two small classes live here, each with one job:
  * `SwitchedEmaScheduler`: pick the decay rate for the current step.
  * `EmaEncoder`: keep the slow copy and update it each step.
"""

from copy import deepcopy

import torch
import torch.nn as nn


class SwitchedEmaScheduler:
    """Pick the EMA decay rate based on the training step.

    The rate starts at the fast value, then flips to the slow value after
    `switch_every` steps, then back, and so on.
    """

    def __init__(self, alpha_fast=0.999, alpha_slow=0.9999, switch_every=20000):
        if not 0.0 < alpha_fast < 1.0 or not 0.0 < alpha_slow < 1.0:
            raise ValueError("decay rates must be between 0 and 1")
        if switch_every <= 0:
            raise ValueError("switch_every must be > 0")
        self.alpha_fast = alpha_fast
        self.alpha_slow = alpha_slow
        self.switch_every = switch_every

    def decay(self, step):
        """Return the decay rate to use at the given step."""
        interval_index = step // self.switch_every
        if interval_index % 2 == 0:
            return self.alpha_fast
        return self.alpha_slow


class EmaEncoder(nn.Module):
    """A frozen moving-average copy of the online encoder."""

    def __init__(self, online_encoder, scheduler=None):
        super().__init__()
        self.encoder = deepcopy(online_encoder)
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad_(False)
        self.scheduler = scheduler or SwitchedEmaScheduler()

    @torch.no_grad()
    def update(self, online_encoder, step):
        """Move the EMA weights a small step toward the online weights.

        Args:
            online_encoder: the encoder being trained.
            step: the current training step, used to pick the decay rate.

        Returns:
            The decay rate that was used for this update.
        """
        alpha = self.scheduler.decay(step)
        ema_params = list(self.encoder.parameters())
        online_params = list(online_encoder.parameters())
        # Fused in-place update: ema = alpha * ema + (1 - alpha) * online.
        # torch._foreach_* is a private API; fall back to a plain loop if a
        # future torch release drops it.
        try:
            torch._foreach_mul_(ema_params, alpha)
            torch._foreach_add_(ema_params, online_params, alpha=1.0 - alpha)
        except AttributeError:
            for ema_param, online_param in zip(ema_params, online_params):
                ema_param.mul_(alpha).add_(online_param, alpha=1.0 - alpha)
        return alpha

    @torch.no_grad()
    def extract_layer(self, waveform, layer_index, padding_mask=None):
        """Return clean features at one layer from the slow encoder."""
        return self.encoder.extract_layer(waveform, layer_index, padding_mask)

    @torch.no_grad()
    def extract_all_layers(self, waveform, padding_mask=None):
        """Return clean features of every layer from one forward pass."""
        return self.encoder.extract_all_layers(waveform, padding_mask)
