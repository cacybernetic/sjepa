"""Run the model on one batch and compute the loss.

This module keeps the per-batch work in one place so the training loop stays
short. It builds the block mask and the padding mask, makes the soft targets
from the clean waveform, runs the model on the augmented waveform, aligns every
tensor to the encoder length, and returns the loss and the parts needed for
metrics.

The encoder may output one or two fewer frames than the analytic estimate, so
we always cut the targets and masks to the shortest shared length.

The model forward can run under `torch.autocast` (bf16) when the config asks
for it; the GMM targets are always built in float32 outside the autocast
region, and the loss itself is computed in float32.
"""

from __future__ import annotations

from contextlib import nullcontext

import torch

from .modules.masking import BlockMaskGenerator, build_padding_mask


class MaskBuilder:
    """Build the block mask and the padding mask for one batch."""

    def __init__(self, mask_ratio=0.65, mask_length=10):
        self.generator = BlockMaskGenerator(mask_ratio, mask_length)

    def build(self, batch_size, frames, frame_lengths, device):
        """Return (block_mask, padding_mask) for a batch."""
        mask = self.generator.generate(batch_size, frames, frame_lengths, device)
        padding = build_padding_mask(batch_size, frames, frame_lengths, device)
        return mask, padding


def _align(length, *tensors):
    """Cut every tensor along dim 1 to a shared length."""
    return tuple(tensor[:, :length] for tensor in tensors)


class ForwardStep:
    """Forward pass plus loss for one batch."""

    def __init__(self, model, objective, mask_builder, hop=320, device="cpu",
                 autocast_dtype=None):
        self.model = model
        self.objective = objective
        self.mask_builder = mask_builder
        self.hop = hop
        self.device = device
        self.autocast_dtype = autocast_dtype

    def _autocast(self):
        """Return the autocast context for the model forward (or a no-op)."""
        if self.autocast_dtype is None or not str(self.device).startswith("cuda"):
            return nullcontext()
        return torch.autocast("cuda", dtype=self.autocast_dtype)

    def _prepare(self, batch):
        """Move the batch to the device and read its shape."""
        waveform = batch["waveform"].to(self.device, non_blocking=True)
        frame_lengths = batch["frame_lengths"]
        frames = waveform.shape[-1] // self.hop
        return waveform, frame_lengths, frames

    def _selection(self, mask, padding):
        """Return the masked-and-real frame selection for metrics."""
        return mask & padding

    def run(self, batch, target_builder, augmentor=None, accumulate=False):
        """Run the batch and return a result dict.

        Args:
            batch: the input batch.
            target_builder: builder for the soft GMM targets.
            augmentor: optional waveform augmentor (training only).
            accumulate: forwarded to the target builder so the Phase 2 online
                GMM can buffer statistics across a gradient-accumulation window.

        Returns:
            A dict with "loss" (tensor), "components" (floats),
            "logits_masked", "targets", and "selection" for metrics.
        """
        waveform, frame_lengths, frames = self._prepare(batch)
        batch_size = waveform.shape[0]
        encoder_input = augmentor(waveform) if augmentor is not None else waveform
        mask, padding = self.mask_builder.build(batch_size, frames,
                                                frame_lengths, self.device)
        targets = target_builder.build(waveform, padding_mask=padding,
                                       accumulate=accumulate)
        # When no clip in the batch is padded, skip the attention mask entirely
        # so the fused attention can pick its fastest kernel (flash attention
        # rejects arbitrary masks). Decided from CPU lengths: no device sync.
        model_padding = None if min(frame_lengths) >= frames else padding
        with self._autocast():
            output = self.model(encoder_input, mask, model_padding)
        length = min(output.logits_masked.shape[1], targets.shape[1])
        return self._finish(output, targets, padding, length)

    def _finish(self, output, targets, padding, length):
        """Align tensors, compute the loss, and pack the result dict."""
        logits_masked, logits_visible, targets = _align(
            length, output.logits_masked, output.logits_visible, targets)
        mask, padding = _align(length, output.mask, padding)
        loss = self.objective(logits_masked, logits_visible, targets, mask,
                              padding)
        return {
            "loss": loss["loss"],
            # Kept as detached tensors: converting to float here would force a
            # host sync on every micro-batch; the logger converts at cadence.
            "components": {"loss_masked": loss["loss_masked"],
                           "loss_visible": loss["loss_visible"]},
            "logits_masked": logits_masked.detach(),
            "targets": targets,
            "selection": self._selection(mask, padding),
        }
