"""Masking helpers for the JEPA objective.

Two jobs live here, each in its own function or class:

  * `build_padding_mask`: mark which frames are real and which are padding.
  * `BlockMaskGenerator`: choose contiguous spans of frames to hide. The encoder
    will zero these frames and the predictor will try to fill them in.

A True value in the block mask means "this frame is masked (hidden)".
A True value in the padding mask means "this frame is real (not padding)".
"""

import torch


def build_padding_mask(batch_size, num_frames, frame_lengths, device):
    """Build a mask that marks real frames versus padding.

    Args:
        batch_size: the number of utterances in the batch.
        num_frames: the padded length in frames.
        frame_lengths: a list with the real length of each utterance in frames.
        device: the device where the mask is built.

    Returns:
        A bool tensor of shape (batch_size, num_frames). True means real frame.
    """
    lengths = torch.as_tensor(frame_lengths, device=device).view(batch_size, 1)
    positions = torch.arange(num_frames, device=device).view(1, num_frames)
    return positions < lengths


class BlockMaskGenerator:
    """Sample contiguous spans of frames to mask.

    The generator hides spans of `mask_length` frames until about `mask_ratio`
    of the real frames are masked. It only masks real frames, never padding.
    """

    def __init__(self, mask_ratio=0.65, mask_length=10):
        if not 0.0 < mask_ratio < 1.0:
            raise ValueError("mask_ratio must be between 0 and 1")
        if mask_length <= 0:
            raise ValueError("mask_length must be > 0")
        self.mask_ratio = mask_ratio
        self.mask_length = mask_length

    def _max_attempts(self, target):
        """Bound the number of spans we try, so the loop always stops.

        We may need extra spans because spans can overlap. This bound is large
        enough to reach the target fraction, but it still stops the loop if the
        target can never be reached.
        """
        spans_needed = target // self.mask_length + 1
        return 4 * spans_needed + 16

    def _mask_one(self, mask_row, real_length):
        """Mask spans inside a single row until the target fraction is met.

        We keep adding spans until the masked count reaches the target. Spans
        may overlap, so we count only the newly masked frames each time. A
        counter stops the loop so it can never run forever.
        """
        if real_length <= 0:
            return
        target = int(real_length * self.mask_ratio)
        if target <= 0:
            return
        high = max(1, real_length - self.mask_length)
        masked = 0
        attempts = 0
        max_attempts = self._max_attempts(target)
        while masked < target and attempts < max_attempts:
            start = int(torch.randint(0, high, (1,)).item())
            end = min(start + self.mask_length, real_length)
            masked += int((~mask_row[start:end]).sum().item())
            mask_row[start:end] = True
            attempts += 1

    @torch.no_grad()
    def generate(self, batch_size, num_frames, frame_lengths, device):
        """Build the block mask for a whole batch.

        The mask is built on the CPU and moved to the device in one transfer:
        the span loop reads back masked counts, and doing that on a CUDA
        tensor would force hundreds of host/device syncs per batch.

        Args:
            batch_size: the number of utterances.
            num_frames: the padded length in frames.
            frame_lengths: real length per utterance in frames.
            device: the device where the mask is returned.

        Returns:
            A bool tensor of shape (batch_size, num_frames). True means masked.
        """
        mask = torch.zeros(batch_size, num_frames, dtype=torch.bool)
        for index in range(batch_size):
            self._mask_one(mask[index], int(frame_lengths[index]))
        return mask.to(device)
