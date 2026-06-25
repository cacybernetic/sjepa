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

    def _num_spans(self, real_length):
        """Compute how many spans to place for one utterance."""
        spans = int(real_length * self.mask_ratio / self.mask_length)
        return max(1, spans)

    def _mask_one(self, mask_row, real_length):
        """Mask spans inside a single row in place."""
        if real_length <= 0:
            return
        high = max(1, real_length - self.mask_length)
        starts = torch.randint(0, high, (self._num_spans(real_length),))
        for start in starts.tolist():
            end = min(start + self.mask_length, real_length)
            mask_row[start:end] = True

    @torch.no_grad()
    def generate(self, batch_size, num_frames, frame_lengths, device):
        """Build the block mask for a whole batch.

        Args:
            batch_size: the number of utterances.
            num_frames: the padded length in frames.
            frame_lengths: real length per utterance in frames.
            device: the device where the mask is built.

        Returns:
            A bool tensor of shape (batch_size, num_frames). True means masked.
        """
        mask = torch.zeros(batch_size, num_frames, dtype=torch.bool,
                           device=device)
        for index in range(batch_size):
            self._mask_one(mask[index], int(frame_lengths[index]))
        return mask
