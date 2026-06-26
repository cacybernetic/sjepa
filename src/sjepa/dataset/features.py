"""Compute 39-dim MFCC features used as the Phase 1 GMM input.

The paper fits the Phase 1 GMM on 13 MFCC coefficients plus their first and
second deltas, for a total of 39 features per frame. The frame rate matches the
encoder: one frame every 320 samples at 16 kHz (20 ms).

This class has one job: turn a waveform into a (frames, 39) feature matrix.
"""

from __future__ import annotations

import torch
import torchaudio
import torchaudio.functional as AF


class MfccExtractor:
    """Build 39-dim MFCC + delta + delta-delta features per frame."""

    def __init__(self, sample_rate=16000, n_mfcc=13, hop=320, n_fft=400,
                 n_mels=23):
        self.hop = hop
        self.dim = n_mfcc * 3
        self.transform = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            melkwargs={"n_fft": n_fft, "hop_length": hop, "n_mels": n_mels},
        )

    def to(self, device):
        """Move the inner transform to a device. Returns self."""
        self.transform = self.transform.to(device)
        return self

    def _stack_deltas(self, mfcc):
        """Stack MFCC with its first and second deltas along the feature axis."""
        delta1 = AF.compute_deltas(mfcc)
        delta2 = AF.compute_deltas(delta1)
        return torch.cat([mfcc, delta1, delta2], dim=-2)

    @torch.no_grad()
    def extract(self, waveform):
        """Compute features for one waveform or a batch.

        Args:
            waveform: tensor of shape (samples,) or (batch, samples).

        Returns:
            A tensor of shape (frames, 39) for one clip, or
            (batch, frames, 39) for a batch.
        """
        single = waveform.dim() == 1
        if single:
            waveform = waveform.unsqueeze(0)
        mfcc = self.transform(waveform.float())
        features = self._stack_deltas(mfcc)
        features = features.transpose(-1, -2)
        return features.squeeze(0) if single else features
