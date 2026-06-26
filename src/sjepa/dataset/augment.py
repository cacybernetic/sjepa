"""Denoising augmentation for the waveform (WavLM style).

The encoder sees an augmented waveform while the GMM target is computed from the
clean waveform. Two effects are used, each with its own probability:

  * noise mix: add another clip as noise at a random signal-to-noise ratio;
  * utterance mix: overlay a slice of another clip at a random energy ratio.

A small rolling buffer keeps a few past clips to use as noise or overlay. This
class has one job: return an augmented copy of a batch of waveforms.
"""

from __future__ import annotations

import random

import torch
import torch.nn.functional as F


def mix_at_snr(clean, noise, snr_db):
    """Add noise to a clean signal at a target signal-to-noise ratio."""
    clean_energy = (clean.pow(2).mean()).clamp(min=1e-8)
    noise_energy = (noise.pow(2).mean()).clamp(min=1e-8)
    scale = torch.sqrt(clean_energy / (10 ** (snr_db / 10) * noise_energy))
    return clean + scale * noise


def mix_utterance(primary, other, ratio_db, max_overlap=0.5):
    """Overlay a slice of `other` on `primary` at a target energy ratio."""
    length = primary.shape[-1]
    span = random.randint(1, max(1, int(length * max_overlap)))
    start1 = random.randint(0, max(0, length - span))
    start2 = random.randint(0, max(0, other.shape[-1] - span))
    span = min(span, length - start1, other.shape[-1] - start2)
    if span <= 0:
        return primary
    region1 = primary[..., start1:start1 + span]
    region2 = other[..., start2:start2 + span]
    energy1 = region1.pow(2).mean().clamp(min=1e-8)
    energy2 = region2.pow(2).mean().clamp(min=1e-8)
    scale = torch.sqrt(energy1 * (10 ** (ratio_db / 10)) / energy2)
    mixed = primary.clone()
    mixed[..., start1:start1 + span] = region1 + scale * region2
    return mixed


class DenoiseAugmentor:
    """Add noise and overlay other clips to a batch of waveforms."""

    def __init__(self, p_noise=0.25, p_mix=0.25, snr_noise=(-5, 20),
                 ratio_mix=(-5, 5), buffer_size=64):
        self.p_noise = p_noise
        self.p_mix = p_mix
        self.snr_noise = snr_noise
        self.ratio_mix = ratio_mix
        self.buffer_size = buffer_size
        self.buffer = []

    @property
    def enabled(self):
        """Return True when either effect can fire."""
        return self.p_noise > 0.0 or self.p_mix > 0.0

    def _remember(self, batch):
        """Keep a few clips from the batch to use as future noise."""
        for index in range(batch.shape[0]):
            clip = batch[index].detach().cpu().clone()
            if len(self.buffer) >= self.buffer_size:
                self.buffer.pop(0)
            self.buffer.append(clip)

    def _pick_other(self, length, device):
        """Take a random buffered clip and fit it to a target length."""
        other = random.choice(self.buffer).to(device)
        if other.shape[-1] < length:
            return F.pad(other, (0, length - other.shape[-1]))
        start = random.randint(0, other.shape[-1] - length)
        return other[..., start:start + length]

    def _augment_one(self, clip):
        """Apply at most one effect to a single waveform of shape (1, T)."""
        roll = random.random()
        length = clip.shape[-1]
        if roll < self.p_mix:
            other = self._pick_other(length, clip.device)
            return mix_utterance(clip, other, random.uniform(*self.ratio_mix))
        if roll < self.p_mix + self.p_noise:
            other = self._pick_other(length, clip.device)
            return mix_at_snr(clip, other, random.uniform(*self.snr_noise))
        return clip

    @torch.no_grad()
    def __call__(self, batch):
        """Return an augmented copy of a batch of shape (batch, 1, samples)."""
        if not self.enabled or len(self.buffer) < 4:
            self._remember(batch)
            return batch
        out = batch.clone()
        for index in range(batch.shape[0]):
            out[index] = self._augment_one(batch[index])
        self._remember(batch)
        return out
