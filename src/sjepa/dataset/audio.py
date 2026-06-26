"""Turn audio bytes into a clean mono waveform tensor.

The loader does four small jobs:

  1. decode the bytes to samples (soundfile reads wav, flac, ogg, mp3);
  2. mix many channels down to one (mono);
  3. resample to the target sample rate when needed;
  4. crop a long clip to a maximum length (random window in training).

The output is a 1D float32 tensor of samples in the range about -1 to 1.
"""

from __future__ import annotations

import random

import soundfile as sf
import torch
import torchaudio.functional as AF


class AudioLoader:
    """Decode and normalize one audio clip into a mono waveform tensor."""

    def __init__(self, sample_rate=16000, max_seconds=15.0, random_crop=True):
        if sample_rate <= 0:
            raise ValueError("sample_rate must be > 0")
        self.sample_rate = sample_rate
        self.max_samples = int(sample_rate * max_seconds)
        self.random_crop = random_crop

    @staticmethod
    def _decode(stream):
        """Read samples and the source rate from an audio stream."""
        data, source_rate = sf.read(stream, dtype="float32", always_2d=True)
        waveform = torch.from_numpy(data).t().contiguous()
        return waveform, source_rate

    @staticmethod
    def _to_mono(waveform):
        """Average all channels into a single channel."""
        if waveform.shape[0] == 1:
            return waveform
        return waveform.mean(dim=0, keepdim=True)

    def _resample(self, waveform, source_rate):
        """Resample the waveform to the target rate when they differ."""
        if source_rate == self.sample_rate:
            return waveform
        return AF.resample(waveform, source_rate, self.sample_rate)

    def _crop(self, waveform):
        """Cut a long clip down to the maximum number of samples."""
        length = waveform.shape[-1]
        if length <= self.max_samples:
            return waveform
        room = length - self.max_samples
        start = random.randint(0, room) if self.random_crop else 0
        return waveform[..., start:start + self.max_samples]

    def load_stream(self, stream):
        """Decode and normalize from an in-memory stream.

        Returns:
            A 1D float32 tensor of samples.
        """
        waveform, source_rate = self._decode(stream)
        waveform = self._to_mono(waveform)
        waveform = self._resample(waveform, source_rate)
        waveform = self._crop(waveform)
        return waveform.squeeze(0).float()

    def probe_stream(self, stream):
        """Return the duration in seconds without loading all samples.

        This is used by the cleaning step to drop empty or unreadable files.
        """
        info = sf.info(stream)
        if info.frames <= 0 or info.samplerate <= 0:
            return 0.0
        return info.frames / float(info.samplerate)
