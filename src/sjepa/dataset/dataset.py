"""PyTorch dataset and collate function for raw audio clips.

The dataset reads one clean sample at a time and returns its mono waveform. The
archive handles are opened lazily inside each worker, so the dataset is safe to
use with several DataLoader workers.

The collate function pads a batch of waveforms to a shared length that is a
multiple of the frame hop. It also returns the real frame length of each clip,
which the masking and padding steps need.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .audio import AudioLoader
from .readers import ArchiveReader


class AudioDataset(Dataset):
    """Return mono waveforms for a list of clean samples."""

    def __init__(self, samples, sample_rate=16000, max_seconds=15.0,
                 random_crop=True):
        if not samples:
            raise ValueError("the sample list is empty")
        self.samples = samples
        self.loader = AudioLoader(sample_rate=sample_rate,
                                  max_seconds=max_seconds,
                                  random_crop=random_crop)
        self._reader = None

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.samples)

    def _reader_handle(self):
        """Return a worker-local archive reader, building it on first use."""
        if self._reader is None:
            self._reader = ArchiveReader()
        return self._reader

    def __getitem__(self, index):
        """Return (waveform, index) for one sample, or None when it fails."""
        ref = self.samples[index].ref
        try:
            stream = self._reader_handle().read_stream(ref)
            waveform = self.loader.load_stream(stream)
        except (OSError, RuntimeError, ValueError):
            return None
        if waveform.numel() == 0:
            return None
        return waveform, index


def _round_up(value, multiple):
    """Round a length up to the next multiple of the frame hop."""
    return ((value + multiple - 1) // multiple) * multiple


class WaveformCollator:
    """Pad and stack a batch of waveforms to a shared length."""

    def __init__(self, hop=320, min_frames=4):
        self.hop = hop
        self.min_samples = min_frames * hop

    def _padded_length(self, lengths):
        """Pick a padded sample length that is a multiple of the hop."""
        longest = max(max(lengths), self.min_samples)
        return _round_up(longest, self.hop)

    def __call__(self, batch):
        """Collate a batch into a dict, dropping items that failed to load."""
        items = [item for item in batch if item is not None]
        if not items:
            return None
        waveforms, indices = zip(*items)
        lengths = [wave.shape[-1] for wave in waveforms]
        target = self._padded_length(lengths)
        padded = [F.pad(wave, (0, target - wave.shape[-1])) for wave in waveforms]
        stacked = torch.stack(padded).unsqueeze(1)
        frame_lengths = [length // self.hop for length in lengths]
        return {
            "waveform": stacked,
            "frame_lengths": frame_lengths,
            "indices": list(indices),
            "seconds": sum(lengths) / float(self.hop * 50),
        }
