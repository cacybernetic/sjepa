"""Dataset building blocks for S-JEPA.

This package finds audio files, cleans them, decodes them, builds features and
augmentations, and serves them as batches. Every file holds one job:

  * `sources`: find audio files in a folder, zip, or tar.
  * `readers`: read the raw bytes of one referenced file.
  * `audio`: decode bytes into a clean mono waveform.
  * `features`: build 39-dim MFCC features for the Phase 1 GMM.
  * `filtering`: drop bad files and cache the good ones.
  * `augment`: add noise and overlay other clips.
  * `dataset`: a PyTorch dataset plus a collate function.
  * `hdf5`: build and read a ready-to-train HDF5 dataset.
"""

from .sources import AudioRef, discover_audio, is_audio_name, AUDIO_EXTENSIONS
from .readers import ArchiveReader
from .audio import AudioLoader
from .features import MfccExtractor
from .filtering import (
    CleanSample,
    CleanCache,
    DatasetCleaner,
    cache_path_for,
    load_or_build_cache,
)
from .augment import DenoiseAugmentor, mix_at_snr, mix_utterance
from .dataset import AudioDataset, WaveformCollator
from .hdf5 import Hdf5Builder, Hdf5AudioDataset

__all__ = [
    "AudioRef",
    "discover_audio",
    "is_audio_name",
    "AUDIO_EXTENSIONS",
    "ArchiveReader",
    "AudioLoader",
    "MfccExtractor",
    "CleanSample",
    "CleanCache",
    "DatasetCleaner",
    "cache_path_for",
    "load_or_build_cache",
    "DenoiseAugmentor",
    "mix_at_snr",
    "mix_utterance",
    "AudioDataset",
    "WaveformCollator",
    "Hdf5Builder",
    "Hdf5AudioDataset",
]
