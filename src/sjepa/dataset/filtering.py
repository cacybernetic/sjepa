"""Clean a dataset and remember the good files in a JSON cache.

Before training we scan every audio file once. A file is kept only when it can
be decoded and holds at least a tiny bit of sound. Bad files (corrupt, empty,
unreadable) are dropped. The good refs are written to a cache file next to the
dataset, so the next run loads the cache instead of scanning again.

Cache name follows the spec: a "train.zip" dataset gets a "train.cache.json"
file in the same folder. The cache also stores the duration of each clip, which
helps later steps without a second scan.
"""

from __future__ import annotations

import json
import os

from ..logging import get_logger
from .audio import AudioLoader
from .readers import ArchiveReader
from .sources import AudioRef, discover_audio

_LOGGER = get_logger()
# A clip shorter than this many seconds is treated as empty and dropped.
_MIN_SECONDS = 0.1
# Bump this when the cache format changes so old caches are rebuilt.
_CACHE_VERSION = 1


def cache_path_for(dataset_path):
    """Return the cache file path that belongs to a dataset path.

    Examples:
        "/data/train.zip" -> "/data/train.cache.json"
        "/data/train"     -> "/data/train.cache.json"
    """
    folder = os.path.dirname(os.path.abspath(dataset_path))
    base = os.path.basename(os.path.normpath(dataset_path))
    stem = base.split(".")[0] if "." in base else base
    return os.path.join(folder, f"{stem}.cache.json")


class CleanSample:
    """One kept sample: a ref plus its duration in seconds."""

    def __init__(self, ref, seconds):
        self.ref = ref
        self.seconds = seconds

    def to_dict(self):
        """Return a JSON-friendly dict for the cache file."""
        data = self.ref.to_dict()
        data["seconds"] = round(self.seconds, 4)
        return data

    @classmethod
    def from_dict(cls, data):
        """Rebuild a sample from a cache dict."""
        return cls(AudioRef.from_dict(data), float(data.get("seconds", 0.0)))


class DatasetCleaner:
    """Scan a dataset, drop bad files, and keep the good refs."""

    def __init__(self, sample_rate=16000, progress=True):
        self.loader = AudioLoader(sample_rate=sample_rate)
        self.progress = progress

    def _check_one(self, reader, ref):
        """Return a CleanSample when the file is valid, else None."""
        try:
            stream = reader.read_stream(ref)
            seconds = self.loader.probe_stream(stream)
        except (OSError, RuntimeError, ValueError):
            return None
        if seconds < _MIN_SECONDS:
            return None
        return CleanSample(ref, seconds)

    def _iter_refs(self, refs):
        """Wrap the refs with a progress bar when progress is on."""
        if not self.progress:
            return refs
        from tqdm import tqdm
        return tqdm(refs, desc="validating dataset", leave=True)

    def scan(self, dataset_path):
        """Scan a dataset path and return the list of kept samples."""
        refs = discover_audio(dataset_path)
        _LOGGER.info("Found {} audio files in {}", len(refs), dataset_path)
        reader = ArchiveReader()
        kept = []
        for ref in self._iter_refs(refs):
            sample = self._check_one(reader, ref)
            if sample is not None:
                kept.append(sample)
        reader.close()
        _LOGGER.info("Kept {}/{} valid files", len(kept), len(refs))
        return kept


class CleanCache:
    """Read and write the JSON cache of kept samples."""

    @staticmethod
    def save(path, samples):
        """Write the samples to a cache file."""
        payload = {
            "version": _CACHE_VERSION,
            "count": len(samples),
            "samples": [sample.to_dict() for sample in samples],
        }
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        _LOGGER.info("Wrote cache with {} samples to {}", len(samples), path)

    @staticmethod
    def load(path):
        """Read samples from a cache file, or None when it is missing or old."""
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("version") != _CACHE_VERSION:
            return None
        return [CleanSample.from_dict(item) for item in payload["samples"]]


def load_or_build_cache(dataset_path, sample_rate=16000, progress=True):
    """Load the cache if present, else scan the dataset and build it.

    Args:
        dataset_path: the folder or archive that holds the audio files.
        sample_rate: the rate used when probing files.
        progress: show a progress bar during a fresh scan.

    Returns:
        A list of `CleanSample`.
    """
    path = cache_path_for(dataset_path)
    cached = CleanCache.load(path)
    if cached is not None:
        _LOGGER.info("Loaded {} samples from cache {}", len(cached), path)
        return cached
    cleaner = DatasetCleaner(sample_rate=sample_rate, progress=progress)
    samples = cleaner.scan(dataset_path)
    CleanCache.save(path, samples)
    return samples
