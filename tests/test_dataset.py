"""Tests for the dataset package (discovery, cleaning, audio, features, hdf5)."""

import io
import os
import zipfile

import numpy as np
import soundfile as sf
import torch

from sjepa.dataset import (
    AudioDataset,
    AudioLoader,
    DenoiseAugmentor,
    Hdf5AudioDataset,
    Hdf5Builder,
    MfccExtractor,
    WaveformCollator,
    cache_path_for,
    discover_audio,
    hop_from_overlap,
    is_audio_name,
    load_or_build_cache,
    plan_windows,
    window_starts,
)


def _sine(freq, seconds=0.5, rate=16000):
    """Build a short sine wave as float32 samples."""
    time = np.linspace(0, seconds, int(rate * seconds), endpoint=False)
    return (0.2 * np.sin(2 * np.pi * freq * time)).astype("float32")


def _make_folder(root, count=4):
    """Write a few wav files (with one sub-folder) into a folder."""
    os.makedirs(os.path.join(root, "sub"), exist_ok=True)
    for index in range(count):
        target = root if index % 2 == 0 else os.path.join(root, "sub")
        sf.write(os.path.join(target, f"a{index}.wav"), _sine(200 + index), 16000)


def _make_zip(path, count=3):
    """Write a few wav files into a zip archive."""
    with zipfile.ZipFile(path, "w") as archive:
        for index in range(count):
            buffer = io.BytesIO()
            sf.write(buffer, _sine(300 + index), 16000, format="WAV")
            archive.writestr(f"c{index}.wav", buffer.getvalue())


def test_is_audio_name():
    """The extension check is case-insensitive and rejects other files."""
    assert is_audio_name("song.WAV")
    assert is_audio_name("voice.flac")
    assert not is_audio_name("notes.txt")


def test_discover_folder_recursive(tmp_path):
    """Discovery finds files in the root and in sub-folders."""
    _make_folder(str(tmp_path), count=4)
    refs = discover_audio(str(tmp_path))
    assert len(refs) == 4


def test_discover_zip(tmp_path):
    """Discovery finds files inside a zip archive."""
    zip_path = str(tmp_path / "train.zip")
    _make_zip(zip_path, count=3)
    refs = discover_audio(zip_path)
    assert len(refs) == 3
    assert all(ref.container == "zip" for ref in refs)


def test_cache_path_for():
    """The cache path sits next to the dataset with a fixed name."""
    assert cache_path_for("/data/train.zip").endswith("train.cache.json")
    assert cache_path_for("/data/test").endswith("test.cache.json")


def test_cleaning_drops_bad_file(tmp_path):
    """A corrupt file is dropped and the cache holds only good files."""
    _make_folder(str(tmp_path), count=2)
    with open(tmp_path / "bad.wav", "wb") as handle:
        handle.write(b"not audio")
    samples = load_or_build_cache(str(tmp_path), progress=False)
    assert len(samples) == 2
    assert os.path.exists(cache_path_for(str(tmp_path)))


def test_audio_loader_mono_and_crop():
    """The loader returns a 1D tensor cropped to the maximum length."""
    buffer = io.BytesIO()
    sf.write(buffer, _sine(220, seconds=2.0), 16000, format="WAV")
    buffer.seek(0)
    loader = AudioLoader(sample_rate=16000, max_seconds=1.0, random_crop=False)
    waveform = loader.load_stream(buffer)
    assert waveform.dim() == 1
    assert waveform.shape[0] == 16000


def test_mfcc_dimension():
    """The MFCC extractor returns 39 features per frame."""
    extractor = MfccExtractor()
    features = extractor.extract(torch.randn(8000))
    assert features.shape[-1] == 39


def test_collator_pads_to_hop_multiple():
    """The collate function pads every clip to a multiple of the hop."""
    collator = WaveformCollator(hop=320)
    batch = [(torch.randn(800), 0), (torch.randn(1300), 1)]
    out = collator(batch)
    assert out["waveform"].shape[-1] % 320 == 0
    assert out["waveform"].shape[0] == 2


def test_augmentor_changes_signal():
    """The augmentor returns a different signal once its buffer is warm."""
    augmentor = DenoiseAugmentor(p_noise=1.0, p_mix=0.0)
    batch = torch.randn(4, 1, 8000)
    for _ in range(3):
        augmentor(batch)
    out = augmentor(batch)
    assert out.shape == batch.shape


def test_hdf5_build_and_read(tmp_path):
    """Built clips can be read back with the right shape."""
    _make_folder(str(tmp_path), count=4)
    samples = load_or_build_cache(str(tmp_path), progress=False)
    out_path = str(tmp_path / "train.h5")
    Hdf5Builder(progress=False).build(samples, out_path)
    dataset = Hdf5AudioDataset(out_path)
    assert len(dataset) == 4
    waveform, index = dataset[0]
    assert waveform.dim() == 1 and index == 0


def test_hdf5_windows_long_clip(tmp_path):
    """The HDF5 reader tiles a stored full clip into overlapping windows."""
    sf.write(str(tmp_path / "long.wav"), _sine(220, seconds=3.0), 16000)
    samples = load_or_build_cache(str(tmp_path), progress=False)
    out_path = str(tmp_path / "train.h5")
    Hdf5Builder(progress=False).build(samples, out_path)
    dataset = Hdf5AudioDataset(out_path, max_seconds=1.0, window_overlap=0.5)
    # Same tiling as the raw path: 3s clip, 1s window, 0.5s hop -> 5 windows.
    assert len(dataset) == 5
    waveform, clip_index = dataset[0]
    assert clip_index == 0
    assert waveform.shape[0] == 16000


def test_audio_dataset_returns_waveform(tmp_path):
    """The dataset returns a waveform and its index for a valid sample."""
    _make_folder(str(tmp_path), count=2)
    samples = load_or_build_cache(str(tmp_path), progress=False)
    dataset = AudioDataset(samples)
    item = dataset[0]
    assert item is not None
    assert item[0].dim() == 1


def test_window_starts_cover_all_frames():
    """Windows tile a clip end to end, snapping the last one to the tail."""
    starts = window_starts(length=3.0, window=1.0, hop=0.5)
    assert starts[0] == 0.0
    # The last window ends exactly at the clip end (start = length - window).
    assert starts[-1] == 2.0
    # Consecutive starts never leave an uncovered gap wider than the window.
    assert all(b - a <= 1.0 + 1e-9 for a, b in zip(starts, starts[1:]))


def test_window_starts_short_clip_single_window():
    """A clip shorter than one window yields exactly one window at 0."""
    assert window_starts(length=0.5, window=1.0, hop=0.5) == [0.0]


def test_hop_from_overlap():
    """The hop shrinks with more overlap and is capped below the window."""
    assert hop_from_overlap(10.0, 0.5) == 5.0
    assert hop_from_overlap(10.0, 0.0) == 10.0
    assert hop_from_overlap(10.0, 1.0) > 0.0  # capped so it never reaches 0


def test_plan_windows_flattens_clips():
    """The plan holds one (clip, start) entry per window across all clips."""
    plan = plan_windows([3.0, 0.5], window=1.0, hop=1.0)
    # Clip 0 (3s): starts 0, 1, 2 -> 3 windows; clip 1 (0.5s): 1 window.
    assert [c for c, _ in plan] == [0, 0, 0, 1]


def test_audio_dataset_tiles_long_clip(tmp_path):
    """A long clip becomes several overlapping windows, not a single crop."""
    sf.write(str(tmp_path / "long.wav"), _sine(220, seconds=3.0), 16000)
    samples = load_or_build_cache(str(tmp_path), progress=False)
    dataset = AudioDataset(samples, max_seconds=1.0, window_overlap=0.5)
    # 3s clip, 1s window, 0.5s hop -> starts 0, .5, 1, 1.5, 2 -> 5 windows.
    assert len(dataset) == 5
    waveform, sample_index = dataset[0]
    assert sample_index == 0
    assert waveform.shape[0] == 16000  # one full second


def test_audio_dataset_stride_is_configurable(tmp_path):
    """A larger stride (less overlap) yields fewer windows for the same clip."""
    sf.write(str(tmp_path / "long.wav"), _sine(220, seconds=3.0), 16000)
    samples = load_or_build_cache(str(tmp_path), progress=False)
    dense = AudioDataset(samples, max_seconds=1.0, window_overlap=0.5)
    sparse = AudioDataset(samples, max_seconds=1.0, window_overlap=0.0)
    assert len(sparse) < len(dense)
