"""Build and read a ready-to-train dataset stored in one HDF5 file.

Decoding audio on the fly costs time. To save that time we can decode every
clip once, store the clean (and maybe augmented) waveform in an HDF5 file, and
read straight from the file during training.

Two classes live here, each with one job:

  * `Hdf5Builder`: write decoded waveforms into an HDF5 file.
  * `Hdf5AudioDataset`: read those waveforms back, like `AudioDataset`.

The file holds one float32 dataset per clip under the "clips" group. File-level
attributes record the sample rate and whether augmented copies are included.
"""

from __future__ import annotations

import h5py
import numpy as np
import torch

from ..logging import get_logger
from .audio import AudioLoader
from .readers import ArchiveReader

_LOGGER = get_logger()


class Hdf5Builder:
    """Decode clean samples and write their waveforms to an HDF5 file."""

    def __init__(self, sample_rate=16000, max_seconds=15.0, augmentor=None,
                 progress=True):
        self.sample_rate = sample_rate
        self.loader = AudioLoader(sample_rate=sample_rate,
                                  max_seconds=max_seconds, random_crop=False)
        self.augmentor = augmentor
        self.progress = progress

    def _decode(self, reader, ref):
        """Return one clean waveform tensor, or None when it fails."""
        try:
            stream = reader.read_stream(ref)
            return self.loader.load_stream(stream)
        except (OSError, RuntimeError, ValueError):
            return None

    def _augment(self, waveform):
        """Return an augmented copy of one waveform, or None when disabled."""
        if self.augmentor is None or not self.augmentor.enabled:
            return None
        batch = waveform.view(1, 1, -1)
        # Warm the augmentor buffer so the first clips can still be mixed.
        self.augmentor._remember(batch)
        return self.augmentor(batch).view(-1)

    def _iter(self, samples):
        """Wrap samples with a progress bar when progress is on."""
        if not self.progress:
            return enumerate(samples)
        from tqdm import tqdm
        return enumerate(tqdm(samples, desc="building hdf5", leave=True,
                              dynamic_ncols=True))

    def _write_clip(self, group, name, waveform):
        """Write one waveform as a compressed float32 dataset."""
        array = waveform.numpy().astype(np.float32)
        group.create_dataset(name, data=array, compression="gzip",
                             compression_opts=4)

    def build(self, samples, out_path):
        """Build the HDF5 file from a list of clean samples."""
        reader = ArchiveReader()
        written = 0
        with h5py.File(out_path, "w") as handle:
            clips = handle.create_group("clips")
            for index, sample in self._iter(samples):
                waveform = self._decode(reader, sample.ref)
                if waveform is None:
                    continue
                self._write_clip(clips, str(written), waveform)
                extra = self._augment(waveform)
                if extra is not None:
                    self._write_clip(clips, f"aug_{written}", extra)
                written += 1
            handle.attrs["sample_rate"] = self.sample_rate
            handle.attrs["count"] = written
            handle.attrs["augmented"] = self.augmentor is not None
        reader.close()
        _LOGGER.info("Wrote {} clips to {}", written, out_path)
        return written


class Hdf5AudioDataset(torch.utils.data.Dataset):
    """Read decoded waveforms back from an HDF5 file."""

    def __init__(self, path):
        self.path = path
        self._handle = None
        with h5py.File(path, "r") as handle:
            self.count = int(handle.attrs["count"])
        if self.count <= 0:
            raise ValueError(f"hdf5 file has no clips: {path}")

    def __len__(self):
        """Return the number of clips in the file."""
        return self.count

    def _file(self):
        """Open the HDF5 file lazily, one handle per worker."""
        if self._handle is None:
            self._handle = h5py.File(self.path, "r")
        return self._handle

    def __getitem__(self, index):
        """Return (waveform, index) for one clip."""
        clips = self._file()["clips"]
        array = clips[str(index)][:]
        return torch.from_numpy(array).float(), index
