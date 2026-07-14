"""Build the train, validation, and test data loaders.

The rules come from the spec:

  * the train loader reads the training dataset;
  * the validation loader reads a fraction `val_prob` of the test dataset;
  * the test loader reads the whole test dataset for the final evaluation.

Both raw audio (read on the fly) and a prebuilt HDF5 file are supported. The
user switches between them with `dataset.use_hdf5` in the config.
"""

from __future__ import annotations

import torch
from torch.utils.data import Subset

from .dataloader import ResumableDataLoader
from .dataset.dataset import AudioDataset, WaveformCollator
from .dataset.filtering import load_or_build_cache
from .dataset.hdf5 import Hdf5AudioDataset
from .logging import get_logger

_LOGGER = get_logger()


def _cap(samples, limit):
    """Keep at most `limit` samples, or all when the limit is missing."""
    if limit is None or limit <= 0:
        return samples
    return samples[:limit]


class DataModule:
    """Create the three data loaders from one config."""

    def __init__(self, config, hop=320, pin_memory=False):
        self.cfg = config
        self.data_cfg = config.dataset
        self.collator = WaveformCollator(hop=hop)
        self.pin_memory = pin_memory
        self._val_indices = None

    def _hdf5_dataset(self, path):
        """Open an HDF5 dataset with the configured windowing."""
        return Hdf5AudioDataset(path, max_seconds=self.data_cfg.max_seconds,
                                window_overlap=self.data_cfg.window_overlap)

    def _raw_dataset(self, path, max_samples):
        """Build an on-the-fly audio dataset from a folder or archive."""
        samples = load_or_build_cache(
            path, sample_rate=self.data_cfg.sample_rate,
            progress=self.data_cfg.validate)
        samples = _cap(samples, max_samples)
        return AudioDataset(samples, sample_rate=self.data_cfg.sample_rate,
                            max_seconds=self.data_cfg.max_seconds,
                            window_overlap=self.data_cfg.window_overlap)

    def _train_dataset(self):
        """Build the training dataset (raw or HDF5)."""
        if self.data_cfg.use_hdf5:
            return self._hdf5_dataset(self.data_cfg.train_h5)
        return self._raw_dataset(self.data_cfg.train_path,
                                 self.data_cfg.max_train_samples)

    def _test_dataset(self):
        """Build the test dataset (raw or HDF5)."""
        if self.data_cfg.use_hdf5:
            return self._hdf5_dataset(self.data_cfg.test_h5)
        return self._raw_dataset(self.data_cfg.test_path,
                                 self.data_cfg.max_test_samples)

    def _split_test(self, test_dataset):
        """Split the test dataset into a validation part and a held-out part.

        The best checkpoint is selected on the validation metrics; if the final
        evaluation reuses those same clips, the reported test score is biased
        by the selection. With `val_disjoint` (the default) the two subsets do
        not overlap. The old behaviour (evaluate on the full test set, val
        included) stays available with `val_disjoint: false`.
        """
        total = len(test_dataset)
        keep = max(1, int(total * self.data_cfg.val_prob))
        generator = torch.Generator().manual_seed(self.cfg.seed)
        order = torch.randperm(total, generator=generator).tolist()
        self._val_indices = sorted(order[:keep])
        val_dataset = Subset(test_dataset, self._val_indices)
        if not self.data_cfg.val_disjoint:
            return val_dataset, test_dataset
        if keep >= total:
            _LOGGER.warning("val_prob={} leaves no held-out test clip; the "
                            "final evaluation will reuse the validation set",
                            self.data_cfg.val_prob)
            return val_dataset, test_dataset
        held_out = Subset(test_dataset, sorted(order[keep:]))
        return val_dataset, held_out

    def _loader(self, dataset, shuffle):
        """Wrap a dataset in a resumable DataLoader for in-epoch checkpointing."""
        return ResumableDataLoader(
            dataset, batch_size=self.cfg.train.batch_size, shuffle=shuffle,
            seed=self.cfg.seed, num_workers=self.data_cfg.num_workers,
            collate_fn=self.collator, pin_memory=self.pin_memory,
            drop_last=False)

    def build(self):
        """Return a dict with the train, val, and test loaders and sizes."""
        train_dataset = self._train_dataset()
        test_dataset = self._test_dataset()
        val_dataset, held_out = self._split_test(test_dataset)
        _LOGGER.info("Data sizes: train={} val={} test={}",
                     len(train_dataset), len(val_dataset), len(held_out))
        return {
            "train": self._loader(train_dataset, True),
            "val": self._loader(val_dataset, False),
            "test": self._loader(held_out, False),
            "sizes": {"train": len(train_dataset), "val": len(val_dataset),
                      "test": len(held_out)},
        }
