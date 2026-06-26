"""Build a ready-to-train HDF5 dataset from a config.

Usage:
    buildsjepa -c cpu/configs/hdf5.yaml

It cleans the train and test datasets (using the JSON cache), decodes every
clip once, and writes the waveforms into "train.h5" and "test.h5". When the
augmentation is enabled, an augmented copy of each clip is stored too. Training
can then read from the HDF5 files and skip the on-the-fly decoding.
"""

from __future__ import annotations

import os

from ..dataset.augment import DenoiseAugmentor
from ..dataset.filtering import load_or_build_cache
from ..dataset.hdf5 import Hdf5Builder
from ..logging import banner, get_logger
from .common import parse_config_arg, setup_run

_LOGGER = get_logger()


def _augmentor(config):
    """Build the augmentor when augmentation is enabled, else None."""
    aug = config.dataset.augment
    if not aug.enabled:
        return None
    return DenoiseAugmentor(p_noise=aug.p_noise, p_mix=aug.p_mix,
                            snr_noise=tuple(aug.snr_noise),
                            ratio_mix=tuple(aug.ratio_mix))


def _build_one(config, source_path, out_path):
    """Clean one dataset and write its clips into an HDF5 file."""
    _LOGGER.info(banner(f"Building {out_path}"))
    samples = load_or_build_cache(source_path,
                                  sample_rate=config.dataset.sample_rate,
                                  progress=config.dataset.validate)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    builder = Hdf5Builder(sample_rate=config.dataset.sample_rate,
                          max_seconds=config.dataset.max_seconds,
                          augmentor=_augmentor(config))
    return builder.build(samples, out_path)


def run(config_path):
    """Build the train and test HDF5 files from a config path."""
    config, layout, _ = setup_run(config_path, "train")
    data = config.dataset
    _build_one(config, data.train_path, data.train_h5)
    _build_one(config, data.test_path, data.test_h5)
    _LOGGER.info("HDF5 datasets ready: {} and {}", data.train_h5, data.test_h5)


def main():
    """Console entry point for the buildsjepa command."""
    config_path = parse_config_arg("Build an HDF5 dataset for S-JEPA")
    run(config_path)


if __name__ == "__main__":
    main()
