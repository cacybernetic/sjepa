"""Build or load the GMM that makes the soft targets.

Phase 1 needs a frozen GMM over MFCC features. We can load one from a saved file
or fit a fresh one on a sample of the training audio. Phase 2 needs an online
GMM over encoder features, seeded from a batch of EMA encoder features.

This file keeps the fitting logic out of the training loop. Each function has
one clear job and stops early when an input is wrong.
"""

from __future__ import annotations

import os

import torch

from .dataset.audio import AudioLoader
from .dataset.features import MfccExtractor
from .dataset.readers import ArchiveReader
from .gmm import DiagonalGMM, GMMFitter, OnlineGMM, ReservoirSampler
from .logging import get_logger

_LOGGER = get_logger()


class Phase1GmmProvider:
    """Load or fit the frozen MFCC GMM used in Phase 1."""

    def __init__(self, config, sample_rate=16000):
        self.config = config
        self.sample_rate = sample_rate
        self.extractor = MfccExtractor(sample_rate=sample_rate)

    def _collect_frames(self, samples, capacity):
        """Stream MFCC frames from a few clips into a reservoir."""
        reservoir = ReservoirSampler(capacity, self.extractor.dim)
        loader = AudioLoader(self.sample_rate, random_crop=False)
        reader = ArchiveReader()
        for sample in samples:
            if reservoir.filled >= capacity:
                break
            self._add_sample(reservoir, loader, reader, sample)
        reader.close()
        return reservoir.collected()

    def _add_sample(self, reservoir, loader, reader, sample):
        """Decode one clip and add its MFCC frames to the reservoir."""
        try:
            waveform = loader.load_stream(reader.read_stream(sample.ref))
        except (OSError, RuntimeError, ValueError):
            return
        features = self.extractor.extract(waveform)
        reservoir.add(features)

    def load(self, path, device):
        """Load a saved GMM from a .pt file."""
        state = torch.load(path, map_location=device, weights_only=False)
        _LOGGER.info("Loaded Phase 1 GMM from {}", path)
        return DiagonalGMM.from_state_dict(state, device=device)

    def fit(self, samples, device):
        """Fit a fresh GMM on a sample of the training clips."""
        _LOGGER.info("Fitting Phase 1 GMM on up to {} frames",
                     self.config.fit_frames)
        frames = self._collect_frames(samples, self.config.fit_frames)
        _LOGGER.info("Collected {} MFCC frames for fitting", frames.shape[0])
        fitter = GMMFitter(self.config.num_clusters, self.config.kmeans_iters,
                           self.config.em_iters)
        gmm = fitter.fit(frames.to(device))
        _LOGGER.info("Phase 1 GMM ready with K={}", gmm.num_clusters)
        return gmm

    def _collect_from_loader(self, loader, device):
        """Stream MFCC frames from a waveform loader into a reservoir."""
        reservoir = ReservoirSampler(self.config.fit_frames, self.extractor.dim)
        extractor = self.extractor.to(device)
        batches = loader.full_iter() if hasattr(loader, "full_iter") else loader
        for batch in batches:
            if batch is None or reservoir.filled >= self.config.fit_frames:
                break
            waveform = batch["waveform"].squeeze(1).to(device)
            features = extractor.extract(waveform).reshape(-1, self.extractor.dim)
            reservoir.add(features.cpu())
        return reservoir.collected()

    def fit_from_loader(self, loader, device):
        """Fit a GMM using waveforms from a data loader (raw or HDF5)."""
        _LOGGER.info("Fitting Phase 1 GMM from the train loader")
        frames = self._collect_from_loader(loader, device)
        _LOGGER.info("Collected {} MFCC frames for fitting", frames.shape[0])
        fitter = GMMFitter(self.config.num_clusters, self.config.kmeans_iters,
                           self.config.em_iters)
        gmm = fitter.fit(frames.to(device))
        _LOGGER.info("Phase 1 GMM ready with K={}", gmm.num_clusters)
        return gmm

    def provide(self, loader, device):
        """Return a GMM, loading from path when given, else fitting one."""
        path = self.config.path
        if path and os.path.exists(path):
            return self.load(path, device)
        return self.fit_from_loader(loader, device)


class OnlineGmmSeeder:
    """Seed the Phase 2 online GMM from EMA encoder features."""

    def __init__(self, config):
        self.config = config

    def _collect(self, ema_encoder, dataloader, layer, device, dim):
        """Collect encoder features from a few batches into a reservoir."""
        reservoir = ReservoirSampler(self.config.fit_frames, dim, device=device)
        batches = dataloader.full_iter() if hasattr(dataloader, "full_iter") \
            else dataloader
        for batch in batches:
            if batch is None or reservoir.filled >= self.config.fit_frames:
                break
            waveform = batch["waveform"].to(device)
            feats = ema_encoder.extract_layer(waveform, layer)
            reservoir.add(feats.reshape(-1, dim))
            if reservoir.filled >= self.config.fit_frames:
                break
        return reservoir.collected()

    def seed(self, ema_encoder, dataloader, layer, device, dim,
             num_clusters=None):
        """Fit and return an online GMM from collected features.

        Args:
            num_clusters: the number of components K. Defaults to the config
                value; the Phase 1 -> Phase 2 transition passes K=500 here.
        """
        num_clusters = num_clusters or self.config.num_clusters
        _LOGGER.info("Seeding Phase 2 online GMM at layer {} with K={}",
                     layer, num_clusters)
        frames = self._collect(ema_encoder, dataloader, layer, device, dim)
        fitter = GMMFitter(num_clusters, self.config.kmeans_iters,
                           self.config.em_iters)
        fitted = fitter.fit(frames)
        online = OnlineGMM.from_gmm(fitted, decay=self.config.param_decay)
        _LOGGER.info("Online GMM seeded with K={}", online.num_clusters)
        return online
