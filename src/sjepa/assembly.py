"""Assemble every part into a ready `Trainer`.

The entry point script stays short: it loads the config, picks the run folder,
sets up logging, and then calls `PipelineBuilder(...).build()`. This file holds
the wiring logic, split into small methods so each one is easy to read.

It also applies the priority rule from the spec: a usable checkpoint is loaded
first; only when there is none do we warm start from an existing weight file.
"""

from __future__ import annotations

import math
import os
import random

import torch

from .checkpointing import CheckpointManager, WeightSaver
from .data_module import DataModule
from .dataset.augment import DenoiseAugmentor
from .gmm_builder import OnlineGmmSeeder, Phase1GmmProvider
from .history import HistoryRecorder
from .logging import banner, get_logger
from .lossfn import build_objective
from .lr_shedulers import build_scheduler
from .model import build_model
from .modules.ema import SwitchedEmaScheduler
from .optimizers import build_optimizer
from .phases import PhaseScheduler
from .plotting import HistoryPlotter
from .step import ForwardStep, MaskBuilder
from .summary import log_model_summary
from .targets import Phase1TargetBuilder, Phase2TargetBuilder
from .trainer import LayerSelector, Trainer

_LOGGER = get_logger()


def resolve_device(name):
    """Map a config device name to a real torch device, with a safe fallback."""
    key = (name or "cpu").lower()
    wants_gpu = key in ("cuda", "rocm", "gpu")
    if wants_gpu and torch.cuda.is_available():
        return "cuda"
    if wants_gpu:
        _LOGGER.warning("GPU asked but not available; using CPU instead")
    return "cpu"


class PipelineBuilder:
    """Build a `Trainer` from a config and a run layout."""

    def __init__(self, config, layout, hop=320):
        self.cfg = config
        self.layout = layout
        self.hop = hop
        self.device = resolve_device(config.device)

    def _seed(self):
        """Seed every RNG stream used in training, not only torch.

        The augmentor and the audio random crop draw from Python's `random`,
        and third-party pieces may draw from numpy; leaving them unseeded made
        runs unrepeatable even with a fixed torch seed.
        """
        torch.manual_seed(self.cfg.seed)
        random.seed(self.cfg.seed)
        try:
            import numpy
            numpy.random.seed(self.cfg.seed % (2 ** 32))
        except ImportError:
            pass

    def _required_frames(self):
        """Return how many frames the longest allowed clip can produce.

        The predictor has one learned position per frame. It must have enough
        positions for the longest clip in the dataset, or it will fail. We add a
        small margin for any rounding in the CNN frontend.
        """
        samples = int(self.cfg.dataset.max_seconds * self.cfg.dataset.sample_rate)
        return samples // self.hop + 2

    def _build_model(self):
        """Build the model at the chosen size and number of clusters."""
        overrides = dict(self.cfg.model.overrides)
        overrides.setdefault("num_clusters", self.cfg.gmm.num_clusters)
        # Make sure the predictor has enough positions for the longest clip,
        # even when a small size preset asks for fewer frames.
        overrides["max_frames"] = max(int(overrides.get("max_frames", 0)),
                                      self._required_frames())
        model = build_model(self.cfg.model.size, **overrides)
        model = model.to(self.device)
        log_model_summary(model, model.config, self.device)
        return model

    def _build_augmentor(self):
        """Build the denoising augmentor from the config."""
        aug = self.cfg.dataset.augment
        if not aug.enabled:
            return None
        return DenoiseAugmentor(p_noise=aug.p_noise, p_mix=aug.p_mix,
                                snr_noise=tuple(aug.snr_noise),
                                ratio_mix=tuple(aug.ratio_mix))

    def _phase1_targets(self, loaders):
        """Build the Phase 1 target builder (frozen MFCC GMM)."""
        provider = Phase1GmmProvider(self.cfg.gmm, self.cfg.dataset.sample_rate)
        gmm = provider.provide(loaders["train"], self.device)
        builder = Phase1TargetBuilder(gmm, self.cfg.dataset.sample_rate,
                                      self.device)
        return builder, None

    def _resolve_layer(self, model):
        """Turn a possibly negative layer index into a real index."""
        layer = self.cfg.gmm.ema_layer
        if layer < 0:
            layer = model.config.num_layers + layer
        return layer

    def _phase2_targets(self, model, loaders, num_clusters=None):
        """Build the Phase 2 target builder (EMA encoder, online GMM)."""
        scheduler = SwitchedEmaScheduler(self.cfg.gmm.ema_decay_fast,
                                         self.cfg.gmm.ema_decay_slow,
                                         self.cfg.gmm.ema_switch_every)
        ema = model.build_ema_encoder(scheduler).to(self.device)
        layer = self._resolve_layer(model)
        seeder = OnlineGmmSeeder(self.cfg.gmm)
        gmm = seeder.seed(ema, loaders["train"], layer, self.device,
                          model.config.hidden_dim, num_clusters=num_clusters)
        builder = Phase2TargetBuilder(ema, gmm, layer, self.device)
        return builder, self._phase2_bundle(model, ema)

    def phase2_scaffold(self, model, num_clusters):
        """Build empty Phase 2 scaffolding without fitting the online GMM.

        Used when restoring a Phase 2 checkpoint: the EMA encoder and a
        placeholder online GMM of the right shape are created so the saved
        weights and GMM state can be loaded on top. No data is read and no
        k-means/EM is run here.
        """
        scheduler = SwitchedEmaScheduler(self.cfg.gmm.ema_decay_fast,
                                         self.cfg.gmm.ema_decay_slow,
                                         self.cfg.gmm.ema_switch_every)
        ema = model.build_ema_encoder(scheduler).to(self.device)
        layer = self._resolve_layer(model)
        dim = model.config.hidden_dim
        means = torch.zeros(num_clusters, dim, device=self.device)
        variances = torch.ones(num_clusters, dim, device=self.device)
        weights = torch.full((num_clusters,), 1.0 / num_clusters,
                             device=self.device)
        from .gmm import OnlineGMM
        gmm = OnlineGMM(means, variances, weights,
                        decay=self.cfg.gmm.param_decay)
        builder = Phase2TargetBuilder(ema, gmm, layer, self.device)
        return builder, self._phase2_bundle(model, ema)

    def _phase2_bundle(self, model, ema):
        """Build the Phase 2 runtime bundle (EMA encoder and layer selector)."""
        selector = None
        if self.cfg.gmm.auto_layer:
            selector = LayerSelector(model.config.num_layers,
                                     self.cfg.gmm.erank_decay)
        return {"ema": ema, "selector": selector}

    def _build_targets(self, model, loaders):
        """Pick the target builder for the configured phase."""
        if self.cfg.train.phase >= 2 or self.cfg.gmm.online:
            return self._phase2_targets(model, loaders)
        return self._phase1_targets(loaders)

    def _steps_per_epoch(self, loaders):
        """Return the number of optimizer steps in one epoch."""
        batches = max(1, len(loaders["train"]))
        return max(1, math.ceil(batches / self.cfg.train.grad_accum))

    def _build_optim(self, model, loaders):
        """Build the optimizer and the learning rate scheduler."""
        optim = self.cfg.optimizer
        optimizer = build_optimizer(model, optim.name, optim.lr,
                                    optim.weight_decay, tuple(optim.betas))
        steps_per_epoch = self._steps_per_epoch(loaders)
        total = steps_per_epoch * self.cfg.train.epochs
        scheduler = build_scheduler(optimizer, self.cfg.scheduler.kind,
                                    self.cfg.scheduler.warmup_steps, total,
                                    self.cfg.scheduler.min_ratio,
                                    self._phase2_start_step(steps_per_epoch),
                                    self.cfg.scheduler.phase2_lr_ratio)
        return optimizer, scheduler

    def _phase2_start_step(self, steps_per_epoch):
        """Optimizer step of the Phase 2 transition, for the LR warm restart.

        Returns None (no restart) unless this is a Phase 1 run scheduled to
        switch to Phase 2 and the warm restart is enabled.
        """
        if not self.cfg.scheduler.rewarm_on_phase2:
            return None
        if self.cfg.train.phase >= 2 or self.cfg.gmm.online:
            return None
        start_epoch = self.cfg.train.phase2_start_epoch
        if start_epoch is None or start_epoch < 0:
            return None
        return start_epoch * steps_per_epoch

    def _build_phase_scheduler(self, loaders):
        """Build the in-run phase scheduler when a transition is configured.

        It is active only for a run that starts in Phase 1 and asks for a
        transition (`train.phase2_start_epoch >= 0`) or for the masked-only
        switch (`train.masked_only_epoch >= 0`). A run that starts directly in
        Phase 2 keeps its target builder and needs no scheduler.
        """
        if self.cfg.train.phase >= 2 or self.cfg.gmm.online:
            return None
        if (self.cfg.train.phase2_start_epoch < 0
                and self.cfg.train.masked_only_epoch < 0):
            return None
        return PhaseScheduler(self, loaders,
                              self.cfg.train.phase2_start_epoch,
                              self.cfg.train.masked_only_epoch,
                              self.cfg.gmm.num_clusters_phase2)

    def _autocast_dtype(self):
        """Map the configured precision to an autocast dtype (or None)."""
        precision = (self.cfg.train.precision or "fp32").lower()
        if precision in ("fp32", "float32", "off"):
            return None
        if precision in ("bf16", "bfloat16"):
            if self.device != "cuda":
                _LOGGER.warning("precision=bf16 asked on CPU; keeping fp32")
                return None
            return torch.bfloat16
        raise ValueError(f"unknown precision '{precision}' (fp32 or bf16)")

    def _build_step(self, model):
        """Build the per-batch forward step and the loss objective."""
        objective = build_objective(self.cfg.train.use_visible_loss)
        masker = MaskBuilder(self.cfg.masking.mask_ratio,
                             self.cfg.masking.mask_length)
        return ForwardStep(model, objective, masker, self.hop, self.device,
                           autocast_dtype=self._autocast_dtype())

    def _build_io(self):
        """Build the checkpoint manager, weight saver, history, and plotter."""
        ckpt_dir = self.cfg.checkpoint.dir or self.layout.checkpoints_dir
        ckpt = CheckpointManager(ckpt_dir, self.cfg.checkpoint.max_checkpoint)
        weights = WeightSaver(self.layout.weights_dir)
        history = HistoryRecorder(self.layout.history_csv)
        plotter = HistoryPlotter(self.layout.plots_dir)
        return ckpt, weights, history, plotter

    def _resume_or_init(self, trainer):
        """Load a checkpoint if present, else warm start from weights.

        Checkpoints are tried newest first: if the latest file is unreadable
        (e.g. the crash happened while it was written by an older, non-atomic
        version), the run falls back to the previous one instead of dying.
        """
        if self.cfg.checkpoint.resume:
            for path in trainer.ckpt.paths_newest_first():
                try:
                    trainer.load_checkpoint(path)
                    return
                except (RuntimeError, EOFError, KeyError, OSError) as exc:
                    _LOGGER.warning("Checkpoint {} unreadable ({}); trying the "
                                    "previous one", path, exc)
        if self.cfg.init_weights and os.path.exists(self.cfg.init_weights):
            trainer.load_weights(self.cfg.init_weights)

    def build(self):
        """Build and return a ready `Trainer`."""
        self._seed()
        model = self._build_model()
        loaders = DataModule(self.cfg, self.hop,
                             pin_memory=(self.device == "cuda")).build()
        targets, phase2 = self._build_targets(model, loaders)
        optimizer, scheduler = self._build_optim(model, loaders)
        step = self._build_step(model)
        ckpt, weights, history, plotter = self._build_io()
        phase_scheduler = self._build_phase_scheduler(loaders)
        trainer = Trainer(model, step, targets, optimizer, scheduler, loaders,
                          self.cfg, self.layout, ckpt, weights, history,
                          plotter, augmentor=self._build_augmentor(),
                          phase2=phase2, phase_scheduler=phase_scheduler)
        self._resume_or_init(trainer)
        _log_run_summary(self.cfg, loaders, self.device, model)
        return trainer


def _log_run_summary(config, loaders, device, model):
    """Log a compact run summary for full traceability."""
    counts = model.count_parameters()
    _LOGGER.info(banner("Run summary", color="green"))
    _LOGGER.info("  device          = {}", device)
    _LOGGER.info("  epochs          = {}", config.train.epochs)
    _LOGGER.info("  batch_size      = {} x grad_accum={}", config.train.batch_size,
                 config.train.grad_accum)
    _LOGGER.info("  optimizer       = {} lr={}", config.optimizer.name,
                 config.optimizer.lr)
    _LOGGER.info("  scheduler       = {}", config.scheduler.kind)
    _LOGGER.info("  best criterion  = {}", config.best.metric)
    _LOGGER.info("  phase           = {}", config.train.phase)
    _LOGGER.info("  model params (M)= {}", round(counts["total_M"], 2))
    _LOGGER.info("  data sizes      = {}", loaders["sizes"])
