"""Phase scheduling for a single continuous training run.

The paper runs S-JEPA as one optimization trajectory in two phases. Phase 1 is
a frozen MFCC GMM with the loss on masked and visible frames; partway through
Phase 2 the loss becomes masked-only and the denoising augmentation is turned
off. Phase 2 also swaps the targets to an online encoder-feature GMM (K=500)
fed by an EMA encoder, with adaptive layer selection.

This module performs those switches inside one run, so the anti-stall machinery
of Phase 2 actually engages instead of requiring two separate trainings. When
no transition epoch is configured the scheduler is inert and the run keeps the
phase it started in.

  * `Phase1To2Transition`: rebuild the cluster head for K=500, swap it into the
    optimizer, and install the EMA encoder + online GMM target builder.
  * `PhaseScheduler`: fire the transition and the masked-only switch on the
    configured epochs.
"""

from __future__ import annotations

from .logging import banner, get_logger
from .optimizers import replace_parameters

_LOGGER = get_logger()


class PhaseScheduler:
    """Fire the Phase 1 -> Phase 2 switches at the configured epochs."""

    def __init__(self, builder, loaders, start_epoch, masked_only_epoch,
                 phase2_clusters):
        self.builder = builder
        self.loaders = loaders
        self.start_epoch = start_epoch
        self.masked_only_epoch = masked_only_epoch
        self.phase2_clusters = phase2_clusters
        self._masked_only_done = False

    def on_epoch_start(self, trainer, epoch):
        """Apply any phase switch that is due at the start of `epoch`."""
        if (self.start_epoch is not None and self.start_epoch >= 0
                and trainer.current_phase < 2 and epoch >= self.start_epoch):
            self.force_phase2(trainer)
        if (not self._masked_only_done and self.masked_only_epoch is not None
                and self.masked_only_epoch >= 0
                and epoch >= self.masked_only_epoch):
            self._to_masked_only(trainer)

    def force_phase2(self, trainer, seed=True):
        """Switch the trainer to Phase 2 (rebuild head, install online GMM).

        Args:
            seed: when True (a real mid-run transition), the online GMM is fit
                from data. When False (restoring a Phase 2 checkpoint), only the
                scaffolding is rebuilt so the saved weights load — the EMA and
                GMM are then overwritten by `Trainer._restore_phase2`, so fitting
                them here would be wasted work and would touch the data loader
                for nothing. This is NOT a return to Phase 1: no Phase 1 epoch is
                replayed, training continues from the checkpoint's epoch.
        """
        if trainer.current_phase >= 2:
            return
        model = trainer.model
        old_params = list(model.cluster_head.parameters())
        model.set_num_clusters(self.phase2_clusters)
        model.cluster_head.to(self.builder.device)
        replace_parameters(trainer.optimizer, old_params,
                            list(model.cluster_head.named_parameters()),
                            self.builder.cfg.optimizer.weight_decay)
        if seed:
            _LOGGER.info(banner("Phase 1 -> Phase 2 transition", color="yellow"))
            targets, phase2 = self.builder._phase2_targets(
                model, self.loaders, num_clusters=self.phase2_clusters)
            _LOGGER.info("Phase 2 active: K={}, online encoder GMM installed",
                         self.phase2_clusters)
        else:
            _LOGGER.info("Rebuilding Phase 2 scaffolding for resume "
                         "(K={}, GMM/EMA restored from checkpoint)",
                         self.phase2_clusters)
            targets, phase2 = self.builder.phase2_scaffold(
                model, self.phase2_clusters)
        trainer.targets = targets
        trainer.phase2 = phase2
        trainer.current_phase = 2

    def _to_masked_only(self, trainer):
        """Drop the visible-frame loss and turn denoising augmentation off."""
        trainer.step.objective.use_visible_loss = False
        trainer.augmentor = None
        self._masked_only_done = True
        _LOGGER.info("Switched to masked-only loss with augmentation off")
