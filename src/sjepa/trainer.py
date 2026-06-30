"""The training engine that ties every part together.

The `Trainer` runs the epoch loop. For each epoch it trains, validates on a
fraction of the test set, records the history, draws the plots, saves a full
checkpoint, and saves the best and last weights. It supports gradient
accumulation, gradient clipping, a learning rate scheduler, resume from a
checkpoint, and the Phase 2 extras (EMA encoder, online GMM, adaptive layer).

The class delegates the small jobs to helpers so every method stays short:

  * `BestTracker`: remember the best validation score.
  * `LayerSelector`: pick the GMM input layer by effective rank (Phase 2).
  * `ForwardStep`: forward pass and loss for one batch.
"""

from __future__ import annotations

import torch

from .logging import banner, colorize, get_logger
from .metrics import MetricGroup, effective_rank
from .metrics.base import AverageMeter
from .progress import EpochProgress, StepProgress

_LOGGER = get_logger()


class BestTracker:
    """Remember the best validation value for the chosen metric."""

    def __init__(self, metric, mode):
        if mode not in ("min", "max"):
            raise ValueError("mode must be 'min' or 'max'")
        self.metric = metric
        self.mode = mode
        self.best = None

    def is_better(self, value):
        """Return True when a value beats the current best."""
        if self.best is None:
            return True
        if self.mode == "min":
            return value < self.best
        return value > self.best

    def update(self, metrics):
        """Check the tracked metric and store it when it improves."""
        value = metrics.get(self.metric)
        if value is None or not self.is_better(value):
            return False, self.best
        self.best = value
        return True, self.best


class LayerSelector:
    """Pick the encoder layer with the highest effective rank (Phase 2).

    The score of each layer is an EMA of its measured effective rank across the
    periodic checks. The decay is the smoothing applied *per check* (not per
    training step), so it must be a moderate value (e.g. 0.9), not the slow
    per-step GMM decay; using the latter would make the score lag for millions
    of steps and the adaptive layer switch would never fire on time. Scores are
    lazily initialized to the first measurement to avoid a cold-start bias.
    """

    def __init__(self, num_layers, decay=0.9):
        if not 0.0 <= decay < 1.0:
            raise ValueError("decay must be in [0, 1)")
        self.num_layers = num_layers
        self.decay = decay
        self.scores = [None] * num_layers

    @torch.no_grad()
    def _layer_rank(self, ema_encoder, waveform, layer):
        """Return the effective rank of one EMA encoder layer."""
        feats = ema_encoder.extract_layer(waveform, layer)
        return effective_rank(feats)

    @torch.no_grad()
    def select(self, ema_encoder, waveform):
        """Update the smoothed scores and return the best layer index."""
        best_layer, best_score = 0, -1.0
        for layer in range(self.num_layers):
            rank = self._layer_rank(ema_encoder, waveform, layer)
            if self.scores[layer] is None:
                self.scores[layer] = rank
            else:
                self.scores[layer] = (self.decay * self.scores[layer]
                                      + (1.0 - self.decay) * rank)
            if self.scores[layer] > best_score:
                best_score, best_layer = self.scores[layer], layer
        return best_layer

    def state_dict(self):
        """Return the smoothed per-layer scores so resume does not start cold."""
        return {"scores": list(self.scores)}

    def load_state_dict(self, state):
        """Restore the smoothed scores when the layer count still matches."""
        scores = state.get("scores")
        if scores is not None and len(scores) == self.num_layers:
            self.scores = list(scores)


class Trainer:
    """Run the full training loop for S-JEPA."""

    def __init__(self, model, forward_step, target_builder, optimizer,
                 scheduler, loaders, config, layout, checkpoint_manager,
                 weight_saver, history, plotter, augmentor=None, phase2=None,
                 phase_scheduler=None):
        self.model = model
        self.step = forward_step
        self.targets = target_builder
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loaders = loaders
        self.cfg = config
        self.layout = layout
        self.ckpt = checkpoint_manager
        self.weights = weight_saver
        self.history = history
        self.plotter = plotter
        self.augmentor = augmentor
        self.phase2 = phase2
        self.phase_scheduler = phase_scheduler
        self.current_phase = 2 if phase2 is not None else 1
        self.metrics = MetricGroup()
        self.train_metrics = MetricGroup()
        self.best = BestTracker(config.best.metric,
                                MetricGroup.mode_of(config.best.metric))
        self.start_epoch = 0
        self.global_step = 0
        self.epoch_seconds = []
        self._pending = False
        # In-epoch checkpointing: save every `ckpt_step` optimizer steps (train)
        # or processed batches (val/test). `_ckpt_seq` is the strictly increasing
        # save sequence that orders checkpoints; `_resume` carries the cursor read
        # from a checkpoint so the interrupted pass resumes at the right batch.
        self.ckpt_step = config.checkpoint.ckpt_step
        self._ckpt_seq = 0
        self._resume = None

    # ----- gradient accumulation and optimizer step -----

    def _backward(self, loss):
        """Scale the loss for accumulation and run backward."""
        (loss / self.cfg.train.grad_accum).backward()
        self._pending = True

    def _clip(self):
        """Clip the gradient norm and return its value."""
        return float(torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.cfg.train.grad_clip_norm))

    def _optimizer_step(self):
        """Apply one optimizer step and the Phase 2 online updates."""
        grad_norm = self._clip()
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        self._pending = False
        self.global_step += 1
        self._phase2_updates()
        return grad_norm

    def _phase2_updates(self):
        """Update the EMA encoder and the online GMM after a step."""
        if self.phase2 is None:
            return
        self.phase2["ema"].update(self.model.encoder, self.global_step)
        self.targets.post_step()

    def _maybe_select_layer(self, waveform):
        """Run adaptive layer selection on the configured cadence."""
        if self.phase2 is None or self.phase2.get("selector") is None:
            return
        cadence = self.cfg.gmm.layer_check_every
        if cadence <= 0 or self.global_step % cadence != 0:
            return
        best = self.phase2["selector"].select(self.phase2["ema"], waveform)
        if best != self.targets.layer:
            _LOGGER.info("Auto layer switch {} -> {}", self.targets.layer, best)
            self.targets.set_layer(best)

    # ----- one training epoch -----

    def _augmentor(self):
        """Return the augmentor when it is enabled, else None."""
        aug = self.augmentor
        return aug if aug is not None and aug.enabled else None

    def _log_step(self, epoch, opt_step, total, grad_norm, result):
        """Write a periodic training step line with loguru."""
        if opt_step % self.cfg.train.log_every != 0:
            return
        comp = result["components"]
        _LOGGER.debug(
            "epoch {}/{} | step {}/{} loss={:.4f} kl_masked={:.4f} "
            "kl_visible={:.4f} grad_norm={:.3f}", epoch, self.cfg.train.epochs,
            opt_step, total, float(result["loss"].detach()),
            comp["loss_masked"], comp["loss_visible"], grad_norm)

    def _train_epoch(self, epoch, resume=None):
        """Train for one epoch and return the average train metrics.

        The same metrics computed at validation (kl, top1, entropy_bits) are also
        computed here on the training batches, so the history holds a `train_X`
        and a `val_X` series for every metric and the plots overlay the two
        curves to expose overfitting.

        When `resume` is given the epoch resumes mid-way: the data loader is
        already positioned (its state was restored), and the running loss meter,
        the train metrics, and the optimizer-step count are restored so the
        averages and the grad-accumulation alignment continue seamlessly.
        """
        self.model.train()
        loader = self.loaders["train"]
        meters = {"loss": AverageMeter()}
        if resume is None:
            loader.set_epoch(epoch)
            self.train_metrics.reset()
            opt_steps = 0
        else:
            meters["loss"].load_state_dict(resume["meters"]["loss"])
            self.train_metrics.load_state_dict(resume["meters"]["metrics"])
            opt_steps = resume["meters"]["opt_steps"]
        total = max(1, len(loader) - loader.batches_done)
        bar = StepProgress(total, "train", epoch, self.cfg.train.epochs)
        augmentor = self._augmentor()
        for index, batch in enumerate(loader):
            if batch is None:
                continue
            opt_steps = self._train_batch(epoch, batch, augmentor, index,
                                          loader, meters, bar, opt_steps)
        opt_steps = self._flush(epoch, len(loader), opt_steps)
        bar.close()
        return {"loss": meters["loss"].average(), **self.train_metrics.compute()}

    def _train_batch(self, epoch, batch, augmentor, index, loader, meters, bar,
                     opt_steps):
        """Run one micro-batch and step the optimizer when accumulation is full."""
        result = self.step.run(batch, self.targets, augmentor, accumulate=True)
        self._backward(result["loss"])
        meters["loss"].update(float(result["loss"].detach()))
        self.train_metrics.update(result["logits_masked"], result["targets"],
                                  result["selection"])
        bar.update({"loss": meters["loss"].average(),
                    "kl": self.train_metrics.compute()["kl"]})
        if (index + 1) % self.cfg.train.grad_accum == 0:
            grad_norm = self._optimizer_step()
            opt_steps += 1
            self._maybe_select_layer(batch["waveform"].to(self.step.device))
            total = max(1, len(loader) // self.cfg.train.grad_accum)
            self._log_step(epoch, opt_steps, total, grad_norm, result)
            self._maybe_checkpoint_train(epoch, meters["loss"], opt_steps)
        return opt_steps

    def _flush(self, epoch, num_batches, opt_steps):
        """Step the optimizer for any leftover accumulation at epoch end."""
        if not self._pending:
            return opt_steps
        grad_norm = self._optimizer_step()
        opt_steps += 1
        _LOGGER.debug("epoch {}/{} | flushed leftover accumulation "
                      "(grad_norm={:.3f})", epoch, self.cfg.train.epochs,
                      grad_norm)
        return opt_steps

    # ----- validation -----

    @torch.no_grad()
    def _validate(self, loader, stage, epoch, resume=None, extra_payload=None):
        """Run a validation or evaluation pass and return the metric values.

        The total objective loss is averaged here too (same definition as in
        training) so the `loss` plot overlays comparable train and val curves,
        alongside kl, top1, and entropy_bits.

        With `resume` the pass continues from a mid-pass checkpoint: the loader
        is already positioned and the running meters are restored. `extra_payload`
        holds fields stitched into any in-epoch checkpoint written here (for the
        validation pass, the already-computed train metrics of this epoch).
        """
        self.model.eval()
        loss_meter = AverageMeter()
        if resume is None:
            loader.set_epoch(epoch)
            self.metrics.reset()
        else:
            loss_meter.load_state_dict(resume["meters"]["loss"])
            self.metrics.load_state_dict(resume["meters"]["metrics"])
        total = max(1, len(loader) - loader.batches_done)
        bar = StepProgress(total, stage, epoch, self.cfg.train.epochs)
        for batch in loader:
            if batch is not None:
                result = self.step.run(batch, self.targets, augmentor=None)
                loss_meter.update(float(result["loss"].detach()))
                self.metrics.update(result["logits_masked"], result["targets"],
                                    result["selection"])
                bar.update(self.metrics.compute())
            # Checked on every batch (even a dropped None one) so the loader
            # position is captured at the right cadence and never skipped.
            self._maybe_checkpoint_eval(epoch, stage, loader, loss_meter,
                                        extra_payload)
        bar.close()
        values = self.metrics.compute()
        values["loss"] = loss_meter.average()
        return values

    # ----- checkpoint and resume -----

    def _loader_states(self):
        """Capture the resumable state of every data loader."""
        states = {}
        for name, loader in self.loaders.items():
            if hasattr(loader, "state_dict"):
                states[name] = loader.state_dict()
        return states

    def _checkpoint_payload(self, epoch, stage, extra=None):
        """Build the full state payload for one checkpoint.

        `stage` records where the epoch was interrupted ("train", "val", "test",
        or "done" for an end-of-epoch checkpoint) so the run resumes at the right
        place. `extra` carries the running meters and, for the validation pass,
        the already-computed train metrics of this epoch.
        """
        payload = {
            "epoch": epoch,
            "global_step": self.global_step,
            "ckpt_seq": self._ckpt_seq,
            "phase": self.current_phase,
            "num_clusters": self.model.config.num_clusters,
            "use_visible_loss": self.step.objective.use_visible_loss,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "best": self.best.best,
            "epoch_seconds": self.epoch_seconds,
            "cursor": {"stage": stage},
            "loaders": self._loader_states(),
        }
        if self.phase2 is not None:
            payload["ema"] = self.phase2["ema"].state_dict()
            payload["gmm"] = self.targets.gmm.state_dict()
            payload["layer"] = self.targets.layer
            if self.phase2.get("selector") is not None:
                payload["selector"] = self.phase2["selector"].state_dict()
        if extra:
            payload.update(extra)
        return payload

    def _save_checkpoint(self, epoch, stage, extra=None):
        """Write one checkpoint with a fresh, strictly increasing sequence."""
        self._ckpt_seq += 1
        payload = self._checkpoint_payload(epoch, stage, extra)
        self.ckpt.save(payload, epoch, self.global_step, self._ckpt_seq)

    def _maybe_checkpoint_train(self, epoch, loss_meter, opt_steps):
        """Write an in-epoch checkpoint every `ckpt_step` optimizer steps."""
        if self.ckpt_step <= 0 or self.global_step % self.ckpt_step != 0:
            return
        extra = {"meters": {"loss": loss_meter.state_dict(),
                            "metrics": self.train_metrics.state_dict(),
                            "opt_steps": opt_steps}}
        self._save_checkpoint(epoch, "train", extra)

    def _maybe_checkpoint_eval(self, epoch, stage, loader, loss_meter,
                               extra_payload):
        """Write an in-epoch checkpoint every `ckpt_step` processed batches."""
        if self.ckpt_step <= 0 or loader.batches_done % self.ckpt_step != 0:
            return
        extra = {"meters": {"loss": loss_meter.state_dict(),
                            "metrics": self.metrics.state_dict()}}
        if extra_payload:
            extra.update(extra_payload)
        self._save_checkpoint(epoch, stage, extra)

    def _restore_loaders(self, loader_states):
        """Restore the data loader positions from a checkpoint."""
        for name, state in (loader_states or {}).items():
            loader = self.loaders.get(name)
            if loader is not None and hasattr(loader, "load_state_dict"):
                loader.load_state_dict(state)

    def load_checkpoint(self, path):
        """Restore the full training state from a checkpoint file."""
        state = self.ckpt.load(path)
        # A checkpoint saved during Phase 2 has a rebuilt cluster head and the
        # Phase 2 bundle. Re-apply the transition before loading the weights so
        # the model shapes and the optimizer parameter groups line up.
        if (state.get("phase", 1) >= 2 and self.current_phase < 2
                and self.phase_scheduler is not None):
            self.phase_scheduler.force_phase2(self, seed=False)
        if not state.get("use_visible_loss", True):
            # The masked-only switch already happened before this checkpoint.
            # Restore that state and stop the scheduler from re-firing it.
            self.step.objective.use_visible_loss = False
            self.augmentor = None
            if self.phase_scheduler is not None:
                self.phase_scheduler._masked_only_done = True
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.global_step = state.get("global_step", 0)
        self._ckpt_seq = state.get("ckpt_seq", 0)
        self.best.best = state.get("best")
        self.epoch_seconds = state.get("epoch_seconds", [])
        self._restore_phase2(state)
        self._restore_cursor(state, path)

    def _restore_cursor(self, state, path):
        """Set the resume point from the checkpoint's in-epoch cursor.

        An end-of-epoch checkpoint ("done", or an old checkpoint without a
        cursor) resumes at the next epoch. A mid-epoch checkpoint ("train" or
        "val") resumes inside the same epoch; a "test" checkpoint resumes the
        final evaluation after the epoch loop. In every mid-pass case the loader
        positions and the running meters are restored before the pass continues.
        """
        epoch = state["epoch"]
        stage = state.get("cursor", {}).get("stage", "done")
        if stage == "done":
            self.start_epoch = epoch + 1
            self._resume = None
        elif stage == "test":
            self.start_epoch = self.cfg.train.epochs
            self._resume = state
        else:  # "train" or "val": resume inside this epoch
            self.start_epoch = epoch
            self._resume = state
        _LOGGER.info(colorize("Resumed from {} at epoch {} (stage {})", "yellow"),
                     path, self.start_epoch, stage)

    def _restore_phase2(self, state):
        """Restore the EMA encoder, online GMM, and layer (Phase 2 only)."""
        if self.phase2 is None or "ema" not in state:
            return
        self.phase2["ema"].load_state_dict(state["ema"])
        from .gmm import OnlineGMM
        self.targets.gmm = OnlineGMM.from_state_dict(
            state["gmm"], device=self.step.device)
        self.targets.set_layer(state.get("layer", self.targets.layer))
        if "selector" in state and self.phase2.get("selector") is not None:
            self.phase2["selector"].load_state_dict(state["selector"])

    def load_weights(self, path):
        """Warm start the model from a saved weight file."""
        state = torch.load(path, map_location=self.step.device,
                           weights_only=False)
        weights = state.get("model", state)
        self.model.load_state_dict(weights, strict=False)
        _LOGGER.info("Loaded start weights from {}", path)

    # ----- the epoch loop -----

    def _log_epoch_metrics(self, epoch, train_metrics, val_metrics):
        """Log every metric of the epoch at INFO level after validation."""
        _LOGGER.info(banner(f"Epoch {epoch}/{self.cfg.train.epochs} metrics"))
        for name, value in train_metrics.items():
            _LOGGER.info("  train {:<14} = {:.4f}", name, value)
        for name, value in val_metrics.items():
            _LOGGER.info("  val   {:<14} = {:.4f}", name, value)

    def _record_epoch(self, epoch, train_metrics, val_metrics):
        """Append the history row and redraw the plots."""
        row = {"epoch": epoch, "lr": self.optimizer.param_groups[0]["lr"]}
        for name, value in train_metrics.items():
            row[f"train_{name}"] = value
        for name, value in val_metrics.items():
            row[f"val_{name}"] = value
        self.history.add(row)
        self.plotter.plot(self.history.history())
        return row

    def _save_epoch_outputs(self, epoch, val_metrics):
        """Save the checkpoint, the last weights, and the best weights.

        The best tracker is updated first so the end-of-epoch checkpoint records
        this epoch's best, not a stale value from the previous epoch.
        """
        improved, value = self.best.update(val_metrics)
        self._save_checkpoint(epoch, "done")
        self.weights.save(self.model, "last.pt", {"epoch": epoch})
        if improved:
            self.weights.save(self.model, "best.pt",
                              {"epoch": epoch, self.best.metric: value})
            _LOGGER.info(colorize("New best {} = {:.4f}", "green"),
                         self.best.metric, value)

    def _run_epoch(self, epoch, epoch_bar):
        """Run one full epoch: train, validate, record, save.

        When `self._resume` points into this epoch the interrupted pass continues
        from its checkpoint: loader positions are restored first, the phase
        transition is skipped (it already fired and its state was restored), and
        the train pass is either resumed or skipped depending on the saved stage.
        """
        import time
        start = time.time()
        resume = self._resume
        self._resume = None
        _LOGGER.info(banner(f"Starting epoch {epoch}/{self.cfg.train.epochs}"))
        if resume is not None:
            self._restore_loaders(resume.get("loaders"))
        elif self.phase_scheduler is not None:
            self.phase_scheduler.on_epoch_start(self, epoch)
        stage = resume["cursor"]["stage"] if resume else "train"
        if stage == "train":
            train_metrics = self._train_epoch(epoch, resume)
            val_resume = None
        else:  # resuming inside the validation pass
            train_metrics = resume["train_metrics"]
            val_resume = resume
        val_metrics = self._validate(self.loaders["val"], "val", epoch,
                                     resume=val_resume,
                                     extra_payload={"train_metrics": train_metrics})
        self._record_epoch(epoch, train_metrics, val_metrics)
        self._save_epoch_outputs(epoch, val_metrics)
        self.epoch_seconds.append(time.time() - start)
        self._log_epoch_metrics(epoch, train_metrics, val_metrics)
        epoch_bar.update(self.best.best,
                         self.optimizer.param_groups[0]["lr"],
                         sum(self.epoch_seconds) / len(self.epoch_seconds))

    def run(self):
        """Run training across all epochs and return the best score."""
        epoch_bar = EpochProgress(self.cfg.train.epochs, self.start_epoch)
        for epoch in range(self.start_epoch, self.cfg.train.epochs):
            self._run_epoch(epoch, epoch_bar)
        epoch_bar.close()
        _LOGGER.info("Training done. Best {} = {}", self.best.metric,
                     self.best.best)
        return self.best.best

    def final_evaluate(self):
        """Evaluate the model on the whole test set after training.

        A long evaluation can itself be checkpointed and resumed: when a "test"
        cursor was restored, the pass continues from the saved batch and meters.
        """
        _LOGGER.info(banner("Final evaluation on the full test set"))
        last_epoch = max(0, self.cfg.train.epochs - 1)
        resume = None
        if self._resume and self._resume["cursor"]["stage"] == "test":
            resume = self._resume
            self._resume = None
            self._restore_loaders(resume.get("loaders"))
        values = self._validate(self.loaders["test"], "test", last_epoch,
                                resume=resume)
        for name, value in values.items():
            _LOGGER.info("  test {:<14} = {:.4f}", name, value)
        return values
