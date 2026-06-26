"""Tests for the training-plateau fixes.

These cover the pieces the paper relies on to keep Phase 2 from stalling:
  * the online GMM re-seeds dead components instead of collapsing,
  * the online GMM is fed every micro-batch of a grad-accumulation window,
  * the effective-rank layer selector has no cold-start bias and reacts fast,
  * the optimizer keeps its state across the Phase 1 -> Phase 2 head swap,
  * the phase scheduler performs the transition and the masked-only switch.
"""

import os

import torch
import torch.nn as nn

from sjepa.gmm import DiagonalGMM, OnlineGMM
from sjepa.lr_shedulers import build_scheduler
from sjepa.optimizers import build_optimizer, replace_parameters
from sjepa.phases import PhaseScheduler
from sjepa.targets import Phase2TargetBuilder
from sjepa.trainer import LayerSelector
from sjepa.modules.losses import JEPAObjective


# ----- online GMM: dead-cluster re-seeding -----

def test_online_gmm_reseeds_dead_component():
    """A near-zero-weight component is re-seeded from a batch frame."""
    dim = 3
    means = torch.zeros(4, dim)
    variances = torch.ones(4, dim)
    weights = torch.tensor([0.0, 0.34, 0.33, 0.33])
    gmm = OnlineGMM(means, variances, weights, decay=0.999)
    features = torch.full((50, dim), 100.0) + torch.randn(50, dim)
    resp = gmm.posteriors(features)
    gmm.update(features, resp)
    # The dead component now has a usable weight and a mean near the batch.
    assert gmm.weights[0] > 0.0
    assert gmm.means[0].abs().sum() > 1.0
    assert torch.isfinite(gmm.means).all()


def test_online_gmm_variance_floor_blocks_spikes():
    """Variances never fall under the floor after an update."""
    dim = 2
    gmm = OnlineGMM(torch.zeros(3, dim), torch.ones(3, dim),
                    torch.full((3,), 1 / 3), decay=0.5)
    features = torch.zeros(20, dim)  # zero spread -> tiny batch variances
    gmm.update(features, gmm.posteriors(features))
    assert (gmm.variances >= 1e-4 - 1e-9).all()


# ----- target builder: accumulation across micro-batches -----

class _FakeEma:
    """Stand-in EMA encoder returning fixed-dimension random features."""

    def __init__(self, dim, frames=5):
        self.dim = dim
        self.frames = frames

    def extract_layer(self, waveform, layer, padding_mask=None):
        batch = waveform.shape[0]
        return torch.randn(batch, self.frames, self.dim)


def _toy_online_gmm(dim, k=4):
    return OnlineGMM(torch.randn(k, dim), torch.ones(k, dim),
                     torch.full((k,), 1.0 / k), decay=0.9)


def test_phase2_builder_accumulates_then_updates():
    """All accumulated micro-batches feed one GMM update, then clear."""
    dim = 6
    builder = Phase2TargetBuilder(_FakeEma(dim), _toy_online_gmm(dim), layer=0)
    waveform = torch.randn(2, 1, 1600)
    builder.build(waveform, accumulate=True)
    builder.build(waveform, accumulate=True)
    assert len(builder._feat_buffer) == 2
    builder.post_step()
    assert builder._feat_buffer == [] and builder._resp_buffer == []


def test_phase2_builder_no_leak_during_validation():
    """With accumulate=False (validation) nothing is buffered."""
    dim = 6
    builder = Phase2TargetBuilder(_FakeEma(dim), _toy_online_gmm(dim), layer=0)
    builder.build(torch.randn(2, 1, 1600), accumulate=False)
    assert builder._feat_buffer == []


# ----- layer selector: lazy init + decoupled decay -----

class _RankedEma:
    """EMA stand-in whose `best_layer` carries a high-rank feature matrix."""

    def __init__(self, num_layers, dim, best_layer):
        self.num_layers = num_layers
        self.dim = dim
        self.best_layer = best_layer

    def extract_layer(self, waveform, layer, padding_mask=None):
        if layer == self.best_layer:
            return torch.randn(1, 64, self.dim)
        base = torch.randn(1, 1, self.dim)
        return base.expand(1, 64, self.dim).clone()  # rank ~ 1


def test_layer_selector_picks_highest_rank_on_first_check():
    """No cold-start bias: the first check already selects the rich layer."""
    selector = LayerSelector(num_layers=4, decay=0.9)
    ema = _RankedEma(4, dim=16, best_layer=2)
    chosen = selector.select(ema, torch.randn(1, 1, 1600))
    assert chosen == 2
    assert all(score is not None for score in selector.scores)


def test_layer_selector_rejects_slow_per_step_decay():
    """A decay of 1.0 (frozen) is refused; moderate values are accepted."""
    LayerSelector(num_layers=3, decay=0.9)
    try:
        LayerSelector(num_layers=3, decay=1.0)
    except ValueError:
        return
    raise AssertionError("decay=1.0 should be rejected")


# ----- optimizer: parameter swap preserves state -----

class _TinyHeadModel(nn.Module):
    def __init__(self, out=4):
        super().__init__()
        self.encoder = nn.Linear(4, 4)
        self.cluster_head = nn.Linear(4, out)


def test_replace_parameters_preserves_encoder_state():
    """Swapping the head keeps encoder optimizer state and adds new params."""
    model = _TinyHeadModel(out=4)
    optimizer = build_optimizer(model, "adamw", lr=1e-3, weight_decay=0.1)
    out = model.cluster_head(model.encoder(torch.randn(8, 4)))
    out.pow(2).mean().backward()
    optimizer.step()
    enc_weight = model.encoder.weight
    assert enc_weight in optimizer.state  # state exists before the swap

    old_params = list(model.cluster_head.parameters())
    model.cluster_head = nn.Linear(4, 8)
    replace_parameters(optimizer, old_params,
                       list(model.cluster_head.named_parameters()),
                       weight_decay=0.1)

    grouped = {id(p) for g in optimizer.param_groups for p in g["params"]}
    assert id(enc_weight) in grouped                       # encoder kept
    assert enc_weight in optimizer.state                   # state preserved
    assert id(model.cluster_head.weight) in grouped        # new head added
    assert all(id(p) not in grouped for p in old_params)   # old head dropped
    # The new head trains without error.
    model.cluster_head(model.encoder(torch.randn(8, 4))).sum().backward()
    optimizer.step()


# ----- phase scheduler: transition + masked-only switch -----

class _FakeBuilderCfg:
    class optimizer:
        weight_decay = 1e-3


class _FakeBuilder:
    """Minimal builder exposing what PhaseScheduler.force_phase2 needs."""

    def __init__(self):
        self.device = "cpu"
        self.cfg = _FakeBuilderCfg()
        self.calls = []

    def _phase2_targets(self, model, loaders, num_clusters=None):
        self.calls.append(("seed", num_clusters))
        return "phase2-targets", {"ema": "ema", "selector": "selector"}

    def phase2_scaffold(self, model, num_clusters):
        self.calls.append(("scaffold", num_clusters))
        return "phase2-scaffold", {"ema": "ema", "selector": "selector"}


class _FakeTrainer:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.current_phase = 1
        self.targets = "phase1-targets"
        self.phase2 = None
        self.augmentor = object()

        class _Step:
            objective = JEPAObjective(use_visible_loss=True)
        self.step = _Step()


def test_phase_scheduler_transition_and_masked_only():
    """force_phase2 rebuilds the head into Phase 2; masked-only flips the loss."""
    from sjepa.model import build_model
    model = build_model("tiny", num_clusters=8)
    optimizer = build_optimizer(model, "adamw", lr=1e-3, weight_decay=1e-3)
    trainer = _FakeTrainer(model, optimizer)
    builder = _FakeBuilder()
    scheduler = PhaseScheduler(builder, loaders={"train": []},
                               start_epoch=2, masked_only_epoch=4,
                               phase2_clusters=16)

    # Before the start epoch: nothing happens.
    scheduler.on_epoch_start(trainer, epoch=0)
    assert trainer.current_phase == 1

    # At the start epoch: transition to Phase 2 with K=16.
    scheduler.on_epoch_start(trainer, epoch=2)
    assert trainer.current_phase == 2
    assert model.config.num_clusters == 16
    assert model.cluster_head.num_clusters == 16
    assert trainer.targets == "phase2-targets"
    assert builder.calls == [("seed", 16)]  # real transition fits the GMM
    grouped = {id(p) for g in optimizer.param_groups for p in g["params"]}
    assert id(model.cluster_head.net[-1].weight) in grouped

    # At the masked-only epoch: visible loss off, augmentation off.
    assert trainer.step.objective.use_visible_loss is True
    scheduler.on_epoch_start(trainer, epoch=4)
    assert trainer.step.objective.use_visible_loss is False
    assert trainer.augmentor is None


def test_force_phase2_resume_uses_scaffold_not_seed():
    """Restoring a Phase 2 checkpoint rebuilds scaffolding, never re-fits."""
    from sjepa.model import build_model
    model = build_model("tiny", num_clusters=8)
    optimizer = build_optimizer(model, "adamw", lr=1e-3, weight_decay=1e-3)
    trainer = _FakeTrainer(model, optimizer)
    builder = _FakeBuilder()
    scheduler = PhaseScheduler(builder, loaders={"train": []},
                               start_epoch=2, masked_only_epoch=4,
                               phase2_clusters=16)

    scheduler.force_phase2(trainer, seed=False)
    assert trainer.current_phase == 2
    assert model.config.num_clusters == 16
    assert trainer.targets == "phase2-scaffold"
    assert builder.calls == [("scaffold", 16)]  # no GMM fit on resume

    # Already in Phase 2: a later epoch never re-triggers the transition.
    scheduler.on_epoch_start(trainer, epoch=5)
    assert builder.calls == [("scaffold", 16)]


def test_lr_warm_restart_at_phase2():
    """The LR warms back up at the Phase 2 step instead of staying decayed."""
    model = nn.Linear(4, 4)
    optimizer = build_optimizer(model, "adamw", lr=1.0)
    total, p2 = 100, 50
    scheduler = build_scheduler(optimizer, "cosine", warmup_steps=5,
                                total_steps=total, min_ratio=0.1,
                                phase2_start_step=p2)
    lrs = []
    for _ in range(total):
        lrs.append(optimizer.param_groups[0]["lr"])
        scheduler.step()
    # Just before Phase 2 the cosine has decayed below peak...
    assert lrs[p2 - 1] < 0.7
    # ...then it warms back up to near the peak a few steps into Phase 2...
    assert max(lrs[p2:p2 + 10]) > 0.9
    # ...and never drops below the floor.
    assert min(lrs) >= 0.1 - 1e-6


def test_lr_no_restart_without_phase2_step():
    """Without a phase2 step the schedule is a plain warmup+cosine to the floor."""
    model = nn.Linear(4, 4)
    optimizer = build_optimizer(model, "adamw", lr=1.0)
    scheduler = build_scheduler(optimizer, "cosine", warmup_steps=5,
                                total_steps=100, min_ratio=0.1)
    lrs = []
    for _ in range(100):
        lrs.append(optimizer.param_groups[0]["lr"])
        scheduler.step()
    assert lrs[5] > lrs[0]            # warmed up
    assert lrs[-1] < lrs[5]           # then decayed
    assert min(lrs) >= 0.1 - 1e-6     # to the floor, not below


def test_history_plots_overlay_train_and_val(tmp_path):
    """Every metric now has both a train and a val series in the plots."""
    from sjepa.plotting import HistoryPlotter
    history = [
        {"epoch": 0, "train_loss": 6.0, "val_loss": 3.1, "train_kl": 3.0,
         "val_kl": 3.1, "train_top1": 0.10, "val_top1": 0.10,
         "train_entropy_bits": 4.0, "val_entropy_bits": 4.0},
        {"epoch": 1, "train_loss": 5.0, "val_loss": 2.8, "train_kl": 2.5,
         "val_kl": 2.8, "train_top1": 0.20, "val_top1": 0.18,
         "train_entropy_bits": 3.5, "val_entropy_bits": 3.6},
    ]
    plotter = HistoryPlotter(str(tmp_path))
    names = {os.path.basename(p) for p in plotter.plot(history)}
    assert {"history_kl.jpg", "history_top1.jpg",
            "history_entropy_bits.jpg", "history_loss.jpg"} <= names
    # Each metric carries a train AND a val curve.
    for metric in ("kl", "top1", "entropy_bits", "loss"):
        assert len(plotter._series(history, "train", metric)[1]) == 2
        assert len(plotter._series(history, "val", metric)[1]) == 2


def test_history_series_skips_non_numeric_cells():
    """Resuming an older CSV (empty cells for new metrics) does not break."""
    from sjepa.plotting import HistoryPlotter
    history = [
        {"epoch": 0, "train_top1": ""},      # old row, metric not recorded yet
        {"epoch": 1, "train_top1": 0.3},     # new row
    ]
    epochs, values = HistoryPlotter._series(history, "train", "top1")
    assert epochs == [1] and values == [0.3]


def test_phase2_scaffold_builds_without_data():
    """PipelineBuilder.phase2_scaffold needs no loader and gives K-shaped GMM."""
    from sjepa.assembly import PipelineBuilder
    from sjepa.config_schema import ExperimentConfig
    from sjepa.rundir import RunLayout
    config = ExperimentConfig.from_dict({"model": {"size": "tiny"},
                                         "gmm": {"ema_layer": 1}})
    builder = PipelineBuilder(config, RunLayout("/tmp/sjepa_scaffold_test"))
    from sjepa.model import build_model
    model = build_model("tiny", num_clusters=24)
    targets, bundle = builder.phase2_scaffold(model, num_clusters=24)
    assert targets.gmm.num_clusters == 24
    assert "ema" in bundle
