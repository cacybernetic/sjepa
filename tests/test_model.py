"""Unit tests for the complete S-JEPA model in src/sjepa/model.py.

These tests check the forward pass, mask alignment, gradient flow, the size
builders, and the helper methods (parameter count, cluster head rebuild, EMA
encoder, feature extraction).
"""

import torch

from sjepa import SJEPA, SJEPAConfig, build_config, build_model, SJEPA_SIZES
from sjepa.model import SJEPAOutput
from sjepa.modules import JEPAObjective


def test_forward_returns_aligned_output(tiny_model, batch):
    out = tiny_model(batch["waveform"], batch["mask"],
                     padding_mask=batch["padding_mask"])
    assert isinstance(out, SJEPAOutput)
    length = out.encoder_output.shape[1]
    # Every returned tensor must share the same frame length.
    assert out.logits_masked.shape[1] == length
    assert out.mask.shape[1] == length
    assert out.padding_mask.shape[1] == length
    assert out.logits_masked.shape[-1] == tiny_model.config.num_clusters


def test_backward_updates_parameters(tiny_model, batch):
    tiny_model.train()
    out = tiny_model(batch["waveform"], batch["mask"],
                     padding_mask=batch["padding_mask"])
    targets = torch.softmax(
        torch.randn(out.mask.shape[0], out.mask.shape[1],
                    tiny_model.config.num_clusters), dim=-1)
    objective = JEPAObjective(use_visible_loss=True)
    result = objective(out.logits_masked, out.logits_visible, targets,
                       out.mask, out.padding_mask)
    result["loss"].backward()
    # At least one parameter must have a gradient after backward.
    grads = [p.grad for p in tiny_model.parameters() if p.grad is not None]
    assert len(grads) > 0


def test_extract_features_runs_without_mask(tiny_model, batch):
    feat = tiny_model.extract_features(batch["waveform"], layer_index=-1,
                                       padding_mask=batch["padding_mask"])
    assert feat.shape[0] == batch["waveform"].shape[0]
    assert feat.shape[-1] == tiny_model.config.hidden_dim


def test_count_parameters_has_all_parts(tiny_model):
    counts = tiny_model.count_parameters()
    for key in ["encoder_M", "predictor_M", "cluster_head_M", "total_M"]:
        assert key in counts
    assert counts["total_M"] >= counts["encoder_M"]


def test_set_num_clusters_rebuilds_head(tiny_model):
    tiny_model.set_num_clusters(500)
    assert tiny_model.config.num_clusters == 500
    assert tiny_model.cluster_head.num_clusters == 500


def test_build_ema_encoder_is_frozen(tiny_model):
    ema = tiny_model.build_ema_encoder()
    grads_off = [not p.requires_grad for p in ema.encoder.parameters()]
    assert all(grads_off)


def test_base_size_matches_paper_param_count():
    model = build_model("base")
    encoder_params = model.count_parameters()["encoder_M"]
    # The paper reports a 51.8M parameter encoder for the 6-layer backbone.
    assert abs(encoder_params - 51.8) < 1.0


def test_all_sizes_build():
    for size in SJEPA_SIZES:
        config = build_config(size)
        assert isinstance(config, SJEPAConfig)
        model = SJEPA(config)
        assert model.count_parameters()["total_M"] > 0


def test_unknown_size_raises():
    try:
        build_config("huge")
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_phase_two_override_sets_clusters():
    model = SJEPA.from_size("tiny", num_clusters=500)
    assert model.config.num_clusters == 500
