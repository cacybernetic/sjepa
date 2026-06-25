"""Performance and precision tests for the important modules.

These tests measure two things the coding style asks for:
  * execution time: a forward pass must finish in a reasonable time on a CPU;
  * precision: the model must be able to lower the loss on a fixed batch, which
    shows the gradients are correct and learning works.

The numbers are printed so the user can read the metrics in the test log. The
time limits are loose so the tests do not fail on a slow machine.
"""

import time

import torch

from sjepa import build_model
from sjepa.config import SJEPAConfig
from sjepa.model import SJEPA
from sjepa.modules import BlockMaskGenerator, JEPAObjective, build_padding_mask


def _make_batch(config, batch_size=2, samples=8000):
    """Build one fixed batch for the timing and learning tests."""
    waveform = torch.randn(batch_size, 1, samples)
    num_frames = samples // config.hop
    frame_lengths = [num_frames] * batch_size
    padding = build_padding_mask(batch_size, num_frames, frame_lengths,
                                 waveform.device)
    mask = BlockMaskGenerator().generate(batch_size, num_frames, frame_lengths,
                                         waveform.device)
    return waveform, mask, padding


def test_forward_pass_speed():
    """The forward pass should be fast on a small model."""
    model = build_model("tiny")
    model.eval()
    waveform, mask, padding = _make_batch(model.config)
    # One warm-up pass so we do not time the first slow call.
    with torch.no_grad():
        model(waveform, mask, padding_mask=padding)
    start = time.perf_counter()
    runs = 5
    with torch.no_grad():
        for _ in range(runs):
            model(waveform, mask, padding_mask=padding)
    elapsed = (time.perf_counter() - start) / runs
    print(f"[perf] tiny forward pass: {elapsed * 1000:.1f} ms per batch")
    # A loose limit. The tiny model must stay well under one second per batch.
    assert elapsed < 1.0


def test_model_can_lower_loss_on_fixed_batch():
    """A short training loop must lower the loss (precision check)."""
    config = SJEPAConfig(conv_dim=32, hidden_dim=48, num_layers=2, num_heads=4,
                         ffn_dim=64, predictor_layers=1, predictor_heads=4,
                         predictor_ffn_dim=64, num_clusters=8, max_frames=128,
                         dropout=0.0, attention_dropout=0.0,
                         activation_dropout=0.0, layer_drop=0.0)
    model = SJEPA(config)
    model.train()
    waveform, mask, padding = _make_batch(config)
    targets = _fixed_targets(model, waveform, mask, padding, config)
    objective = JEPAObjective(use_visible_loss=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    losses = _train_steps(model, objective, optimizer, waveform, mask,
                          padding, targets, steps=30)
    print(f"[precision] first loss={losses[0]:.4f} last loss={losses[-1]:.4f}")
    # The loss after training must be clearly lower than at the start.
    assert losses[-1] < losses[0] * 0.9


def _fixed_targets(model, waveform, mask, padding, config):
    """Build a fixed soft target that the model will try to match."""
    with torch.no_grad():
        out = model(waveform, mask, padding_mask=padding)
    length = out.mask.shape[1]
    logits = torch.randn(waveform.shape[0], length, config.num_clusters)
    return torch.softmax(logits, dim=-1)


def _train_steps(model, objective, optimizer, waveform, mask, padding,
                 targets, steps):
    """Run a few training steps and return the loss at each step."""
    history = []
    for _ in range(steps):
        optimizer.zero_grad()
        out = model(waveform, mask, padding_mask=padding)
        result = objective(out.logits_masked, out.logits_visible, targets,
                           out.mask, out.padding_mask)
        result["loss"].backward()
        optimizer.step()
        history.append(float(result["loss"].detach()))
    return history
