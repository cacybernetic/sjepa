"""Print a full summary of the model before training starts.

The spec asks for a complete architecture summary so we know the layers, the
parameter counts, and the memory size before any heavy work. We use torchinfo
when it can run, and fall back to a simple parameter count otherwise.
"""

from __future__ import annotations

import torch

from .logging import banner, get_logger

_LOGGER = get_logger()


def _count(model):
    """Return (total, trainable) parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def _dummy_waveform(config, device):
    """Build a small waveform for tracing the encoder graph."""
    samples = config.hop * 32
    return torch.randn(1, 1, samples, device=device)


def _log_part_counts(model):
    """Log the parameter count of each part of the model."""
    for name, value in model.count_parameters().items():
        _LOGGER.info("  {:<20} = {:.2f} M", name, value)


def log_model_summary(model, config, device="cpu"):
    """Log the architecture summary and the parameter counts.

    The encoder is the part kept for inference, so we print its full layer
    table with torchinfo. The predictor and cluster head are reported by their
    parameter counts. The whole-model totals come first.

    Args:
        model: the S-JEPA model.
        config: the model `SJEPAConfig` (for the dummy input shape).
        device: where the dummy forward runs.
    """
    _LOGGER.info(banner("model summary", color="cyan"))
    total, trainable = _count(model)
    _LOGGER.info("  total parameters     = {:,}", total)
    _LOGGER.info("  trainable parameters = {:,}", trainable)
    _log_part_counts(model)
    try:
        from torchinfo import summary
        waveform = _dummy_waveform(config, device)
        report = summary(model.encoder, input_data=(waveform,), verbose=0,
                         depth=3)
        for line in str(report).splitlines():
            _LOGGER.info("  {}", line)
    except Exception as error:  # torchinfo is best effort only.
        _LOGGER.warning("torchinfo summary skipped: {}", error)
