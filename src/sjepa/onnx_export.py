"""Export the S-JEPA encoder to ONNX.

After training, only the encoder `f_phi` is kept. For production we export it to
ONNX so it can run with a fast runtime and no Python. The export wraps the
encoder so the ONNX graph takes a raw waveform and returns frame features.

This file has one job: turn a trained model into an ONNX file.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .logging import get_logger

_LOGGER = get_logger()


class EncoderForExport(nn.Module):
    """A thin wrapper that runs the encoder with no mask and returns features."""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, waveform):
        """Return the last-layer features for a waveform (batch, 1, samples)."""
        return self.encoder(waveform, mask=None, padding_mask=None)


def _dummy_waveform(hop, frames=64):
    """Build a small dummy waveform for tracing the graph."""
    return torch.randn(1, 1, hop * frames)


class OnnxExporter:
    """Export a trained S-JEPA encoder to an ONNX file."""

    def __init__(self, opset=17):
        self.opset = opset

    def export(self, model, out_path, hop=320):
        """Write the encoder to an ONNX file at `out_path`."""
        wrapper = EncoderForExport(model.encoder).eval()
        dummy = _dummy_waveform(hop)
        dynamic = {"waveform": {0: "batch", 2: "samples"},
                   "features": {0: "batch", 1: "frames"}}
        torch.onnx.export(
            wrapper, dummy, out_path, input_names=["waveform"],
            output_names=["features"], dynamic_axes=dynamic,
            opset_version=self.opset)
        _LOGGER.info("Exported encoder to ONNX at {}", out_path)
        return out_path
