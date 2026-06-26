"""Export a trained S-JEPA encoder to ONNX.

Usage:
    exportsjepa -c cpu/configs/export.yaml

The config names the model size, the number of clusters, the weight file to load
(`init_weights`), and the output path (`onnx_path`). Only the encoder is
exported; the predictor, cluster head, and GMM are not needed for inference.
"""

from __future__ import annotations

import os

import torch

from ..config_schema import load_experiment_config
from ..logging import get_logger, setup_logging
from ..model import build_model
from ..onnx_export import OnnxExporter

_LOGGER = get_logger()


def _load_weights(model, path):
    """Load model weights from a checkpoint or weight file."""
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"init_weights not found: {path}")
    state = torch.load(path, map_location="cpu", weights_only=False)
    weights = state.get("model", state)
    model.load_state_dict(weights, strict=False)
    _LOGGER.info("Loaded weights from {}", path)


def run(config_path):
    """Build the model, load its weights, and export the encoder to ONNX."""
    config = load_experiment_config(config_path)
    setup_logging(level="DEBUG")
    overrides = dict(config.model.overrides)
    overrides.setdefault("num_clusters", config.gmm.num_clusters)
    model = build_model(config.model.size, **overrides).eval()
    _load_weights(model, config.init_weights)
    out_dir = os.path.dirname(os.path.abspath(config.onnx_path)) or "."
    os.makedirs(out_dir, exist_ok=True)
    OnnxExporter().export(model, config.onnx_path)


def main():
    """Console entry point for the exportsjepa command."""
    from .common import parse_config_arg
    config_path = parse_config_arg("Export the S-JEPA encoder to ONNX")
    run(config_path)


if __name__ == "__main__":
    main()
