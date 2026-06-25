"""Shared fixtures for the test suite.

These fixtures give small, fast objects so the tests run quickly on a CPU.
A tiny config keeps the model small while still exercising every component.
"""

import pytest
import torch

from sjepa import SJEPA, SJEPAConfig
from sjepa.modules import BlockMaskGenerator, build_padding_mask


@pytest.fixture(autouse=True)
def fixed_seed():
    """Use the same random seed in every test for repeatable results."""
    torch.manual_seed(0)


@pytest.fixture
def tiny_config():
    """A small config that builds a fast model for tests."""
    return SJEPAConfig(
        conv_dim=32,
        hidden_dim=48,
        num_layers=2,
        num_heads=4,
        ffn_dim=64,
        predictor_layers=1,
        predictor_heads=4,
        predictor_ffn_dim=64,
        num_clusters=10,
        max_frames=128,
    )


@pytest.fixture
def tiny_model(tiny_config):
    """A small S-JEPA model in eval mode (no dropout, no layer drop)."""
    model = SJEPA(tiny_config)
    model.eval()
    return model


@pytest.fixture
def batch(tiny_config):
    """A small batch with a waveform, a padding mask, and a block mask."""
    batch_size, samples = 2, 8000
    waveform = torch.randn(batch_size, 1, samples)
    num_frames = samples // tiny_config.hop
    frame_lengths = [num_frames, num_frames - 3]
    padding = build_padding_mask(batch_size, num_frames, frame_lengths,
                                 waveform.device)
    mask = BlockMaskGenerator().generate(batch_size, num_frames, frame_lengths,
                                         waveform.device)
    return {
        "waveform": waveform,
        "padding_mask": padding,
        "mask": mask,
        "num_frames": num_frames,
        "frame_lengths": frame_lengths,
    }
