"""CNN feature extractor: raw waveform to frame features.

This is a 7-layer 1D convolution stack. It maps a 16 kHz waveform to a sequence
of frame features. The kernel sizes and strides match HuBERT exactly, so the
total stride is 320 samples. That gives one frame every 20 ms (50 Hz).

The class has one job: turn a waveform into frame features.
"""

import torch.nn as nn

from .normalization import Fp32GroupNorm

# Each tuple is (kernel_size, stride) for one conv layer. The product of the
# strides is 5 * 2 * 2 * 2 * 2 * 2 * 2 = 320 samples per frame.
_LAYER_SHAPES = [
    (10, 5),
    (3, 2),
    (3, 2),
    (3, 2),
    (3, 2),
    (2, 2),
    (2, 2),
]


class ConvFeatureExtractor(nn.Module):
    """Stack of 1D conv layers that build frame features from a waveform.

    The first layer uses float32 group normalization. Every layer uses GELU.
    The output has shape (batch, conv_dim, num_frames).
    """

    def __init__(self, conv_dim=512):
        super().__init__()
        self.conv_dim = conv_dim
        self.conv_layers = nn.ModuleList()
        in_channels = 1
        for index, (kernel, stride) in enumerate(_LAYER_SHAPES):
            block = self._build_block(in_channels, conv_dim, kernel, stride,
                                      use_norm=(index == 0))
            self.conv_layers.append(block)
            in_channels = conv_dim

    @staticmethod
    def _build_block(in_channels, out_channels, kernel, stride, use_norm):
        """Build one conv block: conv, optional norm, then GELU."""
        conv = nn.Conv1d(in_channels, out_channels, kernel, stride, bias=False)
        if use_norm:
            norm = Fp32GroupNorm(1, out_channels)
            return nn.Sequential(conv, norm, nn.GELU())
        return nn.Sequential(conv, nn.GELU())

    def forward(self, waveform):
        """Run the conv stack.

        Args:
            waveform: a tensor of shape (batch, 1, num_samples).

        Returns:
            A tensor of shape (batch, conv_dim, num_frames).
        """
        x = waveform
        for block in self.conv_layers:
            x = block(x)
        return x
