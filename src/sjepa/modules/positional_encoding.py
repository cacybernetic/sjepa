"""Convolutional positional encoding.

The transformer needs to know the order of the frames. Instead of fixed sine
embeddings, HuBERT uses a single grouped 1D convolution as a relative position
signal. The convolution output is added back to the input (a residual add).

This class has one job: add position information to a frame sequence.
"""

import torch.nn as nn
import torch.nn.functional as F


class ConvPositionalEncoding(nn.Module):
    """Grouped 1D conv used as a relative positional encoding.

    The conv is weight-normalized, which helps stable training. When the kernel
    size is even, we drop the last time step so the length stays the same.
    """

    def __init__(self, embed_dim, kernel_size=128, groups=16):
        super().__init__()
        padding = kernel_size // 2
        conv = nn.Conv1d(embed_dim, embed_dim, kernel_size, padding=padding,
                         groups=groups)
        # Weight normalization splits the weight into a direction and a length.
        # This makes the conv easier to train.
        self.conv = nn.utils.parametrizations.weight_norm(conv, name="weight",
                                                          dim=2)
        # With an even kernel the output is one step too long; we remove it.
        self.num_remove = 1 if kernel_size % 2 == 0 else 0

    def forward(self, x):
        """Add the positional signal to the input.

        Args:
            x: a tensor of shape (batch, num_frames, embed_dim).

        Returns:
            A tensor of the same shape as the input.
        """
        # Conv1d wants (batch, channels, time), so we move the axes.
        x_conv = self.conv(x.transpose(1, 2))
        if self.num_remove > 0:
            x_conv = x_conv[:, :, :-self.num_remove]
        x_conv = F.gelu(x_conv).transpose(1, 2)
        return x + x_conv
