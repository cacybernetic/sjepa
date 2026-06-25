"""Gradient scaling helper for the CNN frontend.

The paper scales the gradient that flows back into the CNN feature extractor by
a small factor (0.1). This stops the frontend from learning too fast. The
forward pass is not changed; only the backward pass is scaled.

This file has one autograd function and one thin module wrapper around it.
"""

import torch
import torch.nn as nn


class _GradMultiply(torch.autograd.Function):
    """Multiply the gradient by a fixed scale, leave the value unchanged."""

    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        # The value passes through with no change.
        return x

    @staticmethod
    def backward(ctx, grad):
        # The gradient is scaled on the way back.
        return grad * ctx.scale, None


def scale_gradient(x, scale):
    """Apply gradient scaling to a tensor.

    Args:
        x: the input tensor.
        scale: the factor used on the gradient during backward.

    Returns:
        The same tensor, but its gradient will be scaled.
    """
    return _GradMultiply.apply(x, scale)


class FeatureGradientScaler(nn.Module):
    """Module wrapper that scales the gradient of its input.

    It does nothing when `scale` is 1.0 or when the module is in eval mode.
    This keeps the forward pass fast and exact.
    """

    def __init__(self, scale=0.1):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        if not self.training or self.scale == 1.0:
            return x
        return scale_gradient(x, self.scale)
