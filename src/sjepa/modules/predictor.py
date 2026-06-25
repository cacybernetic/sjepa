"""JEPA predictor (the model `h_psi`).

The predictor takes the encoder output, puts a learned mask token at the hidden
frames, adds learned positional embeddings, and runs one transformer layer. Its
output at masked frames is compared with the GMM target by the loss.

The predictor is used only during training and is discarded afterwards. The
class focuses on one job: predict the hidden frames from the visible context.
"""

import torch
import torch.nn as nn

from .transformer import TransformerEncoderLayer


class JEPAPredictor(nn.Module):
    """Small transformer that fills the masked frames."""

    def __init__(self, config):
        super().__init__()
        dim = config.hidden_dim
        self.max_frames = config.max_frames
        # The mask token is a learned vector placed at every hidden frame.
        self.mask_token = nn.Parameter(torch.randn(dim) * 0.02)
        # Learned positional embeddings, one per frame slot.
        self.pos_embed = nn.Embedding(config.max_frames, dim)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(dim, config.predictor_heads,
                                    config.predictor_ffn_dim,
                                    config.predictor_dropout,
                                    config.predictor_dropout,
                                    config.predictor_dropout)
            for _ in range(config.predictor_layers)
        ])
        self.proj = nn.Linear(dim, dim)

    def _inject_mask_token(self, context, mask):
        """Replace masked frames with the learned mask token."""
        mask_token = self.mask_token.to(context.dtype)
        return torch.where(mask.unsqueeze(-1), mask_token, context)

    def _add_positions(self, x):
        """Add learned positional embeddings for frames 0..T-1."""
        length = x.shape[1]
        if length > self.max_frames:
            raise ValueError(
                f"sequence has {length} frames but the predictor only "
                f"supports {self.max_frames}")
        ids = torch.arange(length, device=x.device)
        pos = self.pos_embed(ids).to(x.dtype).unsqueeze(0)
        return x + pos

    def forward(self, context, mask, padding_mask=None):
        """Predict the hidden frames.

        Args:
            context: encoder output of shape (batch, length, hidden).
            mask: bool tensor (batch, length), True where a frame is masked.
            padding_mask: bool tensor (batch, length), True for real frames.

        Returns:
            A tensor of shape (batch, length, hidden) with predicted frames.
        """
        x = self._inject_mask_token(context, mask)
        x = self._add_positions(x)
        for layer in self.layers:
            x = layer(x, key_padding_mask=padding_mask)
        return self.proj(x)
