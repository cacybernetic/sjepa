"""One transformer encoder layer (post-norm layout).

The layer has two parts: self attention and a feed-forward network. Each part
has a residual connection and a layer normalization applied after the residual
add (post-norm), as in HuBERT-Base.

The class has one job: transform a frame sequence with one transformer block.
"""

import torch.nn as nn
import torch.nn.functional as F

from .attention import MultiHeadSelfAttention
from .normalization import Fp32LayerNorm


class TransformerEncoderLayer(nn.Module):
    """A single post-norm transformer block.

    Order of operations:
      1. self attention, dropout, residual add, layer norm;
      2. feed-forward (Linear, GELU, Linear), dropout, residual add, layer norm.
    """

    def __init__(self, embed_dim=768, num_heads=12, ffn_dim=3072,
                 dropout=0.1, attention_dropout=0.1, activation_dropout=0.1):
        super().__init__()
        self.attention = MultiHeadSelfAttention(embed_dim, num_heads,
                                                attention_dropout)
        self.self_attn_layer_norm = Fp32LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, embed_dim)
        self.final_layer_norm = Fp32LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation_dropout = nn.Dropout(activation_dropout)

    def _attention_block(self, x, key_padding_mask):
        """Self attention sub-layer with residual add and norm."""
        residual = x
        x = self.attention(x, key_padding_mask=key_padding_mask)
        x = self.dropout(x)
        return self.self_attn_layer_norm(residual + x)

    def _feed_forward_block(self, x):
        """Feed-forward sub-layer with residual add and norm."""
        residual = x
        x = F.gelu(self.fc1(x))
        x = self.activation_dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return self.final_layer_norm(residual + x)

    def forward(self, x, key_padding_mask=None):
        """Apply the transformer block.

        Args:
            x: a tensor of shape (batch, length, embed_dim).
            key_padding_mask: optional bool tensor, True for real frames.

        Returns:
            A tensor of the same shape as the input.
        """
        x = self._attention_block(x, key_padding_mask)
        x = self._feed_forward_block(x)
        return x
