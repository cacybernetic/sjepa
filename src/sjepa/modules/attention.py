"""Multi-head self attention.

This layer lets every frame look at every other frame. We use PyTorch's fused
attention function (`scaled_dot_product_attention`). It is faster and more
stable than a hand-written softmax, and it can use flash attention on a GPU.

The class has one job: run multi-head self attention over a frame sequence.
"""

import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    """Standard multi-head self attention with separate q, k, v projections."""

    def __init__(self, embed_dim, num_heads, attention_dropout=0.1):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must divide by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.attention_dropout = attention_dropout
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def _split_heads(self, x, batch, length):
        """Reshape (B, T, C) to (B, heads, T, head_dim)."""
        x = x.view(batch, length, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def forward(self, x, key_padding_mask=None):
        """Run self attention.

        Args:
            x: a tensor of shape (batch, length, embed_dim).
            key_padding_mask: optional bool tensor of shape (batch, length).
                A True value marks a real frame; a False value marks padding
                that must be ignored.

        Returns:
            A tensor of shape (batch, length, embed_dim).
        """
        batch, length, _ = x.shape
        query = self._split_heads(self.q_proj(x), batch, length)
        key = self._split_heads(self.k_proj(x), batch, length)
        value = self._split_heads(self.v_proj(x), batch, length)

        attn_mask = self._build_attn_mask(key_padding_mask)
        dropout = self.attention_dropout if self.training else 0.0
        output = F.scaled_dot_product_attention(query, key, value,
                                                attn_mask=attn_mask,
                                                dropout_p=dropout)
        output = output.transpose(1, 2).reshape(batch, length, self.embed_dim)
        return self.out_proj(output)

    @staticmethod
    def _build_attn_mask(key_padding_mask):
        """Turn a (B, T) padding mask into a (B, 1, 1, T) additive mask."""
        if key_padding_mask is None:
            return None
        # The fused attention adds this mask to the scores. We pass a boolean
        # mask where True means "keep". It is broadcast over heads and queries.
        return key_padding_mask[:, None, None, :]
