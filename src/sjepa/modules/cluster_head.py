"""Cluster head (the model `g_omega`).

The cluster head is a small 3-layer MLP. It maps each frame to K logits, where
K is the number of GMM components (100 in phase 1, 500 in phase 2). The same
head is used for the encoder output (visible frames) and for the predictor
output (masked frames). Only the masked frames drive the training loss.

The class focuses on one job: map frame features to cluster logits.
"""

import torch.nn as nn


class ClusterHead(nn.Module):
    """Three-layer MLP that produces K logits per frame."""

    def __init__(self, hidden_dim, num_clusters):
        super().__init__()
        self.num_clusters = num_clusters
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_clusters),
        )

    def forward(self, features):
        """Map frame features to cluster logits.

        Args:
            features: a tensor of shape (batch, length, hidden_dim).

        Returns:
            A tensor of shape (batch, length, num_clusters).
        """
        return self.net(features)
