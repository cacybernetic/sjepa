"""Build the soft GMM targets for one batch.

The target is what the predictor must match at masked frames. The way it is
built depends on the phase:

  * Phase 1: MFCC features of the clean waveform, fed to the frozen GMM.
  * Phase 2: clean features from the EMA encoder at the active layer, fed to the
    online GMM, which is then updated from those same features.

Both builders return a (batch, frames, K) tensor of soft posteriors. No
gradient flows through this code.
"""

from __future__ import annotations

import torch

from .dataset.features import MfccExtractor


class Phase1TargetBuilder:
    """Make soft targets from MFCC features and a frozen GMM."""

    def __init__(self, gmm, sample_rate=16000, device="cpu"):
        self.gmm = gmm.to(device)
        self.extractor = MfccExtractor(sample_rate=sample_rate).to(device)
        self.device = device

    @torch.no_grad()
    def build(self, clean_waveform, accumulate=False):
        """Return soft targets of shape (batch, frames, K).

        Args:
            clean_waveform: the clean batch waveform.
            accumulate: ignored here; the Phase 1 GMM is frozen. Kept so the two
                builders share one interface.
        """
        waveform = clean_waveform.squeeze(1).to(self.device)
        features = self.extractor.extract(waveform)
        batch, frames, dim = features.shape
        flat = features.reshape(-1, dim)
        posteriors = self.gmm.posteriors(flat)
        return posteriors.view(batch, frames, self.gmm.num_clusters)

    def post_step(self):
        """Phase 1 GMM is frozen, so there is nothing to update."""
        return None


class Phase2TargetBuilder:
    """Make soft targets from EMA encoder features and an online GMM."""

    def __init__(self, ema_encoder, online_gmm, layer, device="cpu"):
        self.ema_encoder = ema_encoder
        self.gmm = online_gmm.to(device)
        self.layer = layer
        self.device = device
        self._feat_buffer = []
        self._resp_buffer = []

    def set_layer(self, layer):
        """Switch the active encoder layer used as the GMM input."""
        self.layer = layer

    @torch.no_grad()
    def build(self, clean_waveform, accumulate=False):
        """Return soft targets, optionally buffering features for the update.

        Args:
            clean_waveform: the clean batch waveform.
            accumulate: when True (training only), the frame features and their
                responsibilities are buffered so that `post_step` updates the
                GMM from every micro-batch of a gradient-accumulation window,
                not just the last one. Validation calls it with False so the
                buffer never grows during evaluation.
        """
        waveform = clean_waveform.to(self.device)
        feats = self.ema_encoder.extract_layer(waveform, self.layer)
        batch, frames, dim = feats.shape
        flat = feats.reshape(-1, dim)
        resp = self.gmm.posteriors(flat)
        if accumulate:
            self._feat_buffer.append(flat)
            self._resp_buffer.append(resp)
        return resp.view(batch, frames, self.gmm.num_clusters)

    @torch.no_grad()
    def post_step(self):
        """Update the online GMM from every buffered micro-batch, then clear."""
        if not self._feat_buffer:
            return None
        features = torch.cat(self._feat_buffer, dim=0)
        resp = torch.cat(self._resp_buffer, dim=0)
        self.gmm.update(features, resp)
        self._feat_buffer.clear()
        self._resp_buffer.clear()
        return None
