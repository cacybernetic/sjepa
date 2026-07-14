"""Build the soft GMM targets for one batch.

The target is what the predictor must match at masked frames. The way it is
built depends on the phase:

  * Phase 1: MFCC features of the clean waveform, fed to the frozen GMM.
  * Phase 2: clean features from the EMA encoder at the active layer, fed to the
    online GMM, which is then updated from those same features.

Both builders return a (batch, frames, K) tensor of soft posteriors. No
gradient flows through this code.

Padding frames are excluded from every GMM statistic (they are digital silence
and would otherwise attract components and bias the mixture weights); the
posteriors of padded frames are still returned so the target tensor keeps the
batch shape, but the loss never selects them.
"""

from __future__ import annotations

import torch

from .dataset.features import MfccExtractor

# Upper bound on the frames kept per accumulation window to re-seed dead GMM
# components. Only a small random sample is needed; keeping every frame would
# hold hundreds of megabytes on the GPU for nothing.
_RESEED_SAMPLE_CAP = 4096


class Phase1TargetBuilder:
    """Make soft targets from MFCC features and a frozen GMM."""

    def __init__(self, gmm, sample_rate=16000, device="cpu"):
        self.gmm = gmm.to(device)
        self.extractor = MfccExtractor(sample_rate=sample_rate).to(device)
        self.device = device

    @torch.no_grad()
    def build(self, clean_waveform, padding_mask=None, accumulate=False):
        """Return soft targets of shape (batch, frames, K).

        Args:
            clean_waveform: the clean batch waveform.
            padding_mask: ignored here; the frozen GMM is not updated, and the
                loss already excludes padded frames. Kept for the shared
                builder interface.
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
    """Make soft targets from EMA encoder features and an online GMM.

    During training the builder accumulates the GMM *sufficient statistics*
    (a (K,) count vector and two (K, D) sums) across the micro-batches of a
    gradient-accumulation window, plus a small random frame sample for dead
    component re-seeding. The raw frames and responsibilities are never
    buffered, so the memory held across a window is O(K * D), not O(N * D).
    """

    def __init__(self, ema_encoder, online_gmm, layer, device="cpu"):
        self.ema_encoder = ema_encoder
        self.gmm = online_gmm.to(device)
        self.layer = layer
        self.device = device
        self._stats = None
        self._sample = []

    def set_layer(self, layer):
        """Switch the active encoder layer used as the GMM input."""
        self.layer = layer

    @staticmethod
    def _real_rows(flat, resp, padding_mask, frames, batch):
        """Return only the rows of real (non padding) frames."""
        if padding_mask is None:
            return flat, resp
        keep = padding_mask[:, :frames].reshape(-1)
        if keep.shape[0] != flat.shape[0]:
            # Defensive: mask and features disagree; fall back to everything.
            return flat, resp
        return flat[keep], resp[keep]

    def _accumulate(self, flat, resp):
        """Add one micro-batch to the running sufficient statistics."""
        if flat.shape[0] == 0:
            return
        counts, sum_x, sum_x2 = self.gmm.sufficient_stats(flat.float(),
                                                          resp.float())
        if self._stats is None:
            self._stats = [counts, sum_x, sum_x2]
        else:
            self._stats[0] += counts
            self._stats[1] += sum_x
            self._stats[2] += sum_x2
        cap = min(_RESEED_SAMPLE_CAP, flat.shape[0])
        index = torch.randint(0, flat.shape[0], (cap,), device=flat.device)
        self._sample.append(flat[index])

    @torch.no_grad()
    def build(self, clean_waveform, padding_mask=None, accumulate=False):
        """Return soft targets, optionally accumulating the GMM statistics.

        Args:
            clean_waveform: the clean batch waveform.
            padding_mask: bool tensor (batch, frames), True for real frames.
                Passed to the EMA encoder attention and used to keep padded
                frames out of the GMM statistics.
            accumulate: when True (training only), the sufficient statistics of
                the real frames are accumulated so that `post_step` updates the
                GMM from every micro-batch of a gradient-accumulation window,
                not just the last one. Validation calls it with False so
                nothing is accumulated during evaluation.
        """
        waveform = clean_waveform.to(self.device)
        feats = self.ema_encoder.extract_layer(waveform, self.layer,
                                               padding_mask=padding_mask)
        batch, frames, dim = feats.shape
        flat = feats.reshape(-1, dim)
        resp = self.gmm.posteriors(flat)
        if accumulate:
            real_flat, real_resp = self._real_rows(flat, resp, padding_mask,
                                                   frames, batch)
            self._accumulate(real_flat, real_resp)
        return resp.view(batch, frames, self.gmm.num_clusters)

    @torch.no_grad()
    def post_step(self):
        """Update the online GMM from the accumulated statistics, then clear."""
        if self._stats is None:
            return None
        counts, sum_x, sum_x2 = self._stats
        sample = torch.cat(self._sample, dim=0) if self._sample else None
        self.gmm.update_from_stats(counts, sum_x, sum_x2, sample=sample)
        self._stats = None
        self._sample = []
        return None
