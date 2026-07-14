"""Gaussian Mixture Model used to build the soft targets of S-JEPA.

The paper trains the model to match the soft posteriors of a diagonal
covariance GMM at masked frames. Two GMM kinds are used:

  * Phase 1: a frozen GMM fit once on 39-dim MFCC features (K = 100).
  * Phase 2: an online GMM over encoder features (K = 500), updated by an
    exponential moving average of minibatch sufficient statistics.

This file keeps every piece small and with a single job:

  * `DiagonalGMM`: hold the parameters and compute soft posteriors.
  * `GMMFitter`: fit a frozen GMM with k-means init plus EM refinement.
  * `OnlineGMM`: a `DiagonalGMM` that can EMA-update its parameters online.
  * `ReservoirSampler`: keep a bounded random subset of streamed frames.

All math follows the closed form in the paper appendix. No gradient flows
through the GMM at any point.
"""

from __future__ import annotations

import math

import torch

# Small floor for variances. It keeps the log and the division stable.
# The paper clamps at 1e-6; a higher floor flattens the posteriors.
_VAR_FLOOR = 1e-6
# Default chunk sizes for the chunked soft assignment (bounds peak memory).
_CHUNK_N = 4096
_CHUNK_K = 512
# A component is considered "dead" when its mixture weight falls below this
# fraction of the uniform weight (1/K). Dead components are re-seeded so the
# online GMM does not slowly collapse onto a handful of live clusters.
_DEAD_WEIGHT_FACTOR = 0.01


class DiagonalGMM:
    """A diagonal-covariance GMM that returns soft posteriors over K parts.

    The parameters are plain tensors (not `nn.Parameter`) because the GMM is
    never trained by gradient descent. They live on one device and dtype.
    """

    def __init__(self, means, variances, weights):
        self.means = means
        self.variances = variances.clamp(min=_VAR_FLOOR)
        self.weights = weights
        self._refresh_log_weights()

    @property
    def num_clusters(self):
        """Return K, the number of mixture components."""
        return self.means.shape[0]

    @property
    def dim(self):
        """Return D, the feature dimension of one component."""
        return self.means.shape[1]

    @property
    def device(self):
        """Return the device that holds the parameters."""
        return self.means.device

    def _refresh_log_weights(self):
        """Recompute the log of the mixture weights after any change."""
        self.log_weights = self.weights.clamp(min=1e-8).log()

    def to(self, device):
        """Move every parameter tensor to a device. Returns self."""
        self.means = self.means.to(device)
        self.variances = self.variances.to(device)
        self.weights = self.weights.to(device)
        self._refresh_log_weights()
        return self

    def _log_prob_block(self, features, k_start, k_end):
        """Return log N for a block of K components, shape (N, k_end-k_start)."""
        means = self.means[k_start:k_end]
        variances = self.variances[k_start:k_end]
        diff = features.unsqueeze(1) - means.unsqueeze(0)
        mahalanobis = (diff * diff / variances.unsqueeze(0)).sum(dim=2)
        log_det = variances.log().sum(dim=1)
        const = self.dim * math.log(2.0 * math.pi)
        return -0.5 * (const + log_det.unsqueeze(0) + mahalanobis)

    def _log_joint(self, features):
        """Return log(pi_k) + log N for every component, shape (N, K)."""
        num = features.shape[0]
        out = features.new_empty((num, self.num_clusters))
        for k_start in range(0, self.num_clusters, _CHUNK_K):
            k_end = min(k_start + _CHUNK_K, self.num_clusters)
            block = self._log_prob_block(features, k_start, k_end)
            out[:, k_start:k_end] = block + self.log_weights[k_start:k_end]
        return out

    @torch.no_grad()
    def posteriors(self, features):
        """Compute soft posteriors q(k | feature) for each frame.

        Args:
            features: tensor of shape (N, D). It is cast to float32 inside.

        Returns:
            A tensor of shape (N, K) where each row sums to one.
        """
        features = features.float()
        num = features.shape[0]
        out = features.new_empty((num, self.num_clusters))
        for start in range(0, num, _CHUNK_N):
            end = min(start + _CHUNK_N, num)
            log_joint = self._log_joint(features[start:end])
            normalizer = log_joint.logsumexp(dim=1, keepdim=True)
            out[start:end] = (log_joint - normalizer).exp()
        return out

    def state_dict(self):
        """Return a plain dict that can be saved with torch.save."""
        return {
            "means": self.means.cpu(),
            "variances": self.variances.cpu(),
            "weights": self.weights.cpu(),
        }

    @classmethod
    def from_state_dict(cls, state, device="cpu"):
        """Rebuild a GMM from a dict made by `state_dict`."""
        means = state["means"].to(device).float()
        variances = state["variances"].to(device).float()
        weights = state["weights"].to(device).float()
        return cls(means, variances, weights)


class ReservoirSampler:
    """Keep a bounded random sample of streamed feature rows.

    This follows Vitter reservoir sampling. It lets us fit the GMM on a fixed
    number of frames without holding the whole corpus in memory.
    """

    def __init__(self, capacity, dim, device="cpu"):
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        self.capacity = capacity
        self.buffer = torch.empty((capacity, dim), device=device)
        self.seen = 0
        self.filled = 0

    @torch.no_grad()
    def add(self, features):
        """Add a block of rows (shape (M, D)) to the reservoir.

        The whole block is processed with tensor operations (no per-row Python
        loop, no per-row host/device sync). Row i of the block, arriving as the
        (seen + i + 1)-th stream element, replaces a uniformly random slot with
        probability capacity / (seen + i + 1), exactly like the sequential
        Vitter algorithm; when several accepted rows target the same slot, the
        latest one wins, which matches sequential processing.
        """
        features = features.reshape(-1, features.shape[-1])
        num = features.shape[0]
        if num == 0:
            return
        take = min(self.capacity - self.filled, num)
        if take > 0:
            self.buffer[self.filled:self.filled + take] = \
                features[:take].to(self.buffer.device)
            self.filled += take
            self.seen += take
            features = features[take:]
            num -= take
        if num == 0:
            return
        sizes = torch.arange(1, num + 1, device=features.device) + self.seen
        slots = (torch.rand(num, device=features.device) * sizes).long()
        accept = slots < self.capacity
        if bool(accept.any()):
            rows = features[accept].to(self.buffer.device)
            chosen = slots[accept].to(self.buffer.device)
            order = torch.arange(chosen.shape[0], device=self.buffer.device)
            last = torch.full((self.capacity,), -1, dtype=torch.long,
                              device=self.buffer.device)
            last.scatter_reduce_(0, chosen, order, reduce="amax",
                                 include_self=True)
            winners = last[last >= 0]
            self.buffer[chosen[winners]] = rows[winners]
        self.seen += num

    def collected(self):
        """Return the rows kept so far, shape (filled, D)."""
        return self.buffer[:self.filled]


class _KMeans:
    """Mini-batch k-means used only to initialize the GMM means."""

    def __init__(self, num_clusters, num_iters=5, batch_size=10000):
        self.num_clusters = num_clusters
        self.num_iters = num_iters
        self.batch_size = batch_size

    @staticmethod
    def _assign(features, centroids):
        """Return the nearest centroid index for each row (chunked over rows)."""
        labels = features.new_empty(features.shape[0], dtype=torch.long)
        for start in range(0, features.shape[0], _CHUNK_N):
            chunk = features[start:start + _CHUNK_N]
            labels[start:start + _CHUNK_N] = torch.cdist(chunk, centroids).argmin(dim=1)
        return labels

    def _init_centroids(self, features):
        """Pick K random rows as the first centroids."""
        index = torch.randperm(features.shape[0])[:self.num_clusters]
        return features[index].clone()

    def _update(self, features, centroids):
        """Run one pass of mini-batch updates over the features."""
        labels = self._assign(features, centroids)
        for k in range(self.num_clusters):
            members = features[labels == k]
            if members.shape[0] > 0:
                centroids[k] = members.mean(dim=0)
        return centroids

    def fit(self, features):
        """Fit centroids and return (centroids, labels)."""
        centroids = self._init_centroids(features)
        for _ in range(self.num_iters):
            centroids = self._update(features, centroids)
        labels = self._assign(features, centroids)
        return centroids, labels


class GMMFitter:
    """Fit a frozen diagonal GMM with k-means init and EM refinement.

    The steps follow the paper appendix:
      1. k-means init (a few iterations) for the means.
      2. per-cluster sample variance and empirical weights.
      3. a fixed number of EM iterations for refinement.
    """

    def __init__(self, num_clusters, kmeans_iters=5, em_iters=20):
        if num_clusters <= 0:
            raise ValueError("num_clusters must be > 0")
        self.num_clusters = num_clusters
        self.kmeans_iters = kmeans_iters
        self.em_iters = em_iters

    def _init_from_kmeans(self, features):
        """Build the first GMM from a k-means labeling of the features."""
        kmeans = _KMeans(self.num_clusters, self.kmeans_iters)
        centroids, labels = kmeans.fit(features)
        variances = torch.ones_like(centroids)
        weights = torch.ones(self.num_clusters, device=features.device)
        for k in range(self.num_clusters):
            members = features[labels == k]
            weights[k] = max(members.shape[0], 1)
            if members.shape[0] > 1:
                variances[k] = members.var(dim=0, unbiased=False)
        weights = weights / weights.sum()
        return DiagonalGMM(centroids, variances, weights)

    @staticmethod
    def _em_step(gmm, features):
        """Run one EM step and return a new GMM with updated parameters.

        The sufficient statistics are accumulated over row chunks so no
        (N, K, D) tensor is ever materialized. For a diagonal GMM the per
        component variance is the closed form ``E[x^2] - E[x]^2`` weighted by
        the responsibilities, which needs only (K, D) running sums.
        """
        num_k, dim = gmm.num_clusters, gmm.dim
        counts = features.new_zeros(num_k)
        sum_x = features.new_zeros(num_k, dim)
        sum_x2 = features.new_zeros(num_k, dim)
        for start in range(0, features.shape[0], _CHUNK_N):
            chunk = features[start:start + _CHUNK_N]
            resp = gmm.posteriors(chunk)              # (M, K)
            counts += resp.sum(dim=0)
            sum_x += resp.t() @ chunk                 # (K, D)
            sum_x2 += resp.t() @ (chunk * chunk)      # (K, D)
        counts = counts.clamp(min=1e-8)
        means = sum_x / counts.unsqueeze(1)
        variances = sum_x2 / counts.unsqueeze(1) - means * means
        weights = counts / counts.sum()
        return DiagonalGMM(means, variances, weights)

    def fit(self, features):
        """Fit the GMM on a feature matrix of shape (N, D)."""
        features = features.float()
        if features.shape[0] < self.num_clusters:
            raise ValueError("need at least K frames to fit a K-component GMM")
        gmm = self._init_from_kmeans(features)
        for _ in range(self.em_iters):
            gmm = self._em_step(gmm, features)
        return gmm


class OnlineGMM(DiagonalGMM):
    """A GMM whose parameters move online with an EMA of batch statistics.

    Phase 2 keeps the GMM following the encoder features as they change. After
    each batch we compute responsibility-weighted means and variances and blend
    them into the current parameters with a slow decay.
    """

    def __init__(self, means, variances, weights, decay=0.999):
        super().__init__(means, variances, weights)
        if not 0.0 < decay < 1.0:
            raise ValueError("decay must be between 0 and 1")
        self.decay = decay

    @staticmethod
    def sufficient_stats(features, resp):
        """Return (counts, sum_x, sum_x2) for one batch of frames.

        Only (K,) and (K, D) sums are kept, so no (N, K, D) tensor is ever
        materialized (the naive per-pair difference would need hundreds of
        gigabytes at K=500, D=768). Same closed form as `GMMFitter._em_step`.
        """
        counts = resp.sum(dim=0)
        sum_x = resp.t() @ features
        sum_x2 = resp.t() @ (features * features)
        return counts, sum_x, sum_x2

    @staticmethod
    def _stats_to_params(counts, sum_x, sum_x2):
        """Turn sufficient statistics into (means, variances, weights)."""
        counts = counts.clamp(min=1e-8)
        means = sum_x / counts.unsqueeze(1)
        variances = (sum_x2 / counts.unsqueeze(1) - means * means).clamp(
            min=_VAR_FLOOR)
        weights = counts / counts.sum()
        return means, variances, weights

    @torch.no_grad()
    def _reseed_dead(self, features):
        """Re-seed components that have decayed below a usable weight.

        Without this, components that stop attracting frames drift toward a zero
        mean and the floor variance, turning into near-degenerate spikes that
        shrink the effective vocabulary and let the soft targets collapse. We
        re-seed each dead component from a random frame of the current batch,
        give it the mean live variance, and a small floor weight. This mirrors
        the paper's "empty clusters are re-seeded from random reservoir frames".

        Returns:
            The number of components that were re-seeded.
        """
        min_weight = (1.0 / self.num_clusters) * _DEAD_WEIGHT_FACTOR
        dead = self.weights < min_weight
        num_dead = int(dead.sum())
        if num_dead == 0 or features.shape[0] == 0:
            return 0
        live = ~dead
        base_var = (self.variances[live].mean(dim=0) if bool(live.any())
                    else self.variances.mean(dim=0))
        index = torch.randint(0, features.shape[0], (num_dead,),
                              device=features.device)
        self.means[dead] = features[index]
        self.variances[dead] = base_var.clamp(min=_VAR_FLOOR)
        self.weights[dead] = min_weight
        self.weights = self.weights / self.weights.sum()
        self._refresh_log_weights()
        return num_dead

    @torch.no_grad()
    def _blend(self, means, variances, weights):
        """Blend one batch estimate into the parameters with the EMA decay."""
        alpha = self.decay
        self.means = alpha * self.means + (1.0 - alpha) * means
        self.variances = (alpha * self.variances
                          + (1.0 - alpha) * variances).clamp(min=_VAR_FLOOR)
        self.weights = alpha * self.weights + (1.0 - alpha) * weights
        self.weights = self.weights / self.weights.sum()
        self._refresh_log_weights()

    @torch.no_grad()
    def update_from_stats(self, counts, sum_x, sum_x2, sample=None):
        """Update from precomputed sufficient statistics.

        Args:
            counts: (K,) responsibility sums accumulated over the window.
            sum_x: (K, D) responsibility-weighted feature sums.
            sum_x2: (K, D) responsibility-weighted squared-feature sums.
            sample: optional (M, D) frame sample used to re-seed dead
                components. When None, dead components are left as they are.
        """
        means, variances, weights = self._stats_to_params(counts, sum_x, sum_x2)
        self._blend(means, variances, weights)
        if sample is not None and sample.shape[0] > 0:
            self._reseed_dead(sample.float())

    @torch.no_grad()
    def update(self, features, resp):
        """Blend batch statistics into the parameters with the EMA decay.

        Args:
            features: tensor (N, D) of encoder features used as GMM input.
            resp: soft posteriors (N, K) for those features.
        """
        features = features.float()
        resp = resp.float()
        counts, sum_x, sum_x2 = self.sufficient_stats(features, resp)
        self.update_from_stats(counts, sum_x, sum_x2, sample=features)

    @classmethod
    def from_gmm(cls, gmm, decay=0.999):
        """Build an online GMM that starts from a fitted frozen GMM."""
        return cls(gmm.means.clone(), gmm.variances.clone(),
                   gmm.weights.clone(), decay=decay)
