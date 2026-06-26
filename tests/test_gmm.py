"""Tests for the GMM module (frozen fit, posteriors, online update)."""

import torch

from sjepa.gmm import (
    DiagonalGMM,
    GMMFitter,
    OnlineGMM,
    ReservoirSampler,
)


def _two_clusters(per=200, dim=6, gap=6.0):
    """Build a simple two-cluster point cloud for fitting tests."""
    left = torch.randn(per, dim) - gap
    right = torch.randn(per, dim) + gap
    return torch.cat([left, right], dim=0)


def test_posteriors_sum_to_one():
    """Each posterior row must be a valid distribution."""
    gmm = GMMFitter(4, kmeans_iters=3, em_iters=5).fit(_two_clusters())
    posteriors = gmm.posteriors(_two_clusters())
    sums = posteriors.sum(dim=1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4)


def test_fitter_separates_clusters():
    """Two far clusters should get near-hard, opposite assignments."""
    data = _two_clusters(gap=10.0)
    gmm = GMMFitter(2, kmeans_iters=5, em_iters=10).fit(data)
    posteriors = gmm.posteriors(data)
    first = posteriors[0].argmax()
    last = posteriors[-1].argmax()
    assert first != last


def test_online_update_keeps_weights_normalized():
    """After an online update the weights still sum to one."""
    gmm = GMMFitter(3, kmeans_iters=3, em_iters=5).fit(_two_clusters())
    online = OnlineGMM.from_gmm(gmm, decay=0.9)
    data = _two_clusters()
    online.update(data, online.posteriors(data))
    assert abs(float(online.weights.sum()) - 1.0) < 1e-5


def test_state_dict_roundtrip():
    """Saving and loading a GMM must keep the parameters."""
    gmm = GMMFitter(3, kmeans_iters=3, em_iters=3).fit(_two_clusters())
    clone = DiagonalGMM.from_state_dict(gmm.state_dict())
    assert torch.allclose(gmm.means, clone.means)
    assert torch.allclose(gmm.weights, clone.weights)


def test_reservoir_respects_capacity():
    """The reservoir never holds more rows than its capacity."""
    reservoir = ReservoirSampler(50, 4)
    reservoir.add(torch.randn(200, 4))
    assert reservoir.collected().shape[0] == 50
    assert reservoir.seen == 200


def test_variance_floor_is_applied():
    """Variances must never drop below the small floor."""
    means = torch.zeros(2, 3)
    variances = torch.zeros(2, 3)
    weights = torch.tensor([0.5, 0.5])
    gmm = DiagonalGMM(means, variances, weights)
    assert float(gmm.variances.min()) > 0.0
