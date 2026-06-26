"""Loss functions for S-JEPA training.

The training signal is a single KL divergence between the GMM soft posteriors
and the predictor softmax at the chosen frames. The real work lives in
`modules/losses.py`; this file is the public entry point and a small factory.

  * Phase 1 (and early Phase 2): KL at masked and visible frames.
  * Late Phase 2: KL at masked frames only.
"""

from __future__ import annotations

from .modules.losses import JEPAObjective, KLDivergenceLoss

__all__ = ["JEPAObjective", "KLDivergenceLoss", "build_objective"]


def build_objective(use_visible_loss=False):
    """Build the training objective.

    Args:
        use_visible_loss: when True, the visible-frame KL is added to the
            masked-frame KL. Phase 1 uses True; late Phase 2 uses False.

    Returns:
        A ready `JEPAObjective`.
    """
    return JEPAObjective(use_visible_loss=use_visible_loss)
