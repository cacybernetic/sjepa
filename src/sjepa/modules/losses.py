"""KL divergence loss for the S-JEPA objective.

The target is a soft posterior over K GMM components. The model output is a
softmax over the same K components. The loss is the KL divergence between the
target and the prediction, averaged over the selected frames.

Phase 1 (and early phase 2) applies the loss at both masked and visible frames.
Later in phase 2 the loss is applied at masked frames only. The objective class
supports both modes through a single flag.

Two classes live here, each with one job:
  * `KLDivergenceLoss`: KL on a chosen set of frames.
  * `JEPAObjective`: build the frame selections and combine the two terms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class KLDivergenceLoss(nn.Module):
    """KL divergence between soft targets and predicted distributions."""

    def forward(self, logits, targets, selection):
        """Compute the KL loss on the selected frames.

        Args:
            logits: model logits of shape (batch, length, num_clusters).
            targets: soft target probabilities, same shape as logits.
            selection: bool tensor (batch, length), True where the frame counts.

        Returns:
            A scalar loss tensor. It is zero when no frame is selected.
        """
        num_clusters = logits.shape[-1]
        flat_selection = selection.reshape(-1)
        if not bool(flat_selection.any()):
            # A detached zero, NOT `logits.sum() * 0.0`: if any logit is
            # non-finite (e.g. a fully padded row through the attention), that
            # expression would turn the whole loss into NaN and destroy the
            # weights on backward. `requires_grad=True` keeps backward legal
            # when this is the only loss term.
            return torch.zeros((), device=logits.device, dtype=torch.float32,
                               requires_grad=True)
        # The KL is computed in float32 even when the model forward ran in
        # bf16 autocast: log_softmax over K classes is cheap and sensitive.
        log_pred = F.log_softmax(
            logits.reshape(-1, num_clusters)[flat_selection].float(), dim=-1)
        target = targets.reshape(-1, num_clusters)[flat_selection].float()
        return F.kl_div(log_pred, target, reduction="batchmean")


class JEPAObjective(nn.Module):
    """Combine the masked-frame loss with an optional visible-frame loss."""

    def __init__(self, use_visible_loss=False):
        super().__init__()
        self.use_visible_loss = use_visible_loss
        self.kl = KLDivergenceLoss()

    @staticmethod
    def _selections(mask, padding_mask):
        """Build the masked and visible frame selections.

        Args:
            mask: bool tensor (batch, length), True where masked.
            padding_mask: bool tensor (batch, length), True for real frames.

        Returns:
            A pair (masked_selection, visible_selection) of bool tensors.
        """
        if padding_mask is None:
            padding_mask = torch.ones_like(mask)
        masked = mask & padding_mask
        visible = (~mask) & padding_mask
        return masked, visible

    def forward(self, logits_masked, logits_visible, targets, mask,
                padding_mask=None):
        """Compute the full training loss.

        Args:
            logits_masked: cluster logits from the predictor output.
            logits_visible: cluster logits from the encoder output.
            targets: soft GMM posteriors of shape (batch, length, num_clusters).
            mask: bool tensor (batch, length), True where masked.
            padding_mask: bool tensor (batch, length), True for real frames.

        Returns:
            A dict with the keys "loss", "loss_masked", and "loss_visible".
        """
        masked_sel, visible_sel = self._selections(mask, padding_mask)
        loss_masked = self.kl(logits_masked, targets, masked_sel)
        loss_visible = self.kl(logits_visible, targets, visible_sel)
        total = loss_masked
        if self.use_visible_loss:
            total = total + loss_visible
        return {
            "loss": total,
            "loss_masked": loss_masked.detach(),
            "loss_visible": loss_visible.detach(),
        }
