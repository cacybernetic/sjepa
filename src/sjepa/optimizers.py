"""Build the optimizer with sensible parameter groups.

Good practice is to skip weight decay on biases and on normalization weights.
Those parameters are small and decaying them hurts training. We split the model
parameters into two groups: one with weight decay and one without.

The factory supports AdamW (the paper default), Adam, and SGD. The paper uses
AdamW with betas (0.9, 0.99) and weight decay 1e-3.
"""

from __future__ import annotations

import torch

from .logging import get_logger, log_hparams

_LOGGER = get_logger()


def _no_decay(name, param):
    """Return True when a parameter should not get weight decay."""
    if param.ndim <= 1:
        return True
    lowered = name.lower()
    return "bias" in lowered or "norm" in lowered or "embed" in lowered


def build_param_groups(model, weight_decay):
    """Split parameters into a decay group and a no-decay group."""
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        target = no_decay if _no_decay(name, param) else decay
        target.append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def replace_parameters(optimizer, old_params, new_named_params, weight_decay):
    """Swap a set of parameters inside an existing optimizer in place.

    Used by the Phase 1 -> Phase 2 transition, where the cluster head is rebuilt
    for the new number of clusters. The encoder and predictor parameters are the
    same tensor objects before and after, so their optimizer state (Adam moments)
    is preserved untouched; only the old cluster-head parameters are dropped and
    the new ones inserted into the right decay group with fresh state.

    Args:
        optimizer: an optimizer built by `build_optimizer` (two groups: a decay
            group first, then a no-decay group).
        old_params: the parameters to remove (the old cluster head).
        new_named_params: (name, parameter) pairs to add (the new cluster head).
        weight_decay: unused here but kept for signature clarity; the group's own
            weight decay applies.
    """
    if len(optimizer.param_groups) < 2:
        raise ValueError("expected a decay group and a no-decay group")
    old_ids = {id(param) for param in old_params}
    for group in optimizer.param_groups:
        group["params"] = [p for p in group["params"] if id(p) not in old_ids]
    for param in old_params:
        optimizer.state.pop(param, None)
    decay_group, no_decay_group = optimizer.param_groups[0], optimizer.param_groups[1]
    for name, param in new_named_params:
        if not param.requires_grad:
            continue
        group = no_decay_group if _no_decay(name, param) else decay_group
        group["params"].append(param)


def _build_adamw(groups, lr, betas, eps):
    """Build an AdamW optimizer from parameter groups."""
    return torch.optim.AdamW(groups, lr=lr, betas=betas, eps=eps)


def _build_adam(groups, lr, betas, eps):
    """Build an Adam optimizer from parameter groups."""
    return torch.optim.Adam(groups, lr=lr, betas=betas, eps=eps)


def _build_sgd(groups, lr, momentum):
    """Build an SGD optimizer from parameter groups."""
    return torch.optim.SGD(groups, lr=lr, momentum=momentum)


def build_optimizer(model, name="adamw", lr=1e-4, weight_decay=1e-3,
                    betas=(0.9, 0.99), eps=1e-8, momentum=0.9):
    """Build an optimizer for the model.

    Args:
        model: the network whose parameters are optimized.
        name: "adamw", "adam", or "sgd".
        lr: the learning rate.
        weight_decay: the weight decay for the decay group.
        betas: the Adam/AdamW beta values.
        eps: the Adam/AdamW epsilon.
        momentum: the SGD momentum.

    Returns:
        A ready optimizer.
    """
    groups = build_param_groups(model, weight_decay)
    key = name.lower()
    if key == "adamw":
        optimizer = _build_adamw(groups, lr, betas, eps)
    elif key == "adam":
        optimizer = _build_adam(groups, lr, betas, eps)
    elif key == "sgd":
        optimizer = _build_sgd(groups, lr, momentum)
    else:
        raise ValueError(f"unknown optimizer '{name}'")
    log_hparams("optimizer", {"name": key, "lr": lr, "weight_decay": weight_decay},
                color="green")
    return optimizer
