"""Loss functions for multi-horizon top-k prediction.

Positive-class fraction is usually ~5–15% depending on top-k and horizon,
so focal loss helps more than plain BCE in our experience.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def bce_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Plain multi-label BCE."""
    return F.binary_cross_entropy_with_logits(logits, targets)


def focal_bce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Focal variant to down-weight easy negatives."""
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p = torch.sigmoid(logits)
    pt = torch.where(targets > 0.5, p, 1.0 - p)
    at = torch.where(
        targets > 0.5,
        torch.full_like(targets, alpha),
        torch.full_like(targets, 1.0 - alpha),
    )
    loss = at * (1.0 - pt).pow(gamma) * bce
    return loss.mean()


LOSS_FNS = {
    "bce": bce_loss,
    "focal": focal_bce_loss,
}
