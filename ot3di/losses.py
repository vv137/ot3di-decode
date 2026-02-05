"""Loss functions for OT3Di training."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def ot_alignment_loss(
    P: torch.Tensor,
    cost: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """OT alignment loss: weighted sum of transport costs.

    L_OT = <P*, C> = sum_{ij} P*_{ij} * C_{ij}

    Since P sums to 1 (balanced OT), <P,C> is already normalized.

    Args:
        P: Transport plan (B, L, L), sums to 1
        cost: Cost matrix (B, L, L)
        mask: Valid position mask (B, L)

    Returns:
        Scalar loss (mean over batch)
    """
    # Element-wise product
    loss_matrix = P * cost  # (B, L, L)

    if mask is not None:
        # Apply mask to both dimensions
        mask_2d = mask[:, :, None] & mask[:, None, :]  # (B, L, L)
        loss_matrix = loss_matrix * mask_2d.float()

    # P already sums to 1, so just sum and mean over batch
    loss_per_sample = loss_matrix.sum(dim=(-2, -1))  # (B,)
    return loss_per_sample.mean()


def soft_ce_loss(
    logits: torch.Tensor,
    P: torch.Tensor | None,
    target_tokens: torch.Tensor | None,
    num_tokens: int,
    mask: torch.Tensor | None = None,
    soft_targets: torch.Tensor | None = None,
) -> torch.Tensor:
    """Soft-label cross-entropy loss using OT weights.

    Computes soft labels: q_{ik} = sum_j (P_{ij} / a_i) * 1[t_j = k]
    Then: L_CE = -sum_i sum_k q_{ik} * log p(k|i)

    Args:
        logits: Predicted logits (B, L, K)
        P: Transport plan (B, L, L). Optional if soft_targets provided.
        target_tokens: Ground-truth token indices (B, L). Optional if soft_targets provided.
        num_tokens: Vocabulary size K
        mask: Valid position mask (B, L)
        soft_targets: Pre-computed soft targets q (B, L, K)

    Returns:
        Scalar loss (mean over batch and valid positions)
    """
    if soft_targets is not None:
        soft_labels = soft_targets
    else:
        if P is None or target_tokens is None:
            raise ValueError("Must provide either soft_targets or (P and target_tokens)")

        B, L, K = logits.shape

        # Normalize P by row sum (source marginal)
        row_sums = P.sum(dim=-1, keepdim=True).clamp(min=1e-8)  # (B, L, 1)
        P_normalized = P / row_sums  # (B, L, L)

        # Create one-hot for target tokens: (B, L, K)
        target_onehot = F.one_hot(target_tokens, num_classes=K).float()

        # Compute soft labels: q_{ik} = sum_j P_normalized_{ij} * target_onehot_{jk}
        # (B, L, L) @ (B, L, K) -> (B, L, K)
        soft_labels = torch.bmm(P_normalized, target_onehot)

    # Log softmax of logits
    log_probs = F.log_softmax(logits, dim=-1)

    # Cross-entropy with soft labels
    loss = -(soft_labels * log_probs).sum(dim=-1)  # (B, L)

    # Apply mask if provided
    if mask is not None:
        loss = loss * mask.float()
        num_valid = mask.float().sum().clamp(min=1.0)
        return loss.sum() / num_valid

    return loss.mean()


def hard_ce_loss(
    logits: torch.Tensor,
    target_tokens: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Standard hard-label cross-entropy loss (for comparison/ablation).

    Args:
        logits: Predicted logits (B, L, K)
        target_tokens: Ground-truth token indices (B, L)
        mask: Valid position mask (B, L)

    Returns:
        Scalar loss
    """
    B, L, K = logits.shape

    # Flatten for cross_entropy
    logits_flat = logits.view(-1, K)
    targets_flat = target_tokens.view(-1)

    # Compute loss per position
    loss = F.cross_entropy(logits_flat, targets_flat, reduction="none")
    loss = loss.view(B, L)

    if mask is not None:
        loss = loss * mask.float()
        num_valid = mask.float().sum().clamp(min=1.0)
        return loss.sum() / num_valid

    return loss.mean()


class OT3DiLoss:
    """Combined loss for OT3Di training.

    L = L_CE + alpha * L_OT

    Args:
        alpha: Weight for OT alignment loss.
        use_soft_ce: If True, use OT-weighted soft CE; else use hard CE.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        use_soft_ce: bool = True,
    ) -> None:
        self.alpha = alpha
        self.use_soft_ce = use_soft_ce

    def __call__(
        self,
        model_output: dict[str, torch.Tensor],
        target_tokens: torch.Tensor,
        num_tokens: int,
        mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute combined loss.

        Args:
            model_output: Output from OT3DiModel.forward()
            target_tokens: Ground-truth 3Di tokens (B, L)
            num_tokens: Vocabulary size
            mask: Valid position mask (B, L)

        Returns:
            Dictionary with:
                - loss: Total loss
                - loss_ce: Cross-entropy loss component
                - loss_ot: OT alignment loss component
        """
        logits = model_output["logits"]
        P = model_output.get("P")
        cost = model_output.get("cost")
        q = model_output.get("q")

        # OT alignment loss
        if P is not None and cost is not None:
            loss_ot = ot_alignment_loss(P, cost, mask)
        else:
            loss_ot = torch.tensor(0.0, device=logits.device)

        # CE loss (soft or hard)
        if self.use_soft_ce:
            loss_ce = soft_ce_loss(logits=logits, P=P, target_tokens=target_tokens, num_tokens=num_tokens, mask=mask, soft_targets=q)
        else:
            loss_ce = hard_ce_loss(logits, target_tokens, mask)

        # Combined loss
        loss = loss_ce + self.alpha * loss_ot

        return {
            "loss": loss,
            "loss_ce": loss_ce,
            "loss_ot": loss_ot,
        }
