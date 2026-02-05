"""Metrics calculation for OT3Di."""

import torch


def compute_ot_metrics(
    P: torch.Tensor,
    cost: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> dict[str, float]:
    """Compute OT metrics (Avg Cost, Entropy).

    Args:
        P: Transport plan (B, L, L) or (L, L)
        cost: Cost matrix (B, L, L) or (L, L)
        mask: Sequence mask (B, L) or None. Used for normalization.

    Returns:
        Dictionary containing:
            - "ot_cost": Average OT cost per token
            - "entropy": Average Coupling Entropy per token
    """
    with torch.no_grad():
        # Handle single sample case
        if P.dim() == 2:
            P = P.unsqueeze(0)
            cost = cost.unsqueeze(0)
            if mask is not None and mask.dim() == 1:
                mask = mask.unsqueeze(0)

        # Basic validation
        if P.shape != cost.shape:
            raise ValueError(f"P shape {P.shape} mismatch with cost shape {cost.shape}")

        # P and Cost are already masked in the model output usually,
        # but explicit masking ensures safety.
        # However, P is sparse/masked by OT aligner logic.

        # Calculate sum(P * C)
        # Sum over L, L dimensions
        total_costs = (P * cost).sum(dim=(1, 2))  # (B,)

        # Calculate Entropy: -sum(P * log P)
        # Add epsilon for log safety
        eps = 1e-10
        P_safe = P + eps
        entropies = -(P * torch.log(P_safe)).sum(dim=(1, 2))  # (B,)

        # Return batch averages
        # Note: We do NOT normalize by sequence length because P typically sums to 1 (probability mass).
        # Thus total_costs is already the "average cost per unit mass".
        # If we divided by length, we would get values ~ 1/L, which are too small.
        # ot_cost -> sum(P_{ij} * C_{ij})
        # entropy -> -sum(P_{ij} * log(P_{ij}))

        return {"ot_cost": total_costs.mean().item(), "entropy": entropies.mean().item()}
