"""Optimal Transport alignment module using vv137/sinkhorn."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum
from sinkhorn import sinkhorn, SinkhornOutput


class OTAligner(nn.Module):
    """Optimal Transport aligner using Triton-accelerated Sinkhorn.

    Uses the sinkhorn package from https://github.com/vv137/sinkhorn
    for high-performance balanced/unbalanced OT computation.

    Args:
        epsilon: Entropic regularization parameter.
        max_iters: Maximum Sinkhorn iterations.
        threshold: Convergence threshold.
        tau_a: KL relaxation for source (None = balanced).
        tau_b: KL relaxation for target (None = balanced).
        backend: "triton" or "pytorch".
    """

    def __init__(
        self,
        epsilon: float = 1.0,
        max_iters: int = 100,
        threshold: float = 1e-6,
        tau_a: float | None = None,
        tau_b: float | None = None,
        backend: str = "triton",
    ) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.max_iters = max_iters
        self.threshold = threshold
        self.tau_a = tau_a
        self.tau_b = tau_b
        self.backend = backend

    def compute_cost_matrix(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        target_weights: torch.Tensor | None = None,
        normalize: bool = True,
    ) -> torch.Tensor:
        """Compute L2 cost matrix between embeddings.

        Args:
            u: Sequence embeddings (B, L_u, D)
            v: Token embeddings (B, L_v, D)
            normalize: Whether to L2-normalize embeddings.

        Returns:
            Cost matrix (B, L_u, L_v)
        """
        if normalize:
            u = F.normalize(u, p=2, dim=-1)
            v = F.normalize(v, p=2, dim=-1)
            similarity = einsum(u, v, "b i d, b j d -> b i j")
            cost = 2.0 - 2.0 * similarity
        else:
            u_sqnorm = (u**2).sum(dim=-1, keepdim=True)
            v_sqnorm = (v**2).sum(dim=-1, keepdim=True)
            similarity = einsum(u, v, "b i d, b j d -> b i j")
            cost = u_sqnorm + v_sqnorm.transpose(-1, -2) - 2.0 * similarity

        if target_weights is not None:
            # target_weights: (B, L_v)
            # cost: (B, L_u, L_v)
            # Broadcast weights over L_u dimension
            cost = cost * target_weights.unsqueeze(1)

        return cost

    def compute_transport_plan(
        self,
        cost: torch.Tensor,
        output: SinkhornOutput,
    ) -> torch.Tensor:
        """Compute transport plan from dual potentials.

        P = exp((f[:,:,None] + g[:,None,:] - C) / epsilon)

        Args:
            cost: Cost matrix (B, N, M)
            output: SinkhornOutput with dual potentials f, g

        Returns:
            Transport plan P (B, N, M)
        """
        f, g = output.f, output.g
        log_P = (f[:, :, None] + g[:, None, :] - cost) / self.epsilon
        return log_P.exp()

    def forward(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        mask_u: torch.Tensor | None = None,
        mask_v: torch.Tensor | None = None,
        target_weights: torch.Tensor | None = None,
        return_cost: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Compute OT alignment.

        Args:
            u: Sequence embeddings (B, L_u, D)
            v: Token embeddings (B, L_v, D)
            mask_u: Mask for source positions (B, L_u)
            mask_v: Mask for target positions (B, L_v)
            target_weights: Weights for target tokens (B, L_v)
            return_cost: Whether to return the cost matrix.

        Returns:
            Dictionary with P, and optionally cost.
        """
        cost = self.compute_cost_matrix(u, v, target_weights=target_weights)

        # Compute OT using sinkhorn package
        output: SinkhornOutput = sinkhorn(
            C=cost,
            epsilon=self.epsilon,
            tau_a=self.tau_a,
            tau_b=self.tau_b,
            mask_a=mask_u,
            mask_b=mask_v,
            backend=self.backend,
            max_iters=self.max_iters,
            threshold=self.threshold,
        )

        # Compute transport plan from dual potentials
        P = self.compute_transport_plan(cost, output)

        # Apply masks to P to prevent leakage
        if mask_u is not None:
            P = P * mask_u[:, :, None].float()
        if mask_v is not None:
            P = P * mask_v[:, None, :].float()

        result = {"P": P, "sinkhorn_output": output}
        if return_cost:
            result["cost"] = cost

        return result
