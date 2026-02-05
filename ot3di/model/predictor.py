"""Token prediction head for 3Di tokens."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenPredictor(nn.Module):
    """Predicts 3Di token distribution from sequence embeddings.

    Uses Inner Product Logit: logit = tau * (u @ e^T)

    Args:
        embed_dim: Input embedding dimension.
        num_tokens: Number of 3Di tokens (vocabulary size).
        dropout: Dropout probability.
    """

    def __init__(
        self,
        embed_dim: int,
        num_tokens: int = 20,
        dropout: float = 0.1,
        embedding_layer: nn.Embedding | None = None,
        **kwargs,  # Ignore extra args like 'mode'
    ) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.num_tokens = num_tokens

        # Token embedding table (used as both teacher embeddings and prediction targets)
        if embedding_layer is not None:
            self.token_embedding = embedding_layer
            if self.token_embedding.embedding_dim != embed_dim:
                # If dimensions mismatch, we might need a projection, but for now enforcing match
                raise ValueError(f"Embedding layer dim {self.token_embedding.embedding_dim} != predictor dim {embed_dim}")
        else:
            self.token_embedding = nn.Embedding(num_tokens, embed_dim)

        # Prediction head
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
        )
        # Learnable temperature parameter tau
        self.tau = nn.Parameter(torch.tensor(1.0))

    def get_token_embeddings(self, tokens: torch.Tensor) -> torch.Tensor:
        """Get embeddings for token indices.

        Args:
            tokens: Token indices (B, L) with values in [0, num_tokens).

        Returns:
            Token embeddings (B, L, D)
        """
        return self.token_embedding(tokens)

    def get_all_token_embeddings(self) -> torch.Tensor:
        """Get all token embeddings.

        Returns:
            All token embeddings (num_tokens, D)
        """
        return self.token_embedding.weight

    def forward(
        self,
        embeddings: torch.Tensor,
        temperature: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        """Predict token distribution from embeddings.

        Args:
            embeddings: Input embeddings (B, L, D)
            temperature: Softmax temperature for probability computation.

        Returns:
            Dictionary with:
                - logits: Raw logits (B, L, num_tokens)
                - probs: Token probabilities (B, L, num_tokens)
        """
        processed = self.head(embeddings)  # (B, L, D)
        token_embeds = self.get_all_token_embeddings()  # (K, D)

        # Normalize for cosine similarity
        processed_norm = F.normalize(processed, p=2, dim=-1)
        token_embeds_norm = F.normalize(token_embeds, p=2, dim=-1)

        # Dot product: (B, L, D) @ (D, K) -> (B, L, K)
        logits = torch.matmul(processed_norm, token_embeds_norm.t())

        # Scale by learnable temperature
        logits = logits * self.tau

        probs = F.softmax(logits / temperature, dim=-1)

        return {"logits": logits, "probs": probs}

    def predict(
        self,
        embeddings: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Predict most likely token for each position.

        Args:
            embeddings: Input embeddings (B, L, D)
            temperature: Softmax temperature (not used for argmax).

        Returns:
            Predicted token indices (B, L)
        """
        result = self.forward(embeddings, temperature=temperature)
        return result["logits"].argmax(dim=-1)
