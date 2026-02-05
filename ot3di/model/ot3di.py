"""Main OT3Di model combining encoder, aligner, and predictor."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn

import torch.nn.functional as F
from .encoder import ESM2Encoder
from .ot_aligner import OTAligner
from .predictor import TokenPredictor
from .structure_encoder import StructuralEncoder


class OT3DiModel(nn.Module):
    """Sequence -> 3Di token prediction via OT alignment.

    Args:
        esm_model: HuggingFace ESM-2 model ID.
        embed_dim: Internal embedding dimension. If None, uses ESM hidden size.
        num_tokens: Number of 3Di tokens (vocabulary size).
        ot_epsilon: OT entropic regularization.
        ot_max_iters: Maximum Sinkhorn iterations.
        ot_backend: Sinkhorn backend ("triton" or "pytorch").
        freeze_esm: Whether to freeze ESM weights.
        ot_idf: Token reweighting scheme ("none", "log", "power").
    """

    def __init__(
        self,
        esm_model: str = "facebook/esm2_t33_650M_UR50D",
        embed_dim: int | None = None,
        num_tokens: int = 20,
        ot_epsilon: float = 0.1,
        ot_max_iters: int = 100,
        ot_backend: str = "triton",
        freeze_esm: bool = True,
        ot_idf: str = "none",
        structure_encoder_cfg: dict | None = None,
        **kwargs,  # Ignore extra config args
    ) -> None:
        super().__init__()

        # Encoder determines embed_dim if not specified
        self.encoder = ESM2Encoder(esm_model, embed_dim=None, freeze=freeze_esm)
        self.embed_dim = embed_dim if embed_dim is not None else self.encoder.embed_dim
        self.num_tokens = num_tokens

        # Add projection if embed_dim differs from ESM
        if embed_dim is not None and embed_dim != self.encoder.esm_dim:
            self.encoder.projection = nn.Linear(self.encoder.esm_dim, embed_dim)
            self.encoder.embed_dim = embed_dim
            self.embed_dim = embed_dim

        self.ot_aligner = OTAligner(ot_epsilon, ot_max_iters, backend=ot_backend)

        # Initialize Structure Encoder
        structure_cfg = structure_encoder_cfg or {}
        self.structure_encoder = StructuralEncoder(num_tokens=num_tokens, d_model=self.embed_dim, **structure_cfg)

        # Initialize Predictor with shared embeddings
        self.predictor = TokenPredictor(self.embed_dim, num_tokens, embedding_layer=self.structure_encoder.embedding)

        # Load token weights if requested
        self.register_buffer("token_weights", torch.ones(num_tokens))

        if ot_idf and ot_idf != "none":
            self._load_token_weights(ot_idf)

    def _load_token_weights(self, mode: str) -> None:
        """Load token weights from resources."""
        # Assume resources/token_weights.json acts as source of truth
        # In a real package, this should be included in package data
        weight_path = Path("resources/token_weights.json")
        if not weight_path.exists():
            print(f"Warning: Token weights not found at {weight_path}. Using uniform weights.")
            return

        with open(weight_path) as f:
            data = json.load(f)

        key = f"{mode}_idf"
        if key not in data:
            print(f"Warning: Weight mode {mode} not found in {weight_path}. Using uniform weights.")
            return

        weights_dict = data[key]

        # Weights dict keys are strings "0", "1", etc.
        # We need to map them to the tensor indices
        new_weights = torch.ones(self.num_tokens)

        # Check mapping or assume direct index mapping
        # Providing we strictly follow 0-19 indexing for 3Di
        for token_id_str, weight in weights_dict.items():
            idx = int(token_id_str)
            if idx < self.num_tokens:
                new_weights[idx] = weight

        self.token_weights.copy_(new_weights)
        print(f"Loaded {mode}-IDF weights for {len(weights_dict)} tokens.")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        target_tokens: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            input_ids: ESM tokenized sequences (B, L+2)
            attention_mask: Attention mask (B, L+2)
            target_tokens: Ground-truth 3Di tokens (B, L) for training

        Returns:
            logits, probs, embeddings, and OT outputs (if training)
        """
        enc = self.encoder(input_ids, attention_mask)
        embeddings, mask = enc["embeddings"], enc["mask"]

        result = {"embeddings": embeddings, "mask": mask}

        # OT alignment during training
        if target_tokens is not None:
            # Step 1: Structural Encoding (Contextualized targets)
            # (B, L_v) -> (B, L_v, D)
            token_emb = self.structure_encoder(target_tokens)

            # NOTE: Before we used predictor.get_token_embeddings which gave static embeddings.
            # Now token_emb is $V$ (contextualized).

            # Prepare weights if enabled
            target_weights = None
            # Check if weights are non-uniform
            if not torch.all(self.token_weights == 1.0):
                # Gather weights: (B, L_v)
                target_weights = self.token_weights[target_tokens]  # (B, L_v)

            ot = self.ot_aligner(embeddings, token_emb, mask, mask, target_weights=target_weights, return_cost=True)
            result.update({"P": ot["P"], "cost": ot["cost"]})

            # Soft Target Generation (Step 3)
            # q_ik = sum_j (P_ij / a_i) * one_hot(t_j = k)
            P = ot["P"]
            a_i = P.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            P_norm = P / a_i

            targets_one_hot = F.one_hot(target_tokens, num_classes=self.num_tokens).float()
            # (B, L_u, L_v) @ (B, L_v, K) -> (B, L_u, K)
            q = torch.matmul(P_norm, targets_one_hot)
            result["q"] = q

        # Token prediction
        pred = self.predictor(embeddings)
        result.update({"logits": pred["logits"], "probs": pred["probs"]})

        return result

    @torch.no_grad()
    def predict(self, sequences: list[str]) -> dict[str, torch.Tensor]:
        """Predict 3Di tokens from raw sequences."""
        self.eval()
        tokens = self.encoder.tokenize(sequences)
        tokens = {k: v.to(self.encoder.device) for k, v in tokens.items()}
        result = self.forward(tokens["input_ids"], tokens["attention_mask"])
        return {"tokens": result["logits"].argmax(dim=-1), "probs": result["probs"]}
