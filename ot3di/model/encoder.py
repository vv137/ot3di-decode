"""ESM-2 sequence encoder using HuggingFace Transformers."""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoTokenizer, EsmModel


class ESM2Encoder(nn.Module):
    """Sequence encoder using ESM-2 from HuggingFace.

    Args:
        model_name: HuggingFace model ID.
        embed_dim: Output embedding dimension. If None, uses ESM native dim.
        freeze: Whether to freeze ESM weights.
    """

    def __init__(
        self,
        model_name: str = "facebook/esm2_t33_650M_UR50D",
        embed_dim: int | None = None,
        freeze: bool = True,
    ) -> None:
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.esm = EsmModel.from_pretrained(model_name)
        self.esm_dim = self.esm.config.hidden_size
        self.embed_dim = embed_dim or self.esm_dim

        self.projection = nn.Linear(self.esm_dim, self.embed_dim) if self.embed_dim != self.esm_dim else nn.Identity()

        if freeze:
            for param in self.esm.parameters():
                param.requires_grad = False

        self.freeze = freeze

    @property
    def device(self) -> torch.device:
        return next(self.esm.parameters()).device

    def tokenize(self, sequences: list[str]) -> dict[str, torch.Tensor]:
        """Tokenize sequences."""
        return self.tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Extract per-residue embeddings with proper BOS/EOS removal.

        ESM-2 adds BOS (cls) at position 0 and EOS at the end of each sequence.
        For padded batches, we need to handle variable-length sequences correctly.

        Args:
            input_ids: Tokenized sequences (B, L+2) including BOS/EOS
            attention_mask: Attention mask (B, L+2)

        Returns:
            embeddings: Per-residue embeddings (B, max_seq_len, D)
            mask: Valid position mask (B, max_seq_len)
        """
        with torch.set_grad_enabled(not self.freeze):
            outputs = self.esm(input_ids=input_ids, attention_mask=attention_mask)

        # Full hidden states including BOS and EOS: (B, L+2, D)
        hidden_states = outputs.last_hidden_state
        B, L_plus_2, D = hidden_states.shape

        if attention_mask is None:
            # No padding case: simple slice
            embeddings = hidden_states[:, 1:-1, :]
            mask = torch.ones(B, L_plus_2 - 2, dtype=torch.bool, device=hidden_states.device)
        else:
            # Padded batch: need to properly remove BOS/EOS per sequence
            # Compute actual sequence lengths (including BOS and EOS)
            seq_lens = attention_mask.sum(dim=1)  # (B,)

            # Max sequence length after removing BOS and EOS
            max_seq_len = (seq_lens - 2).max().item()

            # Pre-allocate output tensors
            embeddings = torch.zeros(
                B,
                max_seq_len,
                D,
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
            mask = torch.zeros(B, max_seq_len, dtype=torch.bool, device=hidden_states.device)

            for i in range(B):
                # Actual length of this sequence (excluding BOS/EOS)
                actual_len = seq_lens[i].item() - 2

                # Extract residue embeddings (skip BOS at 0, stop before EOS)
                # BOS is at position 0, residues are at 1 to actual_len, EOS is at actual_len+1
                embeddings[i, :actual_len] = hidden_states[i, 1 : actual_len + 1]
                mask[i, :actual_len] = True

        embeddings = self.projection(embeddings)

        return {"embeddings": embeddings, "mask": mask}

    @torch.no_grad()
    def encode_sequences(self, sequences: list[str]) -> dict[str, torch.Tensor]:
        """Convenience method to encode raw sequences."""
        tokens = self.tokenize(sequences)
        tokens = {k: v.to(self.device) for k, v in tokens.items()}
        return self.forward(tokens["input_ids"], tokens["attention_mask"])
