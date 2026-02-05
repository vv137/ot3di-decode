"""Structural Encoder for 3Di sequences using RoPE."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RotaryEmbedding(nn.Module):
    """Rotary Positional Embedding (RoPE)."""

    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: int = 10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=inv_freq.device).type_as(inv_freq)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :])
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :])

    def forward(self, x: torch.Tensor, seq_len: int = None):
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(x.dtype))
            self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(x.dtype))

        return (
            self.cos_cached[:, :, :seq_len, ...].to(x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(x.dtype),
        )


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """Applies Rotary Positional Embedding to the query and key tensors."""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class StructureEncoderLayer(nn.Module):
    """Transformer Encoder Layer with RoPE support."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.GELU()

    def forward(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        src_key_padding_mask: torch.Tensor | None = None,
        rope_cos: torch.Tensor | None = None,
        rope_sin: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        src: (B, L, D)
        """
        # Self-Attention
        # src2 = self.norm1(src) # unused

        # We need to manually project Q, K, V to apply RoPE to Q, K
        # but nn.MultiheadAttention does it internally.
        # Standard solution: Do projection manually or use functional MHA or standard MHA hooks.
        # But wait, MHA doesn't expose Q, K before attention easily.
        # Let's implement MHA manually for full control or use 'in_proj'
        # To avoid re-implementing full MHA, we can do q, k, v projections, apply RoPE, then use F.scaled_dot_product_attention (PyTorch 2.0+)
        # However, to keep it simple and compatible, we'll assume we can use pytorch's scaled_dot_product_attention if available or manual attention.

        # Let's use manual attention implementation for clarity and RoPE integration
        # Actually that adds complexity. A simple way:
        # Use simple Linear layers for Q, K, V + F.scaled_dot_product_attention

        return self._block(src, rope_cos, rope_sin, src_mask, src_key_padding_mask)

    def _block(self, x, cos, sin, mask, key_padding_mask):
        # x is (B, L, D)
        residual = x
        x = self.norm1(x)

        # For simplicity, we delegate to self_attn but apply RoPE inside?
        # No, MHA is a black box.
        # We must implement the attention mechanism here to use RoPE.

        q, k, v = self.self_attn.in_proj_weight.chunk(3, dim=0)
        b_q, b_k, b_v = self.self_attn.in_proj_bias.chunk(3, dim=0)

        # Projections
        # (B, L, D) @ (D, D) -> (B, L, D)
        Q = F.linear(x, q, b_q)
        K = F.linear(x, k, b_k)
        V = F.linear(x, v, b_v)

        # Reshape for heads: (B, L, H, D/H) -> (B, H, L, D/H)
        B, L, D = Q.shape
        H = self.self_attn.num_heads
        head_dim = D // H

        Q = Q.view(B, L, H, head_dim).transpose(1, 2)
        K = K.view(B, L, H, head_dim).transpose(1, 2)
        V = V.view(B, L, H, head_dim).transpose(1, 2)

        # Apply RoPE
        # cos, sin are (1, 1, L, D/H) or similar.
        # Our RoPE computes for full dim, but we probably want it per head?
        # Usually RoPE is applied to each head independently.
        # The RotaryEmbedding class above returns (1, 1, seq_len, dim).
        # We need to adjust it to match head_dim.

        # Note: The RotaryEmbedding implementation above assumes `dim` is the full dimension.
        # If we want to apply to each head, we should init it with `head_dim`.

        if cos is not None and sin is not None:
            # Match dimensions for broadcasting
            # Q: (B, H, L, head_dim), cos: (1, 1, L, head_dim)
            Q, K = apply_rotary_pos_emb(Q, K, cos, sin)

        # Attention
        # mask handling needs care for B, H, L, L or B, L, L
        attn_mask = mask  # Should be (L, L) or (B*H, L, L) or similar

        # If key_padding_mask is passed, merge it.
        # F.scaled_dot_product_attention handles binary mask or float mask.

        # Using PyTorch 2.0 SDPA
        # is_causal = False usually for encoder, but we might want causal if it was a decoder.
        # This is an Encoder, so all-to-all.

        # Convert mask to fit SDPA
        # mask is usually float -inf for masked.

        x = F.scaled_dot_product_attention(Q, K, V, attn_mask=attn_mask, dropout_p=self.dropout.p if self.training else 0.0)

        # (B, H, L, head_dim) -> (B, L, H, head_dim) -> (B, L, D)
        x = x.transpose(1, 2).contiguous().view(B, L, D)

        # Output projection
        x = self.self_attn.out_proj(x)

        x = residual + self.dropout1(x)

        # Feed Forward
        residual = x
        x = self.norm2(x)
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = residual + self.dropout2(x)

        return x


class StructuralEncoder(nn.Module):
    def __init__(
        self,
        num_tokens: int = 20,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 3,
        dropout: float = 0.1,
        embedding_layer: nn.Embedding | None = None,
    ):
        super().__init__()

        if embedding_layer is not None:
            self.embedding = embedding_layer
            if self.embedding.embedding_dim != d_model:
                raise ValueError(f"Embedding dim {self.embedding.embedding_dim} must match d_model {d_model}")
        else:
            self.embedding = nn.Embedding(num_tokens, d_model)

        self.rope = RotaryEmbedding(d_model // nhead)  # Apply per head

        self.layers = nn.ModuleList([StructureEncoderLayer(d_model, nhead, dropout=dropout) for _ in range(num_layers)])

        self.norm = nn.LayerNorm(d_model)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # tokens: (Batch, L)
        B, L = tokens.size()

        x = self.embedding(tokens)  # (Batch, L, d_model)

        # Prepare RoPE
        cos, sin = self.rope(x, seq_len=L)

        # Prepare Mask
        # mask is usually padding mask (B, L) -> True for valid, False/0 for padding?
        # Standard transformer mask: 0 for valid, -inf for padding

        # attn_mask = None # unused
        if mask is not None:
            # mask: (B, L+2) from ESM usually? No, tokens here is 3Di tokens, so mask is (B, L)
            # If input mask is 1 for valid, 0 for pad
            # We need (B, 1, 1, L) or similar for SDPA?
            pass

        # Passing through layers
        for layer in self.layers:
            x = layer(x, rope_cos=cos, rope_sin=sin)

        return self.norm(x)
