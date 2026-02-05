"""3Di tokenizer for structure-based tokens."""

from __future__ import annotations

import numpy as np
import torch

# 3Di alphabet from FoldSeek (20 structural states)
THREEDI_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
THREEDI_TO_IDX = {c: i for i, c in enumerate(THREEDI_ALPHABET)}
IDX_TO_THREEDI = {i: c for i, c in enumerate(THREEDI_ALPHABET)}


class ThreeDiTokenizer:
    """Tokenizer for 3Di structure tokens.

    Converts between 3Di string representations and token indices.
    For actual 3Di computation from coordinates, use FoldSeek directly.

    Attributes:
        alphabet: The 3Di alphabet string.
        vocab_size: Number of tokens (20).
    """

    alphabet = THREEDI_ALPHABET
    vocab_size = len(THREEDI_ALPHABET)

    def __init__(self) -> None:
        self.char_to_idx = THREEDI_TO_IDX.copy()
        self.idx_to_char = IDX_TO_THREEDI.copy()

    def encode(self, threedi_string: str) -> torch.Tensor:
        """Convert 3Di string to token indices.

        Args:
            threedi_string: String of 3Di characters.

        Returns:
            Token indices as LongTensor (L,)
        """
        indices = [self.char_to_idx[c] for c in threedi_string]
        return torch.tensor(indices, dtype=torch.long)

    def decode(self, tokens: torch.Tensor | list[int]) -> str:
        """Convert token indices to 3Di string.

        Args:
            tokens: Token indices (L,) as tensor or list

        Returns:
            3Di string
        """
        if isinstance(tokens, torch.Tensor):
            if tokens.dim() > 1:
                raise ValueError("decode expects 1D tensor, use batch_decode for batch")
            tokens = tokens.tolist()
        return "".join(self.idx_to_char[int(t)] for t in tokens)

    def batch_encode(self, threedi_strings: list[str]) -> torch.Tensor:
        """Encode batch of 3Di strings.

        Args:
            threedi_strings: List of 3Di strings (must be same length).

        Returns:
            Token indices (B, L)
        """
        return torch.stack([self.encode(s) for s in threedi_strings])

    def batch_decode(self, tokens: torch.Tensor) -> list[str]:
        """Decode batch of token indices.

        Args:
            tokens: Token indices (B, L)

        Returns:
            List of 3Di strings
        """
        return [self.decode(t) for t in tokens]

    @staticmethod
    def compute_from_coordinates(
        ca_coords: np.ndarray,
        cb_coords: np.ndarray | None = None,
    ) -> str:
        """Compute 3Di tokens from backbone coordinates.

        This is a simplified version. For production, use FoldSeek's
        actual 3Di computation which considers local structural patterns.

        Args:
            ca_coords: CA atom coordinates (L, 3)
            cb_coords: CB atom coordinates (L, 3), optional

        Returns:
            3Di string (L,)

        Note:
            This is a placeholder. Real 3Di computation requires:
            1. Computing virtual center and direction vectors
            2. Discretizing local structural features
            3. Mapping to 3Di states via trained encoder

            For accurate 3Di tokens, use FoldSeek directly:
            `foldseek structureto3didescriptor input.pdb output.tsv`
        """
        raise NotImplementedError(
            "3Di computation from coordinates requires FoldSeek. "
            "Use `foldseek structureto3didescriptor` to precompute 3Di tokens, "
            "then load them as strings."
        )
