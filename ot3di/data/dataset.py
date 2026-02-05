"""Dataset for ProstT5Dataset from HuggingFace."""

from __future__ import annotations

from typing import Callable

import torch
from datasets import load_dataset
from torch.utils.data import Dataset

from .tokenizer import ThreeDiTokenizer


class ProstT5Dataset(Dataset):
    """Dataset wrapper for Rostlab/ProstT5Dataset.

    Loads from local HuggingFace cache (non-streaming).

    Args:
        split: Dataset split ("train", "validation", "test").
        max_length: Maximum sequence length (longer ones are truncated).
    """

    # ProstT5 tokenizer mappings
    # AA: 3-22 → ALGVSREDTIPKFQNYMHWC
    # 3Di: 128-147 → algvsredtipkfqnymhwc (lowercase)
    # Special: 0=<pad>, 1=</s>, 2=<unk>

    AA_TOKENS = {
        3: "A",
        4: "L",
        5: "G",
        6: "V",
        7: "S",
        8: "R",
        9: "E",
        10: "D",
        11: "T",
        12: "I",
        13: "P",
        14: "K",
        15: "F",
        16: "Q",
        17: "N",
        18: "Y",
        19: "M",
        20: "H",
        21: "W",
        22: "C",
    }

    THREEDI_TOKENS = {
        128: "A",
        129: "L",
        130: "G",
        131: "V",
        132: "S",
        133: "R",
        134: "E",
        135: "D",
        136: "T",
        137: "I",
        138: "P",
        139: "K",
        140: "F",
        141: "Q",
        142: "N",
        143: "Y",
        144: "M",
        145: "H",
        146: "W",
        147: "C",
    }

    def __init__(
        self,
        split: str = "train",
        max_length: int = 512,
    ) -> None:
        self.split = split
        self.max_length = max_length
        self.tokenizer = ThreeDiTokenizer()

        # Load from cache (non-streaming), no filtering
        print(f"Loading ProstT5Dataset split='{split}' from cache...")
        self.dataset = load_dataset(
            "Rostlab/ProstT5Dataset",
            split=split,
            streaming=False,
        )
        print(f"Loaded {len(self.dataset)} samples")

    def __len__(self) -> int:
        return len(self.dataset)

    def _decode_aa(self, token_ids: list[int]) -> str:
        """Decode ProstT5 AA token IDs to string."""
        chars = []
        for tid in token_ids:
            if tid in self.AA_TOKENS:
                chars.append(self.AA_TOKENS[tid])
            # Skip special tokens (0, 1, 2)
        return "".join(chars)

    def _decode_3di(self, token_ids: list[int]) -> str:
        """Decode ProstT5 3Di token IDs to string."""
        chars = []
        for tid in token_ids:
            if tid in self.THREEDI_TOKENS:
                chars.append(self.THREEDI_TOKENS[tid])
            # Skip special tokens
        return "".join(chars)

    def __getitem__(self, idx: int) -> dict[str, str | torch.Tensor]:
        item = self.dataset[idx]

        # input_id_y = AA sequence, input_id_x = 3Di
        sequence = self._decode_aa(item["input_id_y"])
        threedi = self._decode_3di(item["input_id_x"])

        # Handle length mismatch
        min_len = min(len(sequence), len(threedi))
        sequence = sequence[:min_len]
        threedi = threedi[:min_len]

        # Truncate if too long
        if len(sequence) > self.max_length:
            sequence = sequence[: self.max_length]
            threedi = threedi[: self.max_length]

        if len(sequence) == 0:
            sequence = "A"
            threedi = "A"

        threedi_tokens = self.tokenizer.encode(threedi)

        return {
            "sequence": sequence,
            "threedi": threedi,
            "threedi_tokens": threedi_tokens,
        }


class StructureDataset(Dataset):
    """Dataset for paired (sequence, 3Di tokens) from JSON file."""

    def __init__(
        self,
        data_path: str,
        max_length: int = 512,
        tokenizer: ThreeDiTokenizer | None = None,
    ) -> None:
        import json
        from pathlib import Path

        self.data_path = Path(data_path)
        self.max_length = max_length
        self.tokenizer = tokenizer or ThreeDiTokenizer()

        with open(self.data_path) as f:
            raw_data = json.load(f)

        self.data = [item for item in raw_data if len(item["sequence"]) <= max_length]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, str | torch.Tensor]:
        item = self.data[idx]
        sequence = item["sequence"]
        threedi = item["threedi"]

        assert len(sequence) == len(threedi)
        threedi_tokens = self.tokenizer.encode(threedi)

        return {
            "sequence": sequence,
            "threedi": threedi,
            "threedi_tokens": threedi_tokens,
        }


def collate_fn(
    batch: list[dict],
    esm_batch_converter: Callable,
    pad_token_id: int = 0,
) -> dict[str, torch.Tensor | list[str]]:
    """Collate function for DataLoader."""
    sequences = [item["sequence"] for item in batch]
    threedi_tokens_list = [item["threedi_tokens"] for item in batch]

    # ESM tokenization
    esm_out = esm_batch_converter(sequences)
    esm_tokens = esm_out["input_ids"]
    attention_mask = esm_out.get("attention_mask")

    # Pad 3Di tokens
    max_len = max(len(t) for t in threedi_tokens_list)
    padded_tokens = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    mask = torch.zeros(len(batch), max_len, dtype=torch.bool)

    for i, tokens in enumerate(threedi_tokens_list):
        padded_tokens[i, : len(tokens)] = tokens
        mask[i, : len(tokens)] = True

    return {
        "esm_tokens": esm_tokens,
        "esm_attention_mask": attention_mask,
        "threedi_tokens": padded_tokens,
        "mask": mask,
        "sequences": sequences,
    }
