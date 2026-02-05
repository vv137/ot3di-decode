"""Evaluation script for OT3Di model on ProstT5 validation/test sets."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ot3di.data import ProstT5Dataset
from ot3di.data.dataset import collate_fn
from ot3di.metrics import compute_ot_metrics
from ot3di.model import OT3DiModel


def load_checkpoint(checkpoint_path: str | Path, device: torch.device) -> OT3DiModel:
    """Load model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    config = ckpt["config"]

    model = OT3DiModel(
        esm_model=config["model"]["esm_model"],
        embed_dim=config["model"].get("embed_dim"),  # Optional
        num_tokens=config["model"]["num_tokens"],
        ot_epsilon=config["ot"]["epsilon"],
        ot_max_iters=config["ot"]["max_iters"],
        ot_backend=config["ot"].get("backend", "triton"),
        freeze_esm=True,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    return model, ckpt


def evaluate(
    model: OT3DiModel,
    dataloader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate model on dataset.

    Returns:
        Dictionary with accuracy metrics.
    """
    model.eval()

    total_correct = 0
    total_tokens = 0
    total_samples = 0
    total_ot_cost = 0.0
    total_entropy = 0.0
    per_position_correct = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["esm_tokens"].to(device)
            attention_mask = batch.get("esm_attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            target_tokens = batch["threedi_tokens"].to(device)
            mask = batch["mask"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, target_tokens=target_tokens)
            predictions = outputs["logits"].argmax(dim=-1)

            # OT Metrics
            if "P" in outputs and "cost" in outputs:
                ot_metrics = compute_ot_metrics(outputs["P"], outputs["cost"], mask)

                # ot_metrics returns averages per token/sample
                # We need to accumulate them weighted by samples to get global average later
                # actually compute_ot_metrics output is already mean over batch.

                # total_ot_cost and total_entropy should act as accumulators for the mean
                # then divide by total_samples.
                total_ot_cost += ot_metrics["ot_cost"] * mask.size(0)
                total_entropy += ot_metrics["entropy"] * mask.size(0)

            # Per-token accuracy
            correct = (predictions == target_tokens) & mask
            total_correct += correct.sum().item()
            total_tokens += mask.sum().item()
            total_samples += mask.size(0)

            # Per-sample accuracy
            for i in range(mask.size(0)):
                sample_mask = mask[i]
                sample_correct = correct[i][sample_mask].float().mean().item()
                per_position_correct.append(sample_correct)

    token_accuracy = total_correct / total_tokens if total_tokens > 0 else 0
    sample_accuracy = sum(per_position_correct) / len(per_position_correct) if per_position_correct else 0
    avg_ot_cost = total_ot_cost / total_samples if total_samples > 0 else 0.0
    avg_entropy = total_entropy / total_samples if total_samples > 0 else 0.0

    return {
        "token_accuracy": token_accuracy,
        "sample_accuracy": sample_accuracy,
        "avg_ot_cost": avg_ot_cost,
        "avg_entropy": avg_entropy,
        "total_tokens": total_tokens,
        "total_samples": total_samples,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate OT3Di checkpoint")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Checkpoint path")
    parser.add_argument(
        "--split",
        type=str,
        default="valid",
        choices=["valid", "test"],
        help="Dataset split",
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--max_length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--device", type=str, default="cuda", help="Device")

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    model, ckpt = load_checkpoint(args.checkpoint, device)
    print(f"  Epoch: {ckpt.get('epoch', 'N/A')}, Step: {ckpt.get('global_step', 'N/A')}")

    # Load dataset
    # Note: ProstT5Dataset uses "valid" for validation
    split_name = "valid" if args.split == "valid" else "test"
    print(f"Loading ProstT5Dataset split='{split_name}'...")
    dataset = ProstT5Dataset(split=split_name, max_length=args.max_length)
    print(f"Dataset size: {len(dataset)}")

    # Create dataloader
    from functools import partial

    collate = partial(collate_fn, esm_batch_converter=model.encoder.tokenize)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,  # Avoid CUDA fork issues
        collate_fn=collate,
    )

    # Evaluate
    results = evaluate(model, dataloader, device)

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Split: {args.split}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Total samples: {results['total_samples']}")
    print(f"Total tokens: {results['total_tokens']}")
    print(f"Token Accuracy: {results['token_accuracy']:.4f} ({results['token_accuracy'] * 100:.2f}%)")
    print(f"Sample Accuracy: {results['sample_accuracy']:.4f} ({results['sample_accuracy'] * 100:.2f}%)")
    print(f"Avg OT Cost: {results['avg_ot_cost']:.4f}")
    print(f"Avg Entropy: {results['avg_entropy']:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
