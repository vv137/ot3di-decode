"""Visualize OT transport plans and predictions for debugging."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import torch

from ot3di.data import ProstT5Dataset, ThreeDiTokenizer
from ot3di.metrics import compute_ot_metrics
from ot3di.model import OT3DiModel


def load_checkpoint(checkpoint_path: str | Path, device: torch.device):
    """Load model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    config = ckpt["config"]

    model = OT3DiModel(
        esm_model=config["model"]["esm_model"],
        embed_dim=config["model"].get("embed_dim"),
        num_tokens=config["model"]["num_tokens"],
        ot_epsilon=config["ot"]["epsilon"],
        ot_max_iters=config["ot"]["max_iters"],
        ot_backend=config["ot"].get("backend", "pytorch"),  # Use pytorch for viz
        freeze_esm=True,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    return model, ckpt


def visualize_sample(
    model: OT3DiModel,
    sample: dict,
    tokenizer: ThreeDiTokenizer,
    device: torch.device,
    save_path: Path,
):
    """Visualize a single sample's transport plan and predictions."""
    model.eval()

    sequence = sample["sequence"]
    threedi_gt = sample["threedi"]
    threedi_tokens = sample["threedi_tokens"]

    # Tokenize and forward
    esm_tokens = model.encoder.tokenize([sequence])
    input_ids = esm_tokens["input_ids"].to(device)
    attention_mask = esm_tokens["attention_mask"].to(device)
    target_tokens = threedi_tokens.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask, target_tokens)

    # Get predictions
    probs = outputs["probs"][0].cpu()  # (L, num_tokens)
    logits = outputs["logits"][0].cpu()
    predictions = logits.argmax(dim=-1)
    pred_threedi = tokenizer.decode(predictions.tolist())

    # Get transport plan if available
    P = outputs.get("P")
    if P is not None:
        P = P[0].cpu().numpy()  # (L, L)

    mask = outputs["mask"][0].cpu()
    seq_len = mask.sum().item()

    # Calculate OT metrics
    ot_cost_val = 0.0
    entropy_val = 0.0

    if "P" in outputs and "cost" in outputs:
        # P and C in outputs are (B, L, L) tensors
        # We need to slice them for the current sample [0] and keep dim 0 as batch=1
        metrics = compute_ot_metrics(
            outputs["P"].cpu(),  # move to cpu if needed, logic handles devices but ensure compatibility
            outputs["cost"].cpu(),
            outputs["mask"].cpu(),
        )
        ot_cost_val = metrics["ot_cost"]
        entropy_val = metrics["entropy"]

    # Truncate to actual sequence length
    sequence = sequence[:seq_len]
    threedi_gt = threedi_gt[:seq_len]
    pred_threedi = pred_threedi[:seq_len]

    # Calculate accuracy
    correct = sum(1 for a, b in zip(pred_threedi, threedi_gt) if a == b)
    accuracy = correct / len(threedi_gt) * 100

    # Create figure with GridSpec
    # Layout:
    # If P exists: Left (Plan), Top-Right (Probs), Bottom-Right (Text)
    # If P missing: Top (Probs), Bottom (Text)

    if P is not None:
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])
        ax_plan = fig.add_subplot(gs[:, 0])
        ax_probs = fig.add_subplot(gs[0, 1])
        ax_text = fig.add_subplot(gs[1, 1])
    else:
        fig = plt.figure(figsize=(12, 10))
        gs = fig.add_gridspec(2, 1, height_ratios=[1, 1])
        ax_probs = fig.add_subplot(gs[0, 0])
        ax_text = fig.add_subplot(gs[1, 0])

    # 1. Transport plan heatmap
    if P is not None:
        P_viz = P[:seq_len, :seq_len].T  # Transpose to have Source on X, Target on Y
        # Use aspect="equal" to show the L x L nature correctly
        im = ax_plan.imshow(P_viz, aspect="equal", cmap="viridis")
        # X-label usually handled by shared axis, but we can set it for clarity or hide it
        ax_plan.set_xlabel("Source position (Input Sequence)")
        ax_plan.set_ylabel("Target position (Ground Truth Sequence)")
        ax_plan.set_title(f"Plan - Acc: {accuracy:.1f}% | Cost: {ot_cost_val:.3f} | Ent: {entropy_val:.3f}")
        plt.colorbar(im, ax=ax_plan, fraction=0.046, pad=0.04)

        # Add sequence labels
        # Only show ticks if sequence is not too long, or decimate them
        if seq_len <= 150:
            ax_plan.set_yticks(range(seq_len))
            ax_plan.set_yticklabels(list(threedi_gt), fontsize=6)
            ax_plan.set_xticks(range(seq_len))
            ax_plan.set_xticklabels(list(sequence), fontsize=6)
        else:
            # For long sequences, just show indices
            ax_plan.set_ylabel("Target Index")
            ax_plan.set_xlabel("Source Index")

    # 2. Prediction probabilities heatmap
    probs_viz = probs[:seq_len].numpy().T  # (num_tokens, L)
    im2 = ax_probs.imshow(probs_viz, aspect="auto", cmap="Blues")
    ax_probs.set_xlabel(f"Source Position (Input Sequence: {len(sequence)} aa)")
    ax_probs.set_ylabel("3Di Token (Vocabulary)")
    ax_probs.set_title("Prediction Probabilities")

    # Y-axis is fixed to 20 tokens
    ax_probs.set_yticks(range(20))
    ax_probs.set_yticklabels(list(tokenizer.alphabet), fontsize=8)
    plt.colorbar(im2, ax=ax_probs, fraction=0.046, pad=0.04)

    # Add red hollow square for ground truth
    # probs_viz is (20, L), x-axis is sequence (0..L-1), y-axis is token (0..19)
    for i in range(seq_len):
        gt_token = threedi_gt[i]
        # Find index of gt_token in tokenizer.alphabet
        # Robust way: tokenizer.token_to_id might give integer ID, but alphabet order is what matters for imshow
        try:
            gt_idx = tokenizer.alphabet.index(gt_token)
            # Create small opaque square
            # Box-size 0.3x0.3 centered at integer coords.
            # Bottom-left = (i - 0.15, gt_idx - 0.15) for 0.3 size
            rect = patches.Rectangle((i - 0.15, gt_idx - 0.15), 0.3, 0.3, linewidth=0, edgecolor="none", facecolor="red", alpha=1.0)
            ax_probs.add_patch(rect)
        except ValueError:
            pass  # GT token not in alphabet? Should not happen usually

    # X-axis labels (Input Sequence)
    if seq_len <= 150:
        ax_probs.set_xticks(range(seq_len))
        ax_probs.set_xticklabels(list(sequence), fontsize=6)
    else:
        ax_probs.set_xlabel("Source Index")

    # 3. Text comparison
    ax_text.axis("off")

    # text = f"Sequence ({len(sequence)} aa):\n{sequence}\n\n"

    # Construct alignment-like string
    # GT:   ...
    # Match:| . | ...
    # Pred: ...

    chunk_size = 60  # Wrap lines if too long

    formatted_text = ""
    formatted_text += f"Accuracy: {correct}/{len(threedi_gt)} = {accuracy:.1f}%\n"
    if P is not None:
        formatted_text += f"Avg OT Cost: {ot_cost_val:.4f}\n"
        formatted_text += f"Avg Entropy: {entropy_val:.4f}\n"
    formatted_text += "\n"

    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        chunk_gt = threedi_gt[start:end]
        chunk_pred = pred_threedi[start:end]

        chunk_match = ""
        for gt, pr in zip(chunk_gt, chunk_pred):
            if gt == pr:
                chunk_match += "|"
            else:
                chunk_match += "."

        formatted_text += f"GT:   {chunk_gt}\n"
        formatted_text += f"      {chunk_match}\n"
        formatted_text += f"Pred: {chunk_pred}\n\n"

    ax_text.text(
        0.01,
        0.99,
        formatted_text,
        transform=ax_text.transAxes,
        fontsize=10,
        fontfamily="monospace",
        verticalalignment="top",
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    return {
        "sequence": sequence,
        "ground_truth": threedi_gt,
        "prediction": pred_threedi,
        "accuracy": accuracy,
    }


def main():
    parser = argparse.ArgumentParser(description="Visualize OT3Di predictions")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", type=str, default="valid", choices=["valid", "test"])
    parser.add_argument("--num_samples", "-n", type=int, default=5)
    parser.add_argument("--output_dir", type=Path, default=Path("visualizations"))
    parser.add_argument("--max_length", type=int, default=200, help="Max seq length for viz")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint: {args.checkpoint}")
    model, ckpt = load_checkpoint(args.checkpoint, device)
    print(f"  Epoch: {ckpt.get('epoch', 'N/A')}, Step: {ckpt.get('global_step', 'N/A')}")

    print(f"Loading ProstT5Dataset split='{args.split}'...")
    dataset = ProstT5Dataset(split=args.split, max_length=args.max_length)
    print(f"Dataset size: {len(dataset)}")

    tokenizer = ThreeDiTokenizer()

    # Select random samples
    indices = random.sample(range(len(dataset)), min(args.num_samples, len(dataset)))

    print(f"\nVisualizing {len(indices)} samples...")
    for i, idx in enumerate(indices):
        sample = dataset[idx]
        save_path = args.output_dir / f"sample_{i + 1}_idx{idx}.png"

        result = visualize_sample(model, sample, tokenizer, device, save_path)
        print(f"  [{i + 1}/{len(indices)}] idx={idx}, len={len(result['sequence'])}, accuracy={result['accuracy']:.1f}%")
        print(f"    Saved: {save_path}")

    print(f"\nAll visualizations saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
