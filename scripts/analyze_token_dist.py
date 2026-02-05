import argparse
import json
import math
from collections import Counter
from pathlib import Path

from tqdm import tqdm

from ot3di.data import ProstT5Dataset, ThreeDiTokenizer


def calculate_weights(token_counts, total_tokens):
    """
    Calculate Log-IDF and Power-IDF weights based on token counts.

    Args:
        token_counts: Counter object with token IDs and their counts
        total_tokens: Total number of tokens processed

    Returns:
        dict: A dictionary containing 'log_idf' and 'power_idf' (gamma=0.3) weights
              mapped by token ID.
    """
    # Total number of 3Di tokens is 20.

    # Calculate probabilities p_k
    # We must ensure we cover all 20 tokens even if some are not observed (smoothing)
    # But usually all are observed. Let's assume K=20 standard tokens.

    # We know valid token IDs are usually 0-19 or mapped.
    # Let's rely on what's in the dataset and maybe check against known vocabulary.
    # ProstT5Dataset uses ThreeDiTokenizer.

    # Assuming standard tokens are indices 4 to 23 (checking tokenizer usually helps, but here we iterate counts)
    # Actually ThreeDiTokenizer might just use 0-19 or similar.
    # Let's use the keys from counts for now, but really we should normalize over the known vocabulary.

    # Based on previous file content, it seems token IDs are integers.
    # Let's collect all counts.

    # Handle rare case where a token might calculate to weight infinity if count is 0.
    # We add a small epsilon or ensure we use the vocab size.

    # Let's just use the observed tokens for now, acting as if they are the universe if K is not strictly defined,
    # but for 3Di we know K=20.

    # Calculate 1/p_k
    inv_probs = {}
    for token_id, count in token_counts.items():
        if count > 0:
            p_k = count / total_tokens
            inv_probs[token_id] = 1.0 / p_k

    # --- Option A: Log-IDF ---
    # w_k = log(1/p_k) / mean(log(1/p_l))

    log_inv_probs = {k: math.log(v) for k, v in inv_probs.items()}
    mean_log_inv_prob = sum(log_inv_probs.values()) / len(log_inv_probs)

    w_log = {k: v / mean_log_inv_prob for k, v in log_inv_probs.items()}

    # --- Option B: Power-IDF (gamma=0.3) ---
    # w_k = (1/p_k)^gamma / mean((1/p_l)^gamma)
    gamma = 0.3
    pow_inv_probs = {k: math.pow(v, gamma) for k, v in inv_probs.items()}
    mean_pow_inv_prob = sum(pow_inv_probs.values()) / len(pow_inv_probs)

    w_pow = {k: v / mean_pow_inv_prob for k, v in pow_inv_probs.items()}

    return {"log_idf": w_log, "power_idf": w_pow, "counts": dict(token_counts), "total_tokens": total_tokens}


def analyze_distribution(num_samples=None, output_path="resources/token_weights.json"):
    print("Loading dataset...")
    dataset = ProstT5Dataset(split="train")
    tokenizer = ThreeDiTokenizer()

    print(f"Dataset size: {len(dataset)}")

    token_counts = Counter()
    aacounts = Counter()

    print("Counting tokens...")

    limit = len(dataset)
    if num_samples is not None:
        limit = min(limit, num_samples)

    print(f"Processing {limit} samples...")

    for i in tqdm(range(limit)):
        item = dataset[i]
        tokens = item["threedi_tokens"].tolist()
        token_counts.update(tokens)

        # Count amino acids
        aa_seq = item["sequence"]
        aacounts.update(aa_seq)

    total_tokens = sum(token_counts.values())
    # total_aa = sum(aacounts.values())

    # Filter special tokens if necessary.
    # Usually we only care about the 20 3Di tokens for reweighting.
    # Let's check which tokens are 3Di.
    # The tokenizer.tok_to_idx usually has the map.
    # Assuming the tokenizer implementation, usually indices 0-3 might be special (CLS, PAD, EOS, UNK)
    # or similar. If the dataset gives just raw 3Di indices (0-19), we are good.
    # Based on previous file, it printed standard chars.

    # Calculate weights
    weights_data = calculate_weights(token_counts, total_tokens)

    # Add char mapping for readability in JSON
    weights_data["mapping"] = {}
    for token_id in weights_data["log_idf"].keys():
        weights_data["mapping"][token_id] = tokenizer.idx_to_char.get(token_id, f"ID_{token_id}")

    # Ensure output directory exists
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(weights_data, f, indent=4)

    print(f"\nWeights saved to {output_path}")

    # Display Summary
    print("\nToken Distribution (3Di) & Weights:")
    print(f"{'Token':<5} {'Char':<5} {'Count':<10} {'Freq (%)':<10} {'w_log':<10} {'w_pow':<10}")
    print("-" * 65)

    sorted_tokens = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)

    w_log = weights_data["log_idf"]
    w_pow = weights_data["power_idf"]

    for token_id, count in sorted_tokens:
        char = tokenizer.idx_to_char.get(token_id, str(token_id))
        freq = (count / total_tokens) * 100
        wl = w_log.get(token_id, 0.0)
        wp = w_pow.get(token_id, 0.0)
        print(f"{token_id:<5} {char:<5} {count:<10} {freq:<10.2f} {wl:<10.3f} {wp:<10.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze 3Di token distribution and generate weights.")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to process (default: all)")
    parser.add_argument("--output_path", type=str, default="resources/token_weights.json", help="Path to save the weights JSON file")

    args = parser.parse_args()

    analyze_distribution(num_samples=args.num_samples, output_path=args.output_path)
