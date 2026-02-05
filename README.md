# OT3DiDecode: Sequence to Structure Token Prediction via Optimal Transport

Predict 3Di structure tokens from protein sequences using Optimal Transport alignment with ESM-2.

## Overview

- **Input**: Protein sequence $S$ (length $L$)
- **Output**: 3Di structure token sequence $T$ (length $L$)

Uses [Triton-accelerated Sinkhorn](https://github.com/vv137/sinkhorn) for OT alignment and [ProstT5Dataset](https://huggingface.co/datasets/Rostlab/ProstT5Dataset) (17M sequences) for training.

## Installation

```bash
git clone https://github.com/vv137/OT3DiDecode.git
cd OT3DiDecode
uv sync
```

## Usage

### Training

```bash
# Single GPU
uv run ot3di-train --config configs/default.yaml --output output/

# Multi-GPU (local, 4 GPUs)
./scripts/train_local.sh 4

# Slurm cluster
sbatch scripts/train_slurm.sh
```

Checkpoints are saved:

- Every 1000 steps: `checkpoint_step_1000.pt`, `checkpoint_step_2000.pt`, ...
- End of each epoch: `checkpoint_epoch_1.pt`, `checkpoint_epoch_2.pt`, ...

### Evaluation

Evaluate on ProstT5 validation/test sets:

```bash
# Validation set (474 samples)
uv run python scripts/evaluate.py --checkpoint output/checkpoint_step_1000.pt --split valid

# Test set (474 samples)
uv run python scripts/evaluate.py --checkpoint output/checkpoint_epoch_1.pt --split test
```

Options:

- `--checkpoint`: Path to checkpoint file
- `--split`: `valid` or `test`
- `--batch_size`: Batch size (default: 16)
- `--device`: Device to use (default: cuda)

### Visualization

Visualize transport plans and predictions for debugging:

```bash
uv run python scripts/visualize.py --checkpoint output/checkpoint_step_1000.pt --split valid -n 5
```

Options:

- `--checkpoint`: Path to checkpoint file
- `--split`: `valid` or `test`
- `-n, --num_samples`: Number of samples to visualize (default: 5)
- `--max_length`: Max sequence length for visualization (default: 100)
- `--output_dir`: Output directory for images (default: `visualizations/`)

Generates PNG files showing:

1. **Transport Plan P**: OT alignment heatmap (sequence × 3Di)
2. **Prediction Probabilities**: Per-position token probability distribution
3. **Text Comparison**: Ground truth vs prediction with accuracy

### Token Reweighting

To address class imbalance in 3Di tokens, you can generate reweighting factors:

```bash
# Analyze distribution and generate weights (default: 20 sample)
uv run scripts/analyze_token_dist.py --num_samples 50000 --output_path resources/token_weights.json
```

This generates `resources/token_weights.json` containing Log-IDF and Power-IDF weights.
To use these weights during training, update your config:

```yaml
ot:
  idf: "log" # or "power", "none"
```

### Inference

```python
from ot3di.model import OT3DiModel
import torch

# Load from checkpoint
ckpt = torch.load("output/checkpoint_epoch_1.pt")
model = OT3DiModel(**ckpt["config"]["model"])
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# Predict
result = model.predict(["MKFLILLFNILCLFPVLAADNHGVGP"])
print(result["tokens"])  # Predicted 3Di tokens
```

---

## Architecture

```text
Sequence → ESM-2 Encoder → Sequence Embeddings ─┐
                                                ├── OT Alignment ──→ Soft Labels
3Di Tokens → Token Embeddings ──────────────────┘        ↓
                                                  Token Predictor → Predicted 3Di
```

**Key Components:**

1. **Dual-Stream Encoding**: Process sequence and structure tokens separately.
2. **Sinkhorn Alignment**: Compute optimal transport plan $P^*$ between sequence and structure.
3. **Soft Target Generation**: Create ideal token distributions $q$ from $P^*$.
4. **Inner Product Logit**: $\text{logit}_{ik} = \tau \cdot (u_i^\top e_k)$ with learnable temperature $\tau$.

### Loss Function

$$\mathcal{L} = \mathcal{L}_{CE} + \alpha \mathcal{L}_{OT}$$

- $\mathcal{L}_{OT} = \langle P^*, C \rangle$: Transport cost (alignment quality)
- $\mathcal{L}_{CE}$: Soft-label cross-entropy using OT-weighted labels

### Token Reweighting Implementation

When optional token reweighting (IDF) is enabled:

1. **Log-IDF** or **Power-IDF** weights are loaded from `resources/token_weights.json`.
2. The OT cost matrix $C$ is scaled by the target token weights $w_t$:
    $$C'_{ij} = w_{t_j} \cdot C_{ij}$$
3. This encourages the OT plan to align source positions to rare tokens (which have higher $w$), improving their representation in the soft targets.

---

## Configuration

```yaml
model:
  esm_model: "facebook/esm2_t33_650M_UR50D"
  num_tokens: 20
  predictor_mode: "embedding"  # Uses Inner Product Logit with learnable temperature
  freeze_esm: true

ot:
  epsilon: 1.0
  max_iters: 100
  backend: "triton"  # or "pytorch"

data:
  split: "train"
  max_length: 512
  num_workers: 4

training:
  lr: 1.0e-4
  batch_size: 16
  epochs: 100
  alpha: 0.5
  save_every_steps: 1000
```

---

## Project Structure

```text
OT3DiDecode/
├── ot3di/
│   ├── model/
│   │   ├── encoder.py      # ESM-2 encoder (HuggingFace)
│   │   ├── ot_aligner.py   # OT alignment (vv137/sinkhorn)
│   │   ├── predictor.py    # Token predictor
│   │   └── ot3di.py        # Main model
│   ├── data/
│   │   ├── dataset.py      # ProstT5Dataset loader
│   │   └── tokenizer.py    # 3Di tokenizer
│   ├── losses.py           # OT + soft CE loss
│   ├── train.py            # DDP training
│   └── cli.py              # CLI entry point
├── scripts/
│   ├── train_local.sh      # Local multi-GPU training
│   ├── train_slurm.sh      # Slurm training
│   ├── evaluate.py         # Evaluation script
│   └── visualize.py        # Visualization script
├── configs/
│   └── default.yaml        # Default config
└── pyproject.toml
```

---

## References

- [ProstT5: Bilingual language model for protein sequence and structure](https://academic.oup.com/nargab/article/6/4/lqae150/7901286)
- [Foldseek: 3Di Tokens](https://www.nature.com/articles/s41587-023-01773-0)
- [Sinkhorn Distances](https://arxiv.org/abs/1306.0895)
