#!/bin/bash
#SBATCH --job-name=ot3di
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# ============================================================
# OT3Di Distributed Training Script for Slurm
# ============================================================

set -e

# Configuration
CONFIG_PATH="${CONFIG_PATH:-configs/default.yaml}"
DATA_PATH="${DATA_PATH:-data/train.json}"
OUTPUT_DIR="${OUTPUT_DIR:-output/$(date +%Y%m%d_%H%M%S)}"

# Create directories
mkdir -p logs
mkdir -p "$OUTPUT_DIR"

# Print job info
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs per node: $SLURM_GPUS_PER_NODE"
echo "Tasks per node: $SLURM_NTASKS_PER_NODE"
echo "Config: $CONFIG_PATH"
echo "Data: $DATA_PATH"
echo "Output: $OUTPUT_DIR"
echo "=============================================="

# Set up distributed training environment
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=${MASTER_PORT:-29500}
export WORLD_SIZE=$SLURM_NTASKS

echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo "World size: $WORLD_SIZE"
echo "=============================================="

# Run training with uv + torchrun
srun --kill-on-bad-exit=1 \
    uv run torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=$SLURM_GPUS_PER_NODE \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    -m ot3di.cli \
    --config "$CONFIG_PATH" \
    --data "$DATA_PATH" \
    --output "$OUTPUT_DIR"

echo "Training complete!"
