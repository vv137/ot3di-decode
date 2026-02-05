#!/bin/bash
# ============================================================
# OT3Di Local Multi-GPU Training Script
# ============================================================
# Usage: ./scripts/train_local.sh [NUM_GPUS]

set -e

NUM_GPUS=${1:-$(nvidia-smi -L | wc -l)}
CONFIG_PATH="${CONFIG_PATH:-configs/default.yaml}"
DATA_PATH="${DATA_PATH:-data/train.json}"
OUTPUT_DIR="${OUTPUT_DIR:-output/$(date +%Y%m%d_%H%M%S)}"

echo "=============================================="
echo "Training with $NUM_GPUS GPUs"
echo "Config: $CONFIG_PATH"
echo "Data: $DATA_PATH"
echo "Output: $OUTPUT_DIR"
echo "=============================================="

mkdir -p "$OUTPUT_DIR"

uv run torchrun \
    --standalone \
    --nproc_per_node="$NUM_GPUS" \
    -m ot3di.cli \
    --config "$CONFIG_PATH" \
    --data "$DATA_PATH" \
    --output "$OUTPUT_DIR"
