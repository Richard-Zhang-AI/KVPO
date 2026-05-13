#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

CONFIG="$REPO_DIR/configs/train_long.yaml"
LOGDIR="$REPO_DIR/logs/train_long"
WANDB_SAVE_DIR="$REPO_DIR/wandb"

mkdir -p "$LOGDIR"
echo "CONFIG=$CONFIG"

torchrun \
  --nproc_per_node=2 \
  "$SCRIPT_DIR/train.py" \
  --config_path "$CONFIG" \
  --logdir "$LOGDIR" \
  --wandb-save-dir "$WANDB_SAVE_DIR" \
  --no-one-logger
