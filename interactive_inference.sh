#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$SCRIPT_DIR"
CONFIG_PATH="$REPO_DIR/configs/interactive_inference_ckpt.yaml"

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
  --nproc_per_node=4 \
  --master_port=29555 \
  "$REPO_DIR/interactive_inference.py" \
  --config_path "$CONFIG_PATH"
