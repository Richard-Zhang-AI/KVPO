#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$SCRIPT_DIR"
CONFIG="$REPO_DIR/configs/train_kvpo_memflow.yaml"

# Read GPU configuration from the YAML config file, with fallback defaults
NUM_GPUS=$(python3 -c "
from omegaconf import OmegaConf
c = OmegaConf.load('$CONFIG')
print(getattr(c, 'num_gpus', 1))
" 2>/dev/null || echo 1)

GPU_IDS=$(python3 -c "
from omegaconf import OmegaConf
c = OmegaConf.load('$CONFIG')
print(getattr(c, 'gpu_ids', '0'))
" 2>/dev/null || echo "0")

echo "CONFIG=$CONFIG  NUM_GPUS=$NUM_GPUS  GPU_IDS=$GPU_IDS"

export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
export NCCL_TIMEOUT=3600

torchrun \
  --nproc_per_node="$NUM_GPUS" \
  "$REPO_DIR/train_kvpo_memflow.py" \
  --config_path "$CONFIG"
