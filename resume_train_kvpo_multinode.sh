#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RUN_DIR="${RUN_DIR:-$SCRIPT_DIR/logs/sinnode/2026-04-11/run_20260411_105432_130}"

export CONFIG_PATH="${CONFIG_PATH:-$RUN_DIR/config_resolved.yaml}"
export TRAIN_SCRIPT="${TRAIN_SCRIPT:-$SCRIPT_DIR/train_kvpo_memflow.py}"
export RESUME_PATH="${RESUME_PATH:-$RUN_DIR/checkpoint_samples_000001200.pt}"

exec "$SCRIPT_DIR/train_kvpo_memflow_multinode.sh"
