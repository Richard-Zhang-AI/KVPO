#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CONFIG_PATH="${CONFIG_PATH:-$SCRIPT_DIR/configs/train_kvpo_longlive.yaml}"
export TRAIN_SCRIPT="${TRAIN_SCRIPT:-$SCRIPT_DIR/train_kvpo_longlive.py}"
export LOG_ROOT="${LOG_ROOT:-$SCRIPT_DIR/logs/multinode/longlive}"

exec bash "$SCRIPT_DIR/train_kvpo_memflow_multinode.sh" "$@"
