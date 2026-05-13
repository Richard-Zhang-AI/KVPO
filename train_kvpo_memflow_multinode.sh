#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$SCRIPT_DIR}"
CONFIG_PATH="${CONFIG_PATH:-$REPO_DIR/configs/train_kvpo_memflow.yaml}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-$REPO_DIR/train_kvpo_memflow.py}"
LOG_ROOT="${LOG_ROOT:-$REPO_DIR/logs/multinode}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%d_%H%M%S)}"
RESUME_PATH="${RESUME_PATH:-}"

read_config_value() {
  local key="$1"
  local default_value="$2"
  python3 - "$CONFIG_PATH" "$key" "$default_value" <<'PY'
import sys
from omegaconf import OmegaConf

config_path, key, default_value = sys.argv[1:4]
cfg = OmegaConf.load(config_path)
value = OmegaConf.select(cfg, key, default=default_value)
print(value)
PY
}

resolve_multinode_from_config() {
  python3 - "$CONFIG_PATH" <<'PY'
import os
import shlex
import sys
from omegaconf import OmegaConf


def emit_scalar(name, value):
    print(f"{name}={shlex.quote(str(value))}")


def emit_array(name, values):
    joined = " ".join(shlex.quote(str(v)) for v in values)
    print(f"{name}=({joined})")


config_path = sys.argv[1]
cfg = OmegaConf.load(config_path)
multinode = OmegaConf.select(cfg, "multinode", default=None)
nodes_cfg = OmegaConf.select(multinode, "nodes", default=None) if multinode is not None else None
if nodes_cfg is None:
    print("CONFIG_HAS_MULTINODE=0")
    sys.exit(0)

nodes = OmegaConf.to_container(nodes_cfg, resolve=True)
if not isinstance(nodes, dict):
    sys.exit("multinode.nodes must be a mapping from alias to ssh config")

master_alias = os.environ.get("MASTER_NODE_ALIAS") or OmegaConf.select(multinode, "master", default=None)
worker_aliases_raw = os.environ.get("WORKER_NODE_ALIASES", "").strip()
if worker_aliases_raw:
    worker_aliases = [item.strip() for item in worker_aliases_raw.replace(",", " ").split() if item.strip()]
else:
    worker_aliases = OmegaConf.select(multinode, "workers", default=[])
    worker_aliases = list(worker_aliases or [])

if not master_alias:
    sys.exit("multinode.master is required when using multinode.nodes")
if master_alias not in nodes:
    sys.exit(f"Unknown multinode.master alias: {master_alias}")
if not worker_aliases:
    sys.exit("multinode.workers must contain at least one worker alias")

unknown_workers = [alias for alias in worker_aliases if alias not in nodes]
if unknown_workers:
    sys.exit(f"Unknown worker aliases in multinode.workers: {', '.join(unknown_workers)}")
if len(set(worker_aliases)) != len(worker_aliases):
    sys.exit("multinode.workers contains duplicate aliases")
if master_alias in worker_aliases:
    sys.exit("multinode.master cannot also appear in multinode.workers")


def resolve_node(alias):
    node = nodes[alias] or {}
    host = node.get("ssh_host") or node.get("host") or node.get("hostname")
    port = node.get("ssh_port") or node.get("port")
    user = node.get("ssh_user") or node.get("user") or "root"
    if not host:
        sys.exit(f"Node {alias} is missing ssh_host/host/hostname")
    if port is None:
        sys.exit(f"Node {alias} is missing ssh_port/port")
    return {
        "host": str(host),
        "port": str(port),
        "user": str(user),
    }


master = resolve_node(master_alias)
workers = [resolve_node(alias) for alias in worker_aliases]

emit_scalar("CONFIG_HAS_MULTINODE", 1)
emit_scalar("CONFIG_MASTER_ALIAS", master_alias)
emit_scalar("CONFIG_MASTER_HOSTNAME", master["host"])
emit_scalar("CONFIG_MASTER_SSH_PORT", master["port"])
emit_scalar("CONFIG_MASTER_USER", master["user"])
emit_array("CONFIG_WORKER_ALIASES", worker_aliases)
emit_array("CONFIG_WORKER_HOSTNAMES", [worker["host"] for worker in workers])
emit_array("CONFIG_WORKER_SSH_PORTS", [worker["port"] for worker in workers])
emit_array("CONFIG_WORKER_USERS", [worker["user"] for worker in workers])
PY
}

#    MASTER_HOSTNAME / MASTER_SSH_PORT / MASTER_USER
#    WORKER_HOSTNAMES / WORKER_SSH_PORTS / WORKER_USERS
#    MASTER_NODE_ALIAS / WORKER_NODE_ALIASES
DIRECT_MASTER_OVERRIDE=0
DIRECT_WORKER_OVERRIDE=0
if [[ "${MASTER_HOSTNAME+x}" == "x" || "${MASTER_SSH_PORT+x}" == "x" || "${MASTER_USER+x}" == "x" ]]; then
  DIRECT_MASTER_OVERRIDE=1
fi
if [[ "${WORKER_HOSTNAMES+x}" == "x" || "${WORKER_SSH_PORTS+x}" == "x" || "${WORKER_USERS+x}" == "x" ]]; then
  DIRECT_WORKER_OVERRIDE=1
fi

CONFIG_HAS_MULTINODE=0
CONFIG_MASTER_ALIAS=""
CONFIG_WORKER_ALIASES=()
CONFIG_WORKER_HOSTNAMES=()
CONFIG_WORKER_SSH_PORTS=()
CONFIG_WORKER_USERS=()
if [[ $DIRECT_MASTER_OVERRIDE -eq 0 && $DIRECT_WORKER_OVERRIDE -eq 0 ]]; then
  eval "$(resolve_multinode_from_config)"
fi

MASTER_NODE_ALIAS="${MASTER_NODE_ALIAS:-}"
WORKER_NODE_ALIASES=()
if [[ "$CONFIG_HAS_MULTINODE" == "1" ]]; then
  MASTER_NODE_ALIAS="$CONFIG_MASTER_ALIAS"
  WORKER_NODE_ALIASES=("${CONFIG_WORKER_ALIASES[@]}")
  MASTER_HOSTNAME="$CONFIG_MASTER_HOSTNAME"
  MASTER_SSH_PORT="$CONFIG_MASTER_SSH_PORT"
  MASTER_USER="$CONFIG_MASTER_USER"
  WORKER_HOSTNAMES=("${CONFIG_WORKER_HOSTNAMES[@]}")
  WORKER_SSH_PORTS=("${CONFIG_WORKER_SSH_PORTS[@]}")
  WORKER_USERS=("${CONFIG_WORKER_USERS[@]}")
else
  if [[ -z "${MASTER_HOSTNAME:-}" || -z "${MASTER_SSH_PORT:-}" || -z "${MASTER_USER:-}" || \
        -z "${WORKER_HOSTNAMES:-}" || -z "${WORKER_SSH_PORTS:-}" || -z "${WORKER_USERS:-}" ]]; then
    echo "No multinode.nodes config found. Set MASTER_HOSTNAME/MASTER_SSH_PORT/MASTER_USER and WORKER_HOSTNAMES/WORKER_SSH_PORTS/WORKER_USERS, or add a multinode section to CONFIG_PATH." >&2
    exit 1
  fi
  MASTER_HOSTNAME="${MASTER_HOSTNAME}"
  MASTER_SSH_PORT="${MASTER_SSH_PORT}"
  MASTER_USER="${MASTER_USER}"
  WORKER_HOSTNAMES=(${WORKER_HOSTNAMES})
  WORKER_SSH_PORTS=(${WORKER_SSH_PORTS})
  WORKER_USERS=(${WORKER_USERS})
fi

if [[ ${#WORKER_HOSTNAMES[@]} -ne ${#WORKER_SSH_PORTS[@]} ]]; then
  echo "WORKER_HOSTNAMES and WORKER_SSH_PORTS must have the same length" >&2
  exit 1
fi
if [[ ${#WORKER_USERS[@]} -ne ${#WORKER_HOSTNAMES[@]} ]]; then
  if [[ ${#WORKER_USERS[@]} -eq 1 ]]; then
    worker_user="${WORKER_USERS[0]}"
    WORKER_USERS=()
    for _ in "${WORKER_HOSTNAMES[@]}"; do
      WORKER_USERS+=("$worker_user")
    done
  else
    echo "WORKER_USERS must have length 1 or match WORKER_HOSTNAMES" >&2
    exit 1
  fi
fi
NNODES=$((1 + ${#WORKER_HOSTNAMES[@]}))
MASTER_PORT="${MASTER_PORT:-29500}"
REMOTE_ENV_SETUP="${REMOTE_ENV_SETUP:-}"
SSH_OPTS="${SSH_OPTS:--o StrictHostKeyChecking=no}"
# Password mode is opt-in. Prefer SSH keys for open-source/default use.
SSH_PASSWORD="${SSH_PASSWORD:-}"
PREFERRED_MASTER_SUBNET="${PREFERRED_MASTER_SUBNET:-11.1.}"
NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-auto}"
GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-auto}"
NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
NCCL_CROSS_NIC="${NCCL_CROSS_NIC:-0}"
NCCL_NET="${NCCL_NET:-Socket}"
NCCL_NET_PLUGIN="${NCCL_NET_PLUGIN:-none}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-memflow}"
CONDA_ENV_PATH="${CONDA_ENV_PATH:-${CONDA_PREFIX:-}}"
CONDA_SH_PATH="${CONDA_SH_PATH:-}"
TORCHRUN_BIN="${TORCHRUN_BIN:-$(command -v torchrun 2>/dev/null || true)}"
PYTHON_BIN="${PYTHON_BIN:-${CONDA_PREFIX:+$CONDA_PREFIX/bin/python}}"
if [[ -z "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(python -c 'import sys; print(sys.executable)' 2>/dev/null || true)"
fi
if [[ -z "$CONDA_ENV_PATH" && -n "$PYTHON_BIN" ]]; then
  python_bindir="$(dirname "$PYTHON_BIN")"
  python_env_root="$(dirname "$python_bindir")"
  if [[ -d "$python_env_root" ]]; then
    CONDA_ENV_PATH="$python_env_root"
  fi
fi
if [[ -z "$CONDA_SH_PATH" && -n "${CONDA_EXE:-}" ]]; then
  candidate_conda_sh="$(cd "$(dirname "$CONDA_EXE")/../etc/profile.d" 2>/dev/null && pwd)/conda.sh"
  if [[ -f "$candidate_conda_sh" ]]; then
    CONDA_SH_PATH="$candidate_conda_sh"
  fi
fi
MASTER_PASSWORD="${MASTER_PASSWORD-$SSH_PASSWORD}"
if [[ "${WORKER_PASSWORDS+x}" == "x" ]]; then
  if [[ -n "$WORKER_PASSWORDS" ]]; then
    WORKER_PASSWORDS=($WORKER_PASSWORDS)
  else
    WORKER_PASSWORDS=("")
  fi
else
  WORKER_PASSWORDS=($SSH_PASSWORD)
fi
LOG_DIR="${LOG_DIR:-$LOG_ROOT/$RUN_ID}"
mkdir -p "$LOG_DIR"

if [[ ${#WORKER_PASSWORDS[@]} -ne ${#WORKER_HOSTNAMES[@]} ]]; then
  if [[ ${#WORKER_PASSWORDS[@]} -eq 1 ]]; then
    worker_password="${WORKER_PASSWORDS[0]}"
    WORKER_PASSWORDS=()
    for _ in "${WORKER_HOSTNAMES[@]}"; do
      WORKER_PASSWORDS+=("$worker_password")
    done
  else
    echo "WORKER_PASSWORDS must have length 1 or match WORKER_HOSTNAMES" >&2
    exit 1
  fi
fi

NUM_GPUS="${NUM_GPUS:-$(read_config_value num_gpus 1)}"
GPU_IDS="${GPU_IDS:-$(read_config_value gpu_ids 0)}"

run_ssh() {
  local user="$1"
  local host="$2"
  local port="$3"
  local password="$4"
  local remote_cmd="$5"
  if [[ -n "$password" ]]; then
    if command -v sshpass >/dev/null 2>&1; then
      sshpass -p "$password" ssh $SSH_OPTS \
        -o PreferredAuthentications=password \
        -o PubkeyAuthentication=no \
        -o NumberOfPasswordPrompts=1 \
        -p "$port" "${user}@${host}" "$remote_cmd"
    else
      echo "sshpass not found; falling back to plain ssh for ${user}@${host}:${port}" >&2
      ssh $SSH_OPTS -p "$port" "${user}@${host}" "$remote_cmd"
    fi
  else
    ssh $SSH_OPTS -p "$port" "${user}@${host}" "$remote_cmd"
  fi
}

MASTER_ADDR="${MASTER_ADDR:-$(run_ssh "$MASTER_USER" "$MASTER_HOSTNAME" "$MASTER_SSH_PORT" "$MASTER_PASSWORD" "PREFERRED_MASTER_SUBNET='$PREFERRED_MASTER_SUBNET' python - <<'PY'
import os
import subprocess

preferred = os.environ.get('PREFERRED_MASTER_SUBNET', '11.1.')
ips = subprocess.check_output(['hostname', '-I'], text=True).split()
for ip in ips:
    if ip.startswith(preferred):
        print(ip)
        break
else:
    print(ips[0] if ips else '')
PY" )}"

resolve_master_port() {
  local requested_port="$1"
  local resolved_port
  resolved_port="$(run_ssh "$MASTER_USER" "$MASTER_HOSTNAME" "$MASTER_SSH_PORT" "$MASTER_PASSWORD" "REQUESTED_PORT='$requested_port' python - <<'PY'
import errno
import os
import socket
import sys

requested = int(os.environ['REQUESTED_PORT'])

for port in range(requested, requested + 100):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind(('', port))
    except OSError as exc:
        sock.close()
        if exc.errno == errno.EADDRINUSE:
            continue
        raise
    else:
        sock.close()
        print(port)
        break
else:
    sys.exit(f'Unable to find a free master port in [{requested}, {requested + 99}]')
PY" )"
  if [[ "$resolved_port" != "$requested_port" ]]; then
    echo "MASTER_PORT $requested_port is busy on ${MASTER_HOSTNAME}; using $resolved_port instead" >&2
  fi
  echo "$resolved_port"
}

MASTER_PORT="$(resolve_master_port "$MASTER_PORT")"

echo "REPO_DIR=$REPO_DIR"
echo "CONFIG_PATH=$CONFIG_PATH"
echo "TRAIN_SCRIPT=$TRAIN_SCRIPT"
echo "RESUME_PATH=${RESUME_PATH:-<none>}"
echo "MASTER_ALIAS=${MASTER_NODE_ALIAS:-<unset>}"
echo "MASTER=${MASTER_USER}@${MASTER_HOSTNAME}:${MASTER_SSH_PORT}"
if [[ ${#WORKER_NODE_ALIASES[@]} -gt 0 ]]; then
  echo "WORKER_ALIASES=${WORKER_NODE_ALIASES[*]}"
fi
echo "WORKERS=${WORKER_HOSTNAMES[*]}"
echo "WORKER_PORTS=${WORKER_SSH_PORTS[*]}"
echo "PASSWORD_MODE=$([[ -n "$MASTER_PASSWORD" ]] && echo enabled || echo disabled)"
echo "CONDA_ENV_NAME=$CONDA_ENV_NAME"
echo "CONDA_ENV_PATH=${CONDA_ENV_PATH:-<unset>}"
echo "CONDA_SH_PATH=${CONDA_SH_PATH:-<auto>}"
echo "PYTHON_BIN=${PYTHON_BIN:-<auto>}"
echo "TORCHRUN_BIN=${TORCHRUN_BIN:-torchrun}"
echo "PREFERRED_MASTER_SUBNET=$PREFERRED_MASTER_SUBNET"
echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"
echo "NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME"
echo "GLOO_SOCKET_IFNAME=$GLOO_SOCKET_IFNAME"
echo "NCCL_DEBUG=$NCCL_DEBUG"
echo "NCCL_IB_DISABLE=$NCCL_IB_DISABLE"
echo "NCCL_CROSS_NIC=$NCCL_CROSS_NIC"
echo "NCCL_NET=$NCCL_NET"
echo "NCCL_NET_PLUGIN=$NCCL_NET_PLUGIN"
echo "NNODES=$NNODES"
echo "NUM_GPUS=$NUM_GPUS"
echo "GPU_IDS=$GPU_IDS"
echo "LOG_DIR=$LOG_DIR"

build_remote_command() {
  local node_rank="$1"
  local env_setup_block
  local python_bindir=""
  if [[ -n "$PYTHON_BIN" ]]; then
    python_bindir="$(dirname "$PYTHON_BIN")"
  fi
  if [[ -n "$REMOTE_ENV_SETUP" ]]; then
    env_setup_block="$REMOTE_ENV_SETUP"
  else
    env_setup_block='if [[ -n "'"$PYTHON_BIN"'" && -x "'"$PYTHON_BIN"'" ]]; then
  export PATH="'"$python_bindir"':$PATH"
elif [[ -n "'"$CONDA_ENV_PATH"'" && -d "'"$CONDA_ENV_PATH"'" ]]; then
  if [[ -n "'"$CONDA_SH_PATH"'" && -f "'"$CONDA_SH_PATH"'" ]]; then
    source "'"$CONDA_SH_PATH"'"
  elif command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
  elif [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
  else
    echo "Unable to find conda.sh on $(hostname)" >&2
    exit 1
  fi
  conda activate "'"$CONDA_ENV_PATH"'"
elif [[ -n "'"$CONDA_ENV_NAME"'" ]]; then
  if [[ -n "'"$CONDA_SH_PATH"'" && -f "'"$CONDA_SH_PATH"'" ]]; then
    source "'"$CONDA_SH_PATH"'"
  elif command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
  elif [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
  else
    echo "Unable to find conda.sh on $(hostname)" >&2
    exit 1
  fi
  conda activate "'"$CONDA_ENV_NAME"'"
else
  echo "No usable PYTHON_BIN, CONDA_ENV_PATH, or CONDA_ENV_NAME was provided" >&2
  exit 1
fi'
  fi
  cat <<EOF
set -eo pipefail
cd "$REPO_DIR"
# Some conda activation hooks assume unset toolchain variables are allowed.
set +u
${env_setup_block}
set -u
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
REMOTE_PYTHON_BIN=""
REMOTE_TORCHRUN_BIN=""
if command -v python >/dev/null 2>&1; then
  REMOTE_PYTHON_BIN="\$(command -v python)"
fi
if command -v torchrun >/dev/null 2>&1; then
  REMOTE_TORCHRUN_BIN="\$(command -v torchrun)"
fi
if [[ -n "\$REMOTE_PYTHON_BIN" ]]; then
  LAUNCHER_CMD=("\$REMOTE_PYTHON_BIN" -m torch.distributed.run)
else
  echo "Unable to resolve python on \$(hostname)" >&2
  exit 1
fi
if [[ "$NCCL_SOCKET_IFNAME" == "auto" || "$GLOO_SOCKET_IFNAME" == "auto" ]]; then
  LOCAL_SOCKET_IFNAME=""
  if command -v ip >/dev/null 2>&1; then
    LOCAL_SOCKET_IFNAME="\$(ip route get "$MASTER_ADDR" 2>/dev/null | awk '{for (i=1; i<=NF; i++) if (\$i=="dev") {print \$(i+1); exit}}')"
  fi
  if [[ -z "\$LOCAL_SOCKET_IFNAME" ]]; then
    echo "Unable to auto-resolve socket interface. Set NCCL_SOCKET_IFNAME and GLOO_SOCKET_IFNAME explicitly." >&2
    exit 1
  fi
else
  LOCAL_SOCKET_IFNAME=""
fi
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
export NCCL_TIMEOUT=3600
if [[ "$NCCL_SOCKET_IFNAME" == "auto" ]]; then
  export NCCL_SOCKET_IFNAME="\$LOCAL_SOCKET_IFNAME"
else
  export NCCL_SOCKET_IFNAME="$NCCL_SOCKET_IFNAME"
fi
if [[ "$GLOO_SOCKET_IFNAME" == "auto" ]]; then
  export GLOO_SOCKET_IFNAME="\$LOCAL_SOCKET_IFNAME"
else
  export GLOO_SOCKET_IFNAME="$GLOO_SOCKET_IFNAME"
fi
export NCCL_DEBUG="$NCCL_DEBUG"
export NCCL_IB_DISABLE="$NCCL_IB_DISABLE"
export NCCL_CROSS_NIC="$NCCL_CROSS_NIC"
export NCCL_NET="$NCCL_NET"
export NCCL_NET_PLUGIN="$NCCL_NET_PLUGIN"
unset NCCL_IB_HCA || true
unset NCCL_IBEXT_DISABLE || true
echo "Resolved local socket interface: \${LOCAL_SOCKET_IFNAME:-<explicit>}"
echo "Effective NCCL_SOCKET_IFNAME=\${NCCL_SOCKET_IFNAME}"
echo "Effective GLOO_SOCKET_IFNAME=\${GLOO_SOCKET_IFNAME}"
echo "Effective NCCL_NET=\${NCCL_NET}"
echo "Effective NCCL_NET_PLUGIN=\${NCCL_NET_PLUGIN}"
echo "Resolved python executable: \${REMOTE_PYTHON_BIN:-<none>}"
echo "Resolved torchrun executable: \${REMOTE_TORCHRUN_BIN:-<none>}"
TRAIN_ARGS=(
  "$TRAIN_SCRIPT"
  --config_path "$CONFIG_PATH"
)
if [[ -n "$RESUME_PATH" ]]; then
  if [[ ! -f "$RESUME_PATH" ]]; then
    echo "Resume checkpoint not found: $RESUME_PATH" >&2
    exit 1
  fi
  TRAIN_ARGS+=(--resume "$RESUME_PATH")
fi
"\${LAUNCHER_CMD[@]}" \\
  --nnodes="$NNODES" \\
  --nproc_per_node="$NUM_GPUS" \\
  --node_rank="$node_rank" \\
  --master_addr="$MASTER_ADDR" \\
  --master_port="$MASTER_PORT" \\
  "\${TRAIN_ARGS[@]}"
EOF
}

launch_remote_job() {
  local user="$1"
  local host="$2"
  local port="$3"
  local password="$4"
  local node_rank="$5"
  local remote_script
  remote_script="$(build_remote_command "$node_rank")"
  if [[ -n "$password" ]]; then
    if command -v sshpass >/dev/null 2>&1; then
      printf '%s\n' "$remote_script" | sshpass -p "$password" ssh $SSH_OPTS \
        -o PreferredAuthentications=password \
        -o PubkeyAuthentication=no \
        -o NumberOfPasswordPrompts=1 \
        -p "$port" "${user}@${host}" "bash -s"
    else
      echo "sshpass not found; falling back to plain ssh for ${user}@${host}:${port}" >&2
      printf '%s\n' "$remote_script" | ssh $SSH_OPTS -p "$port" "${user}@${host}" "bash -s"
    fi
  else
    printf '%s\n' "$remote_script" | ssh $SSH_OPTS -p "$port" "${user}@${host}" "bash -s"
  fi
}

stop_remote_job() {
  local user="$1"
  local host="$2"
  local port="$3"
  local password="$4"
  local remote_cmd
  remote_cmd="pkill -f '$TRAIN_SCRIPT' || true; pkill -f 'torch.distributed.run.*--master_port=$MASTER_PORT' || true; pkill -f 'torchrun.*--master_port=$MASTER_PORT' || true"
  run_ssh "$user" "$host" "$port" "$password" "$remote_cmd" >/dev/null 2>&1 || true
}

terminate_remote_jobs() {
  stop_remote_job "$MASTER_USER" "$MASTER_HOSTNAME" "$MASTER_SSH_PORT" "$MASTER_PASSWORD"
  for i in "${!WORKER_HOSTNAMES[@]}"; do
    stop_remote_job "${WORKER_USERS[$i]}" "${WORKER_HOSTNAMES[$i]}" "${WORKER_SSH_PORTS[$i]}" "${WORKER_PASSWORDS[$i]}"
  done
}

terminate_local_jobs() {
  jobs -pr | xargs -r kill >/dev/null 2>&1 || true
}

CLEANUP_DONE=0
cleanup() {
  local exit_code=${1:-$?}
  if [[ "$CLEANUP_DONE" -eq 1 ]]; then
    return
  fi
  CLEANUP_DONE=1
  if [[ $exit_code -ne 0 ]]; then
    echo "Cleaning up local and remote training processes..." >&2
    terminate_local_jobs
    terminate_remote_jobs
    echo "Multi-node launcher failed with exit code $exit_code" >&2
  fi
  wait || true
}

handle_interrupt() {
  cleanup 130
  exit 130
}

trap 'cleanup $?' EXIT
trap handle_interrupt INT TERM

MASTER_LOG="$LOG_DIR/master_rank0.log"
echo "Launching master on ${MASTER_USER}@${MASTER_HOSTNAME}:${MASTER_SSH_PORT} ..."
echo "Master log: $MASTER_LOG"
launch_remote_job "$MASTER_USER" "$MASTER_HOSTNAME" "$MASTER_SSH_PORT" "$MASTER_PASSWORD" 0 \
  2>&1 | tee "$MASTER_LOG" &

for i in "${!WORKER_HOSTNAMES[@]}"; do
  node_rank=$((i + 1))
  host="${WORKER_HOSTNAMES[$i]}"
  port="${WORKER_SSH_PORTS[$i]}"
  user="${WORKER_USERS[$i]}"
  password="${WORKER_PASSWORDS[$i]}"
  worker_log="$LOG_DIR/worker_rank${node_rank}.log"
  echo "Launching worker rank $node_rank on ${user}@${host}:${port} ..."
  echo "Worker rank $node_rank log: $worker_log"
  launch_remote_job "$user" "$host" "$port" "$password" "$node_rank" \
    2>&1 | tee "$worker_log" &
done

wait
