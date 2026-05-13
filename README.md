<div align="center">

<h1>KVPO: ODE-Native GRPO for Autoregressive Video Alignment via KV Semantic Exploration</h1>

Ruicheng Zhang<sup>1,3</sup>, Kaixi Cong<sup>1</sup>, Jun Zhou<sup>1</sup>, Zhizhou Zhong<sup>2,3</sup>, Zunnan Xu<sup>1</sup>, Shuiyang Mao<sup>3†</sup>, Wei Liu<sup>3</sup>, Xiu Li<sup>1‡</sup>

<sup>1</sup>Tsinghua University, <sup>2</sup>HKUST, <sup>3</sup>Video Rebirth

† Project leader. ‡ Corresponding author.

</div>

This repository contains KVPO training entrypoints for two backbones:

- `memflow`: MemFlow-style memory bank and sparse KV activation.
- `longlive`: LongLive-style frame sink and streaming local context.

## Environment

The released environment is captured in `environment_kvpo.yml`.

```bash
conda env create -f environment_kvpo.yml
conda activate KVPO
```

If your CUDA driver or cluster image differs from the exported environment,
create the environment first, then reinstall the PyTorch / CUDA / flash-attn
stack that matches your machine.

## Checkpoints

Install the Hugging Face CLI and download the released KVPO checkpoints:

```bash
pip install "huggingface_hub[cli]"
huggingface-cli download Richard-ZZZZZ/KVPO --local-dir checkpoints
```

The default configs expect the following files after download:

```text
checkpoints/
  memflow/
    base.pt
    lora.pt
  longlive/
    models/
      longlive_base.pt
      lora.pt
```

Download the Wan base models used by the streaming generators:

```bash
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir wan_models/Wan2.1-T2V-1.3B
huggingface-cli download Wan-AI/Wan2.1-T2V-14B --local-dir wan_models/Wan2.1-T2V-14B
```

Some reward functions also require their own checkpoints. The default training
configs use `video_hpsv3` and/or `videoalign_*`; keep the reward checkpoints in
the paths provided by the KVPO checkpoint release, or update
`reward_components` in the training config to use scorers available locally.

## Prompts

Training prompts are hosted separately at
https://huggingface.co/datasets/Richard-ZZZZZ/KVPO-prompt/tree/main.
Download the prompt files before training and place them under the repository
relative `prompts/` directory.

You can download them manually from the dataset page, or use the Hugging Face
CLI:

```bash
huggingface-cli download Richard-ZZZZZ/KVPO-prompt \
  --repo-type dataset \
  --local-dir prompts
```

After download, make sure each training config's `data_path` points to the
prompt file you want to use.

## Configure Training

Before launching a run, edit the matching YAML file:

- `configs/train_kvpo_memflow.yaml`
- `configs/train_kvpo_longlive.yaml`

The most common fields to change are:

- `num_gpus` and `gpu_ids`: devices used by the launcher.
- `data_path`: prompt file for online rollouts.
- `generator_ckpt` and `lora_ckpt`: initialization checkpoints.
- `K`: number of KV exploration branches per prompt.
- `reward_components`: preference reward mix and weights.
- `output_dir`: where logs and checkpoints are written.

By default, each launcher reads `num_gpus` and `gpu_ids` from its YAML config and
then starts `torchrun`.

## Single-Node Training

MemFlow backend:

```bash
bash train_kvpo_memflow.sh
```

LongLive backend:

```bash
bash train_kvpo_longlive.sh
```

To override the script or resume manually, call the Python entrypoint directly:

```bash
torchrun --nproc_per_node=8 train_kvpo_memflow.py \
  --config_path configs/train_kvpo_memflow.yaml \
  --resume logs/memflow/<run_name>/checkpoint_samples_XXXXXXX.pt
```

Replace the script and config with the backend you are training:

```text
train_kvpo_memflow.py   configs/train_kvpo_memflow.yaml
train_kvpo_longlive.py  configs/train_kvpo_longlive.yaml
```

## Multi-Node Training

Each backend also has a multi-node launcher:

```bash
bash train_kvpo_memflow_multinode.sh
bash train_kvpo_longlive_multinode.sh
```

The launcher reads the `multinode` section in the config. Replace the example
hostnames, ports, and users with your cluster information:

```yaml
multinode:
  nodes:
    node0:
      ssh_host: master.example.com
      ssh_port: 22
      ssh_user: user
    node1:
      ssh_host: worker1.example.com
      ssh_port: 22
      ssh_user: user
  master: node0
  workers:
    - node1
```

You can also override the topology without editing YAML:

```bash
MASTER_NODE_ALIAS=node0 WORKER_NODE_ALIASES="node1 node2" \
  bash train_kvpo_longlive_multinode.sh
```

or provide direct SSH fields:

```bash
MASTER_HOSTNAME=master.example.com MASTER_SSH_PORT=22 MASTER_USER=user \
WORKER_HOSTNAMES="worker1.example.com worker2.example.com" \
WORKER_SSH_PORTS="22 22" WORKER_USERS="user" \
  bash train_kvpo_memflow_multinode.sh
```

Password mode is disabled by default. Prefer SSH keys; if your cluster requires
password authentication, set `SSH_PASSWORD`, `MASTER_PASSWORD`, or
`WORKER_PASSWORDS` explicitly.

## Outputs

With `group_outputs_by_run: true`, every run writes to a timestamped directory
under `output_dir`, for example:

```text
logs/memflow/run_YYYYMMDD_HHMMSS_mmm/
  config_resolved.yaml
  train_log.jsonl
  checkpoint_samples_*.pt
  checkpoint_samples_*_ema.pt
```

Use `*_ema.pt` checkpoints for EMA evaluation or inference when
`ema_decay > 0`.

## Inference

Single-prompt generation:

```bash
bash inference.sh
```

Interactive long-video generation:

```bash
bash interactive_inference.sh
```

Update `configs/inference.yaml` or `configs/interactive_inference.yaml` to point
to the checkpoint you want to evaluate.

## Acknowledgements

This codebase builds on Wan2.1, MemFlow, LongLive, HPS, and VideoAlign
components. Please follow the licenses of the upstream projects and
downloaded checkpoints.
