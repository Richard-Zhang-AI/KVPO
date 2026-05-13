"""KVPO (KVPO) training entry point (multi-GPU via torchrun).

Usage:
    # single GPU
    python train_kvpo.py --config_path configs/train_kvpo.yaml

    # multi-GPU (preferred — handled by train_kvpo.sh)
    torchrun --nproc_per_node=2 train_kvpo.py --config_path configs/train_kvpo.yaml
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
LONG_LIVE_ROOT = Path(__file__).resolve().parent
if str(LONG_LIVE_ROOT) not in sys.path:
    sys.path.insert(0, str(LONG_LIVE_ROOT))

from models.longlive.trainer.kvpo import KVPOTrainer
from models.longlive.utils.misc import set_seed
from models.longlive.utils.prompt_sequences import load_prompt_sequences


def _resolve_repo_path(path_value: str | None) -> str | None:
    if path_value is None:
        return None
    text = str(path_value).strip()
    if not text:
        return text
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    else:
        path = path.resolve()
    return str(path)


def _resolve_config_paths(config):
    for key in ("data_path", "generator_ckpt", "lora_ckpt", "output_dir"):
        value = getattr(config, key, None)
        if value:
            setattr(config, key, _resolve_repo_path(value))
    return config


def setup_distributed():
    """Initialize distributed training if launched via torchrun."""
    if "RANK" not in os.environ:
        return 0, 0, 1

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", device_id=torch.device(f"cuda:{local_rank}"))
    return rank, local_rank, world_size


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def _make_unique_run_dir(output_root: str, requested_run_name: str | None) -> str:
    run_name = str(requested_run_name or "").strip()
    if not run_name:
        now = time.time()
        stamp = time.strftime("run_%Y%m%d_%H%M%S", time.gmtime(now))
        millis = int((now - int(now)) * 1000)
        run_name = f"{stamp}_{millis:03d}"

    run_dir = os.path.join(output_root, run_name)
    if not os.path.exists(run_dir):
        return run_dir

    suffix = 1
    while True:
        candidate = os.path.join(output_root, f"{run_name}_{suffix:02d}")
        if not os.path.exists(candidate):
            return candidate
        suffix += 1


def _broadcast_optional_string(value: str | None, src: int = 0) -> str | None:
    if not dist.is_initialized():
        return value
    device = torch.device(f"cuda:{torch.cuda.current_device()}") if torch.cuda.is_available() else torch.device("cpu")
    if dist.get_rank() == src:
        payload = b"" if value is None else value.encode("utf-8")
        size = len(payload) if value is not None else -1
    else:
        payload = b""
        size = 0

    size_tensor = torch.tensor([size], dtype=torch.int64, device=device)
    dist.broadcast(size_tensor, src=src)
    size = int(size_tensor.item())
    if size < 0:
        return None

    if dist.get_rank() == src:
        buf = torch.tensor(list(payload), dtype=torch.uint8, device=device)
    else:
        buf = torch.empty(size, dtype=torch.uint8, device=device)
    dist.broadcast(buf, src=src)
    return bytes(buf.cpu().tolist()).decode("utf-8")


def resolve_output_dir(config, resume_path: str | None, is_main: bool) -> str:
    output_root = os.path.abspath(str(config.output_dir))
    group_outputs_by_run = bool(getattr(config, "group_outputs_by_run", True))
    resume_output_dir_from_checkpoint = bool(
        getattr(config, "resume_output_dir_from_checkpoint", True)
    )
    run_dir = None
    if dist.is_initialized():
        if is_main:
            if resume_path and resume_output_dir_from_checkpoint:
                run_dir = os.path.abspath(os.path.dirname(resume_path))
            elif group_outputs_by_run:
                run_dir = _make_unique_run_dir(
                    output_root,
                    getattr(config, "run_name", ""),
                )
            else:
                run_dir = output_root
        run_dir = _broadcast_optional_string(run_dir, src=0)
    else:
        if resume_path and resume_output_dir_from_checkpoint:
            run_dir = os.path.abspath(os.path.dirname(resume_path))
        elif group_outputs_by_run:
            run_dir = _make_unique_run_dir(
                output_root,
                getattr(config, "run_name", ""),
            )
        else:
            run_dir = output_root
    return run_dir


def crossed_interval(prev_value: int, current_value: int, interval: int) -> bool:
    if interval <= 0:
        return False
    return (prev_value // interval) < (current_value // interval)


def checkpoint_base_name(samples_seen: int, final: bool = False) -> str:
    stem = "checkpoint_final_samples" if final else "checkpoint_samples"
    return f"{stem}_{samples_seen:09d}"


def main():
    parser = argparse.ArgumentParser("KVPO training")
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    args = parser.parse_args()

    config_path = _resolve_repo_path(args.config_path)
    resume_path = _resolve_repo_path(args.resume) if args.resume else None

    config = OmegaConf.load(config_path)
    default_config = OmegaConf.load(str((REPO_ROOT / "configs" / "default_config.yaml").resolve()))
    config = OmegaConf.merge(default_config, config)
    config = _resolve_config_paths(config)

    rank, local_rank, world_size = setup_distributed()
    is_main = (rank == 0)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    config.output_dir = resolve_output_dir(config, resume_path, is_main)

    set_seed(int(config.seed) + rank)

    if is_main:
        os.makedirs(config.output_dir, exist_ok=True)
        resolved_config_path = os.path.join(config.output_dir, "config_resolved.yaml")
        with open(resolved_config_path, "w", encoding="utf-8") as f:
            f.write(OmegaConf.to_yaml(config))
    if dist.is_initialized():
        dist.barrier()

    trainer = KVPOTrainer(config, device, rank=rank, world_size=world_size)
    reward_summary = ", ".join(
        f"{comp['key']}={comp['name']}*{comp['weight']:.3f}"
        for comp in trainer.reward_components
    )

    if resume_path:
        trainer.load_checkpoint(resume_path)
        if is_main:
            print(f"[Main] Resumed from {resume_path}, starting at step {trainer.step}")

    all_data = load_prompt_sequences(
        config.data_path,
        min_segments=getattr(config, "prompt_min_segments", None),
        max_segments=getattr(config, "prompt_max_segments", None),
    )
    n_eval_raw = getattr(config, "eval_prompts", 0)
    if isinstance(n_eval_raw, str):
        n_eval_key = n_eval_raw.strip().lower()
        if n_eval_key in {"world_size", "num_gpus", "auto"}:
            n_eval = world_size
        else:
            raise ValueError(
                "eval_prompts must be an integer or one of "
                "{'world_size', 'num_gpus', 'auto'} "
                f"(got {n_eval_raw!r})"
            )
    else:
        n_eval = int(n_eval_raw)
    if n_eval > 0 and n_eval < len(all_data):
        data_items = all_data[:-n_eval]
        eval_items = all_data[-n_eval:]
    else:
        data_items = all_data
        eval_items = []
    if is_main:
        print(f"[Main] eval_prompts={n_eval_raw!r} -> resolved to {n_eval}")
        print(f"[Main] Loaded {len(all_data)} prompt sequences "
              f"(train={len(data_items)}, eval={len(eval_items)})")

    max_steps = int(config.max_train_steps)
    save_interval_steps = int(getattr(config, "save_interval_steps", 0))
    save_interval_samples = int(getattr(config, "save_interval_samples", 0))

    if is_main:
        print(f"\n{'='*60}")
        print(f"KVPO Training  (world_size={world_size})")
        print(f"  output_dir   = {config.output_dir}")
        print(f"  max_steps   = {max_steps}")
        print(f"  G (branches)= {trainer.G}")
        print(f"  kl_coef     = {trainer.kl_coef}")
        print(f"  kl_mode     = {trainer.kl_loss_mode}")
        print(f"  kl_ref_init = {trainer.kl_reference_initial}")
        print(
            f"  clip_range  = low {trainer.clip_range_low} / high {trainer.clip_range_high} "
            f"(ratio ∈ [{trainer.ratio_clip_low}, {trainer.ratio_clip_high}]; "
            f"symmetric default key clip_range={trainer.clip_range})"
        )
        print(f"  lr          = {config.lr}")
        print(f"  max_grad    = {trainer.max_grad_norm}")
        print(f"  rewards     = {reward_summary}")
        if save_interval_steps > 0:
            print(f"  save_every  = {save_interval_steps} optimizer steps")
        else:
            print(f"  save_every  = {save_interval_samples} samples")
        print(f"{'='*60}\n")

    log_path = os.path.join(config.output_dir, "train_log.jsonl")
    eval_interval_steps = int(getattr(config, "eval_interval_steps", 0))
    eval_interval_samples = int(getattr(config, "eval_interval_samples", 0))
    eval_parallel_online_ema = bool(getattr(config, "eval_parallel_online_ema", True))
    accum_steps = int(getattr(config, "gradient_accumulation_steps", 1))
    total_rollouts = max_steps * accum_steps
    rollout_count = int(getattr(trainer, "rollout_count", 0)) or (trainer.step * accum_steps + trainer._accum_count)
    samples_seen = int(getattr(trainer, "samples_seen", 0)) or (rollout_count * world_size)
    trainer.rollout_count = rollout_count
    trainer.samples_seen = samples_seen
    prev_trainer_step = trainer.step
    prev_samples_seen = samples_seen

    def _run_eval(tag: str):
        if not eval_items:
            return
        if trainer.ema is not None and eval_parallel_online_ema:
            if is_main:
                print(f"\n{'='*40} EVAL ({tag}, online+ema parallel) {'='*40}")
            eval_results = trainer.evaluate_both(eval_items)
            if eval_results and is_main:
                with open(log_path, "a") as f:
                    for eval_result in eval_results:
                        eval_result["tag"] = tag
                        f.write(json.dumps(eval_result) + "\n")
                        print(f"{'='*40} END EVAL ({eval_result['model_type']}) {'='*40}")
                print()
            if dist.is_initialized():
                dist.barrier()
            return

        def _run_one(model_type: str, use_ema: bool = False):
            if is_main:
                print(f"\n{'='*40} EVAL ({tag}, {model_type}) {'='*40}")
            eval_result = trainer.evaluate(eval_items, use_ema=use_ema)
            if eval_result and is_main:
                eval_result["tag"] = tag
                with open(log_path, "a") as f:
                    f.write(json.dumps(eval_result) + "\n")
                print(f"{'='*40} END EVAL ({model_type}) {'='*40}\n")
            if dist.is_initialized():
                dist.barrier()

        _run_one("online", use_ema=False)
        if trainer.ema is not None:
            _run_one("ema", use_ema=True)

    _run_eval("before_training")

    while trainer.step < max_steps:
        t0 = time.time()
        rollout_count += 1

        sample_idx = ((rollout_count - 1) * world_size + rank) % len(data_items)
        prompts_list = data_items[sample_idx]["prompts"]

        if is_main:
            accum_idx = trainer._accum_count + 1
            print(f"\n--- Rollout {rollout_count} "
                  f"(effective_step {trainer.step + 1}/{max_steps}, "
                  f"accum {accum_idx}/{accum_steps}, "
                  f"rank0 sample {sample_idx}) ---")

        rollout_result = trainer.rollout(prompts_list, sample_idx=sample_idx)
        t_rollout = time.time() - t0

        t1 = time.time()
        metrics = trainer.train_step(rollout_result)
        t_update = time.time() - t1

        samples_seen = rollout_count * world_size
        trainer.rollout_count = rollout_count
        trainer.samples_seen = samples_seen

        metrics["time_rollout"] = t_rollout
        metrics["time_update"] = t_update
        metrics["time_total"] = time.time() - t0
        metrics["rollout_count"] = rollout_count
        metrics["samples_seen"] = samples_seen

        stepped = (trainer.step > prev_trainer_step)

        if is_main and (stepped or rollout_count == 1):
            cur_lr = trainer.optimizer.param_groups[0]["lr"]
            parts = []
            for tag in ["local", "global"]:
                rk = f"rewards_per_video_{tag}"
                ak = f"anchor_reward_{tag}"
                if rk in metrics:
                    rs = metrics[rk]
                    ar = metrics.get(ak, float('nan'))
                    r_str = ", ".join(f"{r:.3f}" for r in rs)
                    parts.append(f"  {tag}: rewards=[{r_str}] anchor={ar:.3f}")
            for line in parts:
                print(line)
            for tag in ["local", "global"]:
                ck = f"reward_components_{tag}"
                if ck in metrics:
                    for name, payload in metrics[ck].items():
                        r_str = ", ".join(f"{r:.3f}" for r in payload["rewards_per_video"])
                        a_str = ", ".join(f"{a:.3f}" for a in payload["advantages"])
                        print(
                            f"  {tag}:{name}: rewards=[{r_str}] "
                            f"anchor={payload['anchor_reward']:.3f} adv=[{a_str}]"
                        )
            print(f"  lr={cur_lr:.2e} "
                  f"final_loss={metrics['loss_total']:.4f} "
                  f"time={metrics['time_total']:.1f}s "
                  f"(rollout={t_rollout:.1f}s update={t_update:.1f}s)")

        if is_main:
            def _json_safe(obj):
                if isinstance(obj, torch.Tensor):
                    return obj.detach().cpu().tolist()
                if isinstance(obj, dict):
                    return {k: _json_safe(v) for k, v in obj.items()}
                if isinstance(obj, (list, tuple)):
                    return [_json_safe(x) for x in obj]
                return obj

            with open(log_path, "a") as f:
                f.write(json.dumps(_json_safe(metrics)) + "\n")

        should_save = False
        if stepped:
            if save_interval_steps > 0:
                should_save = crossed_interval(prev_trainer_step, trainer.step, save_interval_steps)
            else:
                should_save = crossed_interval(prev_samples_seen, samples_seen, save_interval_samples)

        if should_save and is_main:
            ckpt_stem = checkpoint_base_name(samples_seen)
            ckpt_path = os.path.join(config.output_dir, f"{ckpt_stem}.pt")
            extra_state = {
                "samples_seen": samples_seen,
                "rollout_count": rollout_count,
                "world_size": world_size,
            }
            trainer.save_checkpoint(ckpt_path, extra_state=extra_state)
            if trainer.ema is not None:
                ema_path = os.path.join(config.output_dir, f"{ckpt_stem}_ema.pt")
                trainer.save_ema_checkpoint(ema_path, extra_state=extra_state)

        should_eval = False
        if stepped:
            if eval_interval_steps > 0:
                should_eval = crossed_interval(prev_trainer_step, trainer.step, eval_interval_steps)
            else:
                should_eval = crossed_interval(prev_samples_seen, samples_seen, eval_interval_samples)

        if should_eval:
            _run_eval(f"step_{trainer.step}")

        prev_trainer_step = trainer.step
        prev_samples_seen = samples_seen
        del rollout_result
        torch.cuda.empty_cache()

        if dist.is_initialized():
            dist.barrier()

    if is_main:
        final_stem = checkpoint_base_name(samples_seen, final=True)
        final_path = os.path.join(config.output_dir, f"{final_stem}.pt")
        final_state = {
            "samples_seen": samples_seen,
            "rollout_count": rollout_count,
            "world_size": world_size,
        }
        trainer.save_checkpoint(final_path, extra_state=final_state)
        if trainer.ema is not None:
            trainer.save_ema_checkpoint(
                os.path.join(config.output_dir, f"{final_stem}_ema.pt"),
                extra_state=final_state,
            )
        print("\n[Main] Training complete!")

    cleanup_distributed()


if __name__ == "__main__":
    main()
