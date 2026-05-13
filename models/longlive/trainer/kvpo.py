"""KVPO (KV-Perturbation Policy Optimisation) trainer for autoregressive long-video generation.

Rollout uses the frozen old policy to generate G branches; training uses **per-step**
trajectory log-probabilities (replay) and a PPO/GRPO-style clipped ratio on the
induced branch distribution π(g).
"""

import copy
import fcntl
import json
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from omegaconf import OmegaConf
from tqdm.auto import tqdm
from torchvision.io import write_video

from models.longlive.pipeline.diversity_sampling import DiversitySamplingPipeline
from models.longlive.utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper
from models.longlive.utils.memory import DynamicSwapInstaller


@dataclass
class ClipSpec:
    """Specification for a video clip to be scored."""
    name: str
    start_frame: int
    end_frame: int
    prompt: str
    group: str  # "normal" or "transition"


class _ParameterEMA:
    """Lightweight EMA over trainable parameters, stored on CPU."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for n, p in model.named_parameters():
                if p.requires_grad:
                    self.shadow[n] = p.detach().float().cpu().clone()

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        d = self.decay
        for n, p in model.named_parameters():
            if n in self.shadow:
                self.shadow[n].mul_(d).add_(p.detach().float().cpu(), alpha=1.0 - d)

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, sd):
        self.shadow = {k: v.clone() for k, v in sd.items()}

    @torch.no_grad()
    def copy_to(self, model: torch.nn.Module):
        for n, p in model.named_parameters():
            if n in self.shadow:
                p.data.copy_(self.shadow[n].to(dtype=p.dtype, device=p.device))


def _broadcast_optional_string(value: str | None, src: int = 0, device: Optional[torch.device] = None) -> str | None:
    if not dist.is_initialized():
        return value
    if device is None:
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


class KVPOTrainer:
    """Trainer for KVPO (KV-Perturbation Policy Optimisation)."""

    @staticmethod
    def _normalize_kl_loss_mode(raw_mode: Any) -> str:
        mode = str(raw_mode if raw_mode is not None else "discrete_kl").strip().lower()
        if mode in {"discrete_kl", "discrete-kl", "distribution", "pi_kl", "branch_discrete_kl"}:
            return "discrete_kl"
        if mode in {"try_kl", "try-kl", "anchor_l2", "anchor_mse", "ref_anchor_l2"}:
            return "try_kl"
        raise ValueError(
            "longlive KVPO only supports kl_loss_mode in {'discrete_kl', 'try_kl'} "
            f"(got {raw_mode!r}; anchor_window_block_l2 has been removed)."
        )

    def __init__(self, config, device, rank: int = 0, world_size: int = 1):
        self.config = config
        self.device = device
        self.dtype = torch.bfloat16
        self.step = 0
        self.samples_seen = 0
        self.rollout_count = 0
        self.rank = rank
        self.world_size = world_size
        self.is_main = (rank == 0)
        self.gradient_checkpointing = bool(getattr(config, "gradient_checkpointing", False))
        self.offload_models = bool(getattr(config, "offload_models", False))
        self.kl_reference_initial = bool(getattr(config, "kl_reference_initial", False))
        self.kl_loss_mode = self._normalize_kl_loss_mode(
            getattr(config, "kl_loss_mode", "discrete_kl")
        )
        self.needs_initial_generator = self.kl_reference_initial
        self.use_raw_weighted_reward_advantage = bool(
            getattr(config, "use_raw_weighted_reward_advantage", False)
        )

        self._build_models()
        self._build_optimizer()
        self._build_pipeline()
        self._build_reward_fn()

        self.min_update_entropy_ratio = float(getattr(config, "min_update_entropy_ratio", 0.0))
        self.policy_log_prob_mode = self._normalize_policy_log_prob_mode(
            getattr(config, "policy_log_prob_mode", "per_step_x")
        )
        _pspm = str(getattr(config, "per_step_protect_mode", "mechanism_1")).strip().lower()
        if _pspm in ("mechanism_1", "m1", "reward_gate"):
            self.per_step_protect_mode = "mechanism_1"
        elif _pspm in ("mechanism_2", "m2", "anchor_entropy"):
            self.per_step_protect_mode = "mechanism_2"
        else:
            raise ValueError(
                "per_step_protect_mode must be 'mechanism_1' or 'mechanism_2' "
                f"(got {getattr(config, 'per_step_protect_mode', None)!r})"
            )
        self.log_prob_grad_steps = int(getattr(config, "log_prob_grad_steps", 2))
        # Asymmetric PPO ratio clip: ratio ∈ [1 - clip_range_low, 1 + clip_range_high].
        # If only ``clip_range`` is set, it applies to both sides (symmetric).
        _sym = float(getattr(config, "clip_range", 0.2))
        self.clip_range_low = float(getattr(config, "clip_range_low", _sym))
        self.clip_range_high = float(getattr(config, "clip_range_high", _sym))
        self.clip_range = _sym
        self.ratio_clip_low = 1.0 - self.clip_range_low
        self.ratio_clip_high = 1.0 + self.clip_range_high
        self.adv_clip_max = float(getattr(config, "adv_clip_max", 5.0))
        self.kl_coef = float(getattr(config, "kl_coef", getattr(config, "beta", 0.0)))
        self.G = int(config.K)
        self.branch_backward_chunk_size = max(
            0, int(getattr(config, "branch_backward_chunk_size", 0))
        )
        self.max_grad_norm = float(config.max_grad_norm)
        self.perturb_num_blocks = int(getattr(config, "perturb_num_blocks", 3))
        self.anchor_grad_steps = int(getattr(config, "anchor_grad_steps", 4))
        self.ppo_epochs = int(getattr(config, "ppo_epochs", 3))
        self.train_local = bool(getattr(config, "train_local", True))
        self.train_global = bool(getattr(config, "train_global", True))
        self.local_loss_weight = float(getattr(config, "local_loss_weight", 1.0))
        self.global_loss_weight = float(getattr(config, "global_loss_weight", 1.0))
        assert self.train_local or self.train_global, "At least one of train_local/train_global must be true"
        self.policy_window_frames = int(getattr(config, "policy_window_frames", 0))
        self.policy_window_multiple = float(getattr(config, "policy_window_multiple", 2.0))
        self.advantage_fusion_mode = str(
            getattr(config, "advantage_fusion_mode", "std")
        ).strip().lower()
        self.advantage_fusion_local_weight = float(
            getattr(config, "advantage_fusion_local_weight", self.local_loss_weight)
        )
        self.advantage_fusion_global_weight = float(
            getattr(config, "advantage_fusion_global_weight", self.global_loss_weight)
        )
        self.gradient_accumulation_steps = int(getattr(config, "gradient_accumulation_steps", 1))
        self._accum_count = 0
        ema_decay = float(getattr(config, "ema_decay", 0))
        self.ema = _ParameterEMA(self.generator, ema_decay) if ema_decay > 0 else None
        self.show_progress = bool(getattr(config, "show_progress", True)) and self.is_main
        self.align_switch_to_chunk = bool(getattr(config, "align_switch_to_chunk", True))

        self.num_training_frames = int(getattr(config, "num_training_frames", 21))
        self.vae_temporal_stride = int(getattr(config, "vae_temporal_stride", 4))
        default_global_clip_len = self._latent_to_pixel_frame(self.num_training_frames)
        self.long_reward_enable_normal = bool(getattr(config, "long_reward_enable_normal", True))
        self.long_reward_enable_transition = bool(getattr(config, "long_reward_enable_transition", True))
        self.long_reward_normal_weight = float(getattr(config, "long_reward_normal_weight", 1.0))
        self.long_reward_transition_weight = float(getattr(config, "long_reward_transition_weight", 0.5))
        self.long_reward_normal_clip_len = int(
            getattr(config, "long_reward_normal_clip_len", default_global_clip_len)
        )
        self.long_reward_transition_clip_len = int(getattr(config, "long_reward_transition_clip_len", 48))

        self.debug_save_first_video_group = bool(
            getattr(config, "debug_save_first_video_group", False)
        )
        self.debug_video_group_fps = int(getattr(config, "debug_video_group_fps", 16))
        self.debug_video_group_subdir = str(
            getattr(config, "debug_video_group_subdir", "debug_first_video_group")
        )
        self._debug_video_group_saved = False
        self._last_rollout_progress_signature = None
        self._init_progress_run_dir()

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _broadcast_module_state(self, module: torch.nn.Module, label: str) -> None:
        """Broadcast parameters and buffers from rank 0 to all ranks.

        KVPO does manual gradient averaging instead of wrapping the model in
        DDP/FSDP, so model state must be synchronized explicitly after any
        per-rank random initialization such as freshly created LoRA weights.
        """
        if not dist.is_initialized() or self.world_size <= 1:
            return

        for tensor in module.parameters():
            dist.broadcast(tensor.data, src=0)
        for tensor in module.buffers():
            dist.broadcast(tensor.data, src=0)

        if self.is_main:
            print(f"[KVPO] Broadcast {label} state from rank 0")

    def _distributed_bool_any(self, value: bool) -> bool:
        """True if any rank reports ``value`` True (MAX over 0/1 flags)."""
        if not dist.is_initialized() or self.world_size <= 1:
            return bool(value)
        flag = torch.tensor(1 if value else 0, device=self.device, dtype=torch.int32)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        return bool(flag.item())

    def _distributed_bool_all(self, value: bool) -> bool:
        """True iff every rank reports ``value`` True (MIN over 0/1 flags)."""
        if not dist.is_initialized() or self.world_size <= 1:
            return bool(value)
        flag = torch.tensor(1 if value else 0, device=self.device, dtype=torch.int32)
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
        return bool(flag.item())

    def _average_trainable_grads(self) -> None:
        """All-reduce mean gradient for every trainable parameter on all ranks.

        PROTECT ranks may skip backward into the generator, leaving ``p.grad`` as
        ``None``. Those ranks must still enter the same collectives with zeros so
        (1) NCCL does not deadlock and (2) every rank applies the same mean grad.
        """
        if not dist.is_initialized() or self.world_size <= 1:
            return
        ws = float(self.world_size)
        for p in self.generator.parameters():
            if not p.requires_grad:
                continue
            if p.grad is None:
                p.grad = torch.zeros_like(p.data)
            else:
                p.grad = p.grad.contiguous()
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
            p.grad.div_(ws)

    def _build_models(self):
        cfg = self.config
        model_kwargs = OmegaConf.to_container(cfg.model_kwargs, resolve=True)

        self.generator = WanDiffusionWrapper(**model_kwargs, is_causal=True)

        if cfg.generator_ckpt:
            ckpt = torch.load(cfg.generator_ckpt, map_location="cpu", weights_only=False)
            raw = ckpt.get("generator", ckpt.get("generator_ema", ckpt.get("model")))
            self.generator.load_state_dict(raw)
            if self.is_main:
                print(f"[KVPO] Loaded generator from {cfg.generator_ckpt}")

        adapter_cfg = getattr(cfg, "adapter", None)
        if adapter_cfg:
            from models.longlive.utils.lora_utils import configure_lora_for_model
            import peft
            self.generator.model = configure_lora_for_model(
                self.generator.model, model_name="generator",
                lora_config=adapter_cfg, is_main_process=True)
            lora_ckpt = getattr(cfg, "lora_ckpt", None)
            if lora_ckpt:
                lc = torch.load(lora_ckpt, map_location="cpu", weights_only=False)
                sd = lc["generator_lora"] if isinstance(lc, dict) and "generator_lora" in lc else lc
                peft.set_peft_model_state_dict(self.generator.model, sd)
                if self.is_main:
                    print(f"[KVPO] Loaded LoRA from {lora_ckpt}")

        self.generator.to(dtype=self.dtype, device=self.device)
        if adapter_cfg:
            self.generator.model.train()
        else:
            self.generator.model.requires_grad_(True)
        self.generator.train()
        self._broadcast_module_state(self.generator, "generator")

        self.old_generator = copy.deepcopy(self.generator)
        self.old_generator.requires_grad_(False)
        self.old_generator.eval()
        if self.is_main:
            print("[KVPO] Old-policy model created (frozen)")

        self.initial_generator: Optional[WanDiffusionWrapper] = None
        if self.needs_initial_generator:
            self.initial_generator = copy.deepcopy(self.generator)
            self.initial_generator.requires_grad_(False)
            self.initial_generator.eval()
            if self.is_main:
                print("[KVPO] KL penalty will use frozen initial generator (kl_reference_initial=True)")

        self.text_encoder = WanTextEncoder()
        self.text_encoder.requires_grad_(False)
        DynamicSwapInstaller.install_model(self.text_encoder, device=self.device)

        self.vae = WanVAEWrapper()
        self.vae.requires_grad_(False)
        self.vae.to(dtype=self.dtype, device=self.device)

        self.scheduler = self.generator.get_scheduler()

    def _build_optimizer(self):
        train_params = [p for p in self.generator.parameters() if p.requires_grad]
        assert len(train_params) > 0, "No trainable parameters in generator"
        old_params = list(self.old_generator.parameters())
        for rp in old_params:
            assert not rp.requires_grad, "Ref model param has requires_grad=True"

        self.optimizer = torch.optim.AdamW(
            train_params,
            lr=float(self.config.lr),
            weight_decay=float(self.config.weight_decay),
        )

        from diffusers.optimization import get_scheduler
        sched_type = str(getattr(self.config, "lr_scheduler", "constant_with_warmup"))
        warmup = int(getattr(self.config, "lr_warmup_steps", 10))
        total_steps = int(getattr(self.config, "max_train_steps", 3000))
        self.lr_scheduler = get_scheduler(
            sched_type,
            optimizer=self.optimizer,
            num_warmup_steps=warmup,
            num_training_steps=total_steps,
        )

        if self.is_main:
            print(f"[KVPO] Optimizer: AdamW, lr={self.config.lr}, "
                  f"scheduler={sched_type}, warmup={warmup}, "
                  f"trainable params={sum(p.numel() for p in train_params):,}")

    @torch.no_grad()
    def _sync_old_policy(self):
        """Sync the frozen reference to the previous-update policy.

        This is a quasi-PPO old-policy: after each optimiser step, we copy
        the current generator weights into ``old_generator``. The next
        training step then regularises against the immediately previous
        policy instead of a fixed initial reference.
        """
        self.old_generator.load_state_dict(self.generator.state_dict(), strict=True)
        self.old_generator.to(dtype=self.dtype, device=self.device)
        self.old_generator.requires_grad_(False)
        self.old_generator.eval()

    def _build_pipeline(self):
        cfg = self.config
        model_kwargs = OmegaConf.to_container(cfg.model_kwargs, resolve=True)

        denoising_step_list = torch.tensor(cfg.denoising_step_list, dtype=torch.long)
        if getattr(cfg, "warp_denoising_step", False):
            ts = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
            denoising_step_list = ts[1000 - denoising_step_list]
        self.denoising_step_list = denoising_step_list

        self.pipeline = DiversitySamplingPipeline(
            denoising_step_list=denoising_step_list,
            scheduler=self.scheduler,
            generator=self.generator,
            num_frame_per_block=getattr(cfg, "num_frame_per_block", 3),
            same_step_across_blocks=getattr(cfg, "same_step_across_blocks", True),
            last_step_only=getattr(cfg, "last_step_only", True),
            context_noise=getattr(cfg, "context_noise", 0),
            local_attn_size=model_kwargs.get("local_attn_size", 12),
            slice_last_frames=getattr(cfg, "slice_last_frames", 21),
            m_nearest_frames=getattr(cfg, "m_nearest_frames", 2),
            recache_full_kv_cache_after_switch=bool(
                getattr(cfg, "recache_full_kv_cache_after_switch", False)
            ),
        )

    def _build_reward_fn(self):
        from rewards.rewards import build_named_reward_fn

        def _build_component_list(raw_components, *, field_name: str, allow_fallback: bool):
            if raw_components is None and allow_fallback:
                raw_components = [getattr(self.config, "reward_fn", "random")]
            elif OmegaConf.is_config(raw_components):
                raw_components = OmegaConf.to_container(raw_components, resolve=True)

            if not isinstance(raw_components, list) or len(raw_components) == 0:
                raise ValueError(f"{field_name} must be a non-empty list when provided")

            reward_components = []
            seen_keys = set()
            for idx, spec in enumerate(raw_components):
                if isinstance(spec, str):
                    name = spec
                    weight = 1.0
                    key = None
                    kwargs = {}
                elif isinstance(spec, dict):
                    name = spec.get("name") or spec.get("reward_fn")
                    if not name:
                        raise ValueError(f"{field_name}[{idx}] must define 'name'")
                    weight = float(spec.get("weight", 1.0))
                    key = spec.get("alias")
                    kwargs = dict(spec.get("kwargs") or {})
                else:
                    raise TypeError(
                        f"{field_name}[{idx}] must be a string or dict, got {type(spec).__name__}"
                    )

                canonical_name, scorer = build_named_reward_fn(name, device=self.device, **kwargs)
                key = str(key or canonical_name)
                if key in seen_keys:
                    raise ValueError(f"Duplicate reward component key in {field_name}: {key}")
                seen_keys.add(key)
                reward_components.append(
                    {
                        "key": key,
                        "name": canonical_name,
                        "weight": float(weight),
                        "kwargs": kwargs,
                        "fn": scorer,
                    }
                )
            return reward_components

        reward_components = _build_component_list(
            getattr(self.config, "reward_components", None),
            field_name="reward_components",
            allow_fallback=True,
        )
        raw_eval_components = getattr(self.config, "eval_reward_components", None)
        eval_reward_components = (
            reward_components
            if raw_eval_components is None
            else _build_component_list(
                raw_eval_components,
                field_name="eval_reward_components",
                allow_fallback=False,
            )
        )

        self.reward_components = reward_components
        self.eval_reward_components = eval_reward_components
        self.reward_fn = reward_components[0]["fn"] if len(reward_components) == 1 else None

        if self.is_main:
            if len(reward_components) == 1:
                comp = reward_components[0]
                print(f"[KVPO] Reward: {comp['name']} (weight={comp['weight']:.3f})")
            else:
                formatted = ", ".join(
                    f"{comp['key']}={comp['name']}*{comp['weight']:.3f}"
                    for comp in reward_components
                )
                print(f"[KVPO] Reward components: {formatted}")
                print(
                    "[KVPO] Multi-reward fusion: "
                    f"use_raw_weighted_reward_advantage={self.use_raw_weighted_reward_advantage} "
                    "(False=sum of weighted per-component advantages; True=advantage from weighted raw aggregate)"
                )
            if eval_reward_components is reward_components:
                print("[KVPO] Eval reward components: reuse train reward components")
            else:
                eval_formatted = ", ".join(
                    f"{comp['key']}={comp['name']}*{comp['weight']:.3f}"
                    for comp in eval_reward_components
                )
                print(f"[KVPO] Eval reward components: {eval_formatted}")

        if self.offload_models:
            self.old_generator.to("cpu")
            if self.initial_generator is not None:
                self.initial_generator.to("cpu")
            self.vae.to("cpu")
            self._reward_models_to_cpu()
            torch.cuda.empty_cache()
            if self.is_main:
                print("[KVPO] offload_models=True: old_generator, reward models (if supported), and VAE moved to CPU")

    def _iter_all_reward_components(self):
        seen = set()
        for attr in ("reward_components", "eval_reward_components"):
            for comp in getattr(self, attr, []) or []:
                comp_id = id(comp.get("fn"))
                if comp_id in seen:
                    continue
                seen.add(comp_id)
                yield comp

    @staticmethod
    def _normalize_policy_log_prob_mode(raw_mode) -> str:
        mode = str(raw_mode).strip().lower().replace("-", "_")
        aliases = {
            "perstep": "per_step_x",
            "per_step": "per_step_x",
            "per_step_x": "per_step_x",
            "perstep_x": "per_step_x",
        }
        if mode not in aliases:
            raise ValueError(
                "longlive KVPO supports policy_log_prob_mode='per_step' "
                f"(got {raw_mode!r}; 'per_step_x' is accepted as a legacy alias for "
                "the current x0-replay implementation; 'l2_distance' global "
                "latent-L2 policy modeling has been removed)."
            )
        return aliases[mode]

    @staticmethod
    def _anchor_advantage(branch_r: torch.Tensor, anchor_r: torch.Tensor) -> torch.Tensor:
        all_r = torch.cat([branch_r, anchor_r.unsqueeze(0)])
        sigma = all_r.std() + 1e-8
        return (branch_r - anchor_r) / sigma

    @staticmethod
    def _per_step_branch_log_pi(log_scores: torch.Tensor) -> torch.Tensor:
        """log π(g) over G branches from raw trajectory scores (logits), dim=0."""
        return torch.log_softmax(log_scores, dim=0)

    def _per_step_mechanism2_anchor_scale_entropy(
        self,
        log_pi_theta: torch.Tensor,
        rank_is_safe: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Augment branch logits with an extra anchor slot in the softmax pool.

        ``log_pi_theta`` should be G-dimensional branch logits; in per-step training
        pass ``log_softmax(trajectory_scores)`` so branch masses match π(g) before
        concatenating the anchor logit.

        When ``rank_is_safe``: anchor logit is -inf (no mass on anchor). When unsafe:
        anchor logit is 0 so the anchor slot competes with branches; loss is scaled by
        ``1 - pi_anchor``. Entropy is ``-(π_branch log π_branch).sum()`` over G branches.
        """
        anchor_logit = torch.tensor(
            float("-inf") if rank_is_safe else 0.0,
            device=log_pi_theta.device,
            dtype=log_pi_theta.dtype,
        )
        logits = torch.cat([log_pi_theta, anchor_logit.unsqueeze(0)], dim=0)
        pi_full = F.softmax(logits, dim=0)
        pi_anchor = pi_full[-1]
        loss_scale = 1.0 - pi_anchor
        pi_branch = pi_full[:-1]
        log_pi_branch = torch.log(pi_branch + 1e-10)
        entropy = -(pi_branch * log_pi_branch).sum()
        return loss_scale, entropy, pi_anchor

    def _aggregate_component_advantages(
        self,
        component_rewards: Dict[str, torch.Tensor],
        component_anchor_rewards: Dict[str, torch.Tensor],
    ):
        total_adv = None
        component_advs = {}

        for comp in self.reward_components:
            key = comp["key"]
            weight = float(comp["weight"])
            comp_adv = self._anchor_advantage(
                component_rewards[key],
                component_anchor_rewards[key],
            )
            component_advs[key] = comp_adv
            weighted_adv = comp_adv * weight
            total_adv = weighted_adv if total_adv is None else total_adv + weighted_adv

        return total_adv, component_advs

    def _align_frames_to_block(self, frame_count: int) -> int:
        num_fpb = self.pipeline.num_frame_per_block
        if frame_count <= 0:
            return 0
        return ((int(frame_count) + num_fpb - 1) // num_fpb) * num_fpb

    def _resolve_policy_window_frames(self, available_postfork_frames: int) -> int:
        perturbed_frames = self.perturb_num_blocks * self.pipeline.num_frame_per_block
        requested = int(self.policy_window_frames)
        if requested <= 0:
            requested = int(round(perturbed_frames * self.policy_window_multiple))
        requested = max(requested, perturbed_frames)
        requested = self._align_frames_to_block(requested)
        available = self._align_frames_to_block(int(available_postfork_frames))
        resolved = min(requested, available)
        if resolved <= 0:
            raise ValueError(
                f"Resolved policy window must be positive, got {resolved} "
                f"(requested={requested}, available={available})"
            )
        return resolved

    def _normalize_advantage_for_fusion(self, adv: torch.Tensor) -> torch.Tensor:
        mode = self.advantage_fusion_mode
        if mode == "none":
            return adv
        std = adv.std(unbiased=False)
        if mode == "std":
            if std <= 1e-8:
                return adv
            return adv / (std + 1e-8)
        if mode != "zscore":
            raise ValueError(
                f"Unsupported advantage_fusion_mode={self.advantage_fusion_mode!r}; "
                "expected 'none', 'std', or 'zscore'"
            )
        mean = adv.mean()
        if std <= 1e-8:
            return adv - mean
        return (adv - mean) / (std + 1e-8)

    def _fuse_signal_advantages(
        self,
        adv_local: Optional[torch.Tensor],
        adv_global: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        fused = None
        details: Dict[str, Any] = {"mode": self.advantage_fusion_mode}

        if adv_local is not None:
            local_norm = self._normalize_advantage_for_fusion(adv_local.float())
            details["local"] = {
                "weight": self.advantage_fusion_local_weight,
                "raw": adv_local.detach().cpu().tolist(),
                "normalized": local_norm.detach().cpu().tolist(),
            }
            fused = local_norm * self.advantage_fusion_local_weight

        if adv_global is not None:
            global_norm = self._normalize_advantage_for_fusion(adv_global.float())
            details["global"] = {
                "weight": self.advantage_fusion_global_weight,
                "raw": adv_global.detach().cpu().tolist(),
                "normalized": global_norm.detach().cpu().tolist(),
            }
            weighted_global = global_norm * self.advantage_fusion_global_weight
            fused = weighted_global if fused is None else fused + weighted_global

        if fused is None:
            raise ValueError("At least one of adv_local/adv_global must be available for fusion")

        details["fused"] = fused.detach().cpu().tolist()
        return fused, details

    # ------------------------------------------------------------------
    # Dynamic model offloading helpers
    # ------------------------------------------------------------------

    def _old_gen_to_gpu(self):
        """Move frozen old policy to this rank's GPU for rollout / log_pi_old."""
        self.old_generator.to(device=self.device, dtype=self.dtype)

    def _old_gen_to_cpu(self):
        """Move old policy off GPU so only the trainable generator stays during backward."""
        self.old_generator.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _initial_gen_to_gpu(self):
        if self.initial_generator is None:
            return
        self.initial_generator.to(device=self.device, dtype=self.dtype)

    def _initial_gen_to_cpu(self):
        if self.initial_generator is None:
            return
        self.initial_generator.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _vae_to_gpu(self):
        if self.offload_models:
            self.vae.to(device=self.device, dtype=self.dtype)

    def _vae_to_cpu(self):
        if self.offload_models:
            self.vae.to("cpu")
            torch.cuda.empty_cache()

    def _move_reward_models(self, target_device: str) -> None:
        if not self.offload_models:
            return
        for comp in self._iter_all_reward_components():
            move_fn = getattr(comp.get("fn"), "_kvpo_move_to_device", None)
            if callable(move_fn):
                move_fn(target_device)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _move_reward_component(self, comp: Dict[str, Any], target_device: str) -> None:
        if not self.offload_models:
            return
        move_fn = getattr(comp.get("fn"), "_kvpo_move_to_device", None)
        if callable(move_fn):
            move_fn(target_device)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _reward_component_to_gpu(self, comp: Dict[str, Any]) -> None:
        self._move_reward_component(comp, str(self.device))

    def _reward_component_to_cpu(self, comp: Dict[str, Any]) -> None:
        self._move_reward_component(comp, "cpu")

    def _reward_models_to_gpu(self) -> None:
        self._move_reward_models(str(self.device))

    def _reward_models_to_cpu(self) -> None:
        self._move_reward_models("cpu")

    def _init_progress_run_dir(self) -> None:
        run_name = None
        if self.is_main:
            now = time.time()
            stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime(now))
            millis = int((now - int(now)) * 1000)
            run_name = f"run_{stamp}_{millis:03d}"

        if dist.is_initialized():
            run_name = _broadcast_optional_string(run_name, src=0, device=self.device)

        if run_name is None:
            now = time.time()
            stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime(now))
            millis = int((now - int(now)) * 1000)
            run_name = f"run_{stamp}_{millis:03d}"

        self.rollout_progress_run_dir = os.path.join(
            self.config.output_dir,
            ".rollout_progress",
            run_name,
        )
        if self.is_main:
            os.makedirs(self.rollout_progress_run_dir, exist_ok=True)
            print(f"[KVPO] Rollout progress dir: {self.rollout_progress_run_dir}")
        if dist.is_initialized():
            dist.barrier()

    def _make_rollout_tag(self) -> str:
        return f"step_{self.step + 1:06d}"

    def _rollout_progress_file(self, rollout_tag: str) -> str:
        return os.path.join(self.rollout_progress_run_dir, f"{rollout_tag}.json")

    def _write_rollout_progress(
        self,
        rollout_tag: str,
        *,
        completed: int,
        total: int,
        videos_done: int,
        finished: bool,
    ) -> None:
        progress_path = self._rollout_progress_file(rollout_tag)
        payload = {
            "rank": self.rank,
            "completed": int(completed),
            "total": int(total),
            "videos_done": int(videos_done),
            "finished": bool(finished),
            "timestamp": time.time(),
        }

        with open(progress_path, "a+", encoding="utf-8") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.seek(0)
                raw = f.read().strip()
                if raw:
                    data = json.loads(raw)
                else:
                    data = {"rollout_tag": rollout_tag, "ranks": {}}
                data["rollout_tag"] = rollout_tag
                data["updated_at"] = time.time()
                data.setdefault("ranks", {})[str(self.rank)] = payload
                f.seek(0)
                f.truncate()
                json.dump(data, f, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def _aggregate_rollout_progress(self, rollout_tag: str) -> Dict[str, int]:
        progress_path = self._rollout_progress_file(rollout_tag)
        totals = {
            "completed": 0,
            "total": 0,
            "videos_done": 0,
            "finished_ranks": 0,
            "seen_ranks": 0,
        }
        if not os.path.isfile(progress_path):
            return totals

        try:
            with open(progress_path, encoding="utf-8") as f:
                payload = json.load(f)
        except (json.JSONDecodeError, OSError):
            return totals

        for rank_payload in payload.get("ranks", {}).values():
            totals["seen_ranks"] += 1
            totals["completed"] += int(rank_payload.get("completed", 0))
            totals["total"] += int(rank_payload.get("total", 0))
            totals["videos_done"] += int(rank_payload.get("videos_done", 0))
            totals["finished_ranks"] += int(bool(rank_payload.get("finished", False)))

        return totals

    def _refresh_rollout_pbar(
        self,
        pbar: Optional[tqdm],
        rollout_tag: str,
        *,
        force: bool = False,
    ) -> None:
        if pbar is None or not self.is_main:
            return

        summary = self._aggregate_rollout_progress(rollout_tag)
        signature = (
            rollout_tag,
            summary["completed"],
            summary["total"],
            summary["videos_done"],
            summary["finished_ranks"],
        )
        if not force and signature == self._last_rollout_progress_signature:
            return
        self._last_rollout_progress_signature = signature

        global_total = max(summary["total"], 1)
        if pbar.total != global_total:
            pbar.total = global_total
        pbar.n = min(summary["completed"], global_total)
        pbar.set_postfix_str(
            f"videos {summary['videos_done']}/{self.G * self.world_size} | "
            f"ranks {summary['finished_ranks']}/{self.world_size}"
        )
        pbar.refresh()

    def _wait_for_rollout_completion(self, pbar: Optional[tqdm], rollout_tag: str) -> None:
        if pbar is None or not self.is_main or self.world_size <= 1:
            return

        while True:
            summary = self._aggregate_rollout_progress(rollout_tag)
            self._refresh_rollout_pbar(pbar, rollout_tag)
            if (
                summary["finished_ranks"] >= self.world_size
                and summary["completed"] >= summary["total"] > 0
            ):
                break
            time.sleep(0.5)

    @staticmethod
    def _parse_switch_frame_indices(raw_value: Any) -> List[int]:
        return [int(x) for x in str(raw_value).split(",") if str(x).strip()]

    def _resolve_switch_frame_indices(
        self,
        raw_value: Any,
        *,
        chunks: List[Dict[str, Any]],
        num_segments: int,
    ) -> List[int]:
        switch_indices = self._parse_switch_frame_indices(raw_value)
        expected = max(num_segments - 1, 0)
        assert len(switch_indices) >= expected, (
            f"Expected at least {expected} switch_frame_indices for {num_segments} prompt segments, "
            f"but got {len(switch_indices)}"
        )
        if len(switch_indices) > expected:
            original = switch_indices
            switch_indices = switch_indices[:expected]
            if self.is_main:
                print(
                    f"[KVPO] Truncate switch_frame_indices for {num_segments} prompt segments: "
                    f"{original} -> {switch_indices}"
                )

        if not self.align_switch_to_chunk or not switch_indices:
            return switch_indices

        chunk_starts = [int(chunk["start_frame"]) for chunk in chunks[1:]]
        aligned: List[int] = []
        for switch_idx in switch_indices:
            aligned_idx = next((start for start in chunk_starts if start >= switch_idx), chunk_starts[-1])
            aligned.append(aligned_idx)

        for prev, cur in zip(aligned, aligned[1:]):
            assert cur > prev, f"Aligned switch_frame_indices must be strictly increasing, got {aligned}"

        if aligned != switch_indices and self.is_main:
            print(
                f"[KVPO] Align switch_frame_indices to chunk starts: "
                f"{switch_indices} -> {aligned}"
            )
        return aligned

    def _maybe_save_debug_video_group(
        self,
        videos: List[torch.Tensor],
        prompts_list: List[str],
        switch_indices: List[int],
        rewards: List[float],
        sample_idx: int,
        reward_summary: Optional[Dict[str, Any]] = None,
        debug_source: str = "full_video",
        fork_abs_frame: Optional[int] = None,
    ) -> None:
        if (
            not self.debug_save_first_video_group
            or self._debug_video_group_saved
            or not self.is_main
        ):
            return

        self._debug_video_group_saved = True
        out_dir = os.path.join(self.config.output_dir, self.debug_video_group_subdir)
        os.makedirs(out_dir, exist_ok=True)

        prefix = f"step_{self.step + 1:06d}_sample_{sample_idx:06d}"
        for branch_idx, video in enumerate(videos):
            out_path = os.path.join(out_dir, f"{prefix}_branch_{branch_idx:02d}.mp4")
            vid_for_write = video.cpu()
            if vid_for_write.ndim == 4 and vid_for_write.shape[1] == 3:
                vid_for_write = vid_for_write.permute(0, 2, 3, 1)  # [T,C,H,W] -> [T,H,W,C]
            if vid_for_write.is_floating_point():
                vid_for_write = (vid_for_write * 255).clamp(0, 255).to(torch.uint8)
            write_video(out_path, vid_for_write, fps=self.debug_video_group_fps)

        meta_path = os.path.join(out_dir, f"{prefix}_metadata.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "step": self.step + 1,
                    "sample_idx": sample_idx,
                    "switch_frame_indices": switch_indices,
                    "debug_source": debug_source,
                    "fork_abs_frame": fork_abs_frame,
                    "fork_pixel_frame": (
                        self._latent_to_pixel_frame(fork_abs_frame)
                        if fork_abs_frame is not None else None
                    ),
                    "prompts": prompts_list,
                    "rewards": rewards,
                    "reward_summary": reward_summary,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"[KVPO] Saved debug first video group to {out_dir}")

    # ------------------------------------------------------------------
    # Rollout
    # ------------------------------------------------------------------

    def rollout(self, prompts_list: List[str], sample_idx: int = 0) -> Dict[str, Any]:
        """Run G-branch rollout for one prompt sequence.

        Returns a dict with all data needed by ``train_step``.
        """
        cfg = self.config
        device, dtype = self.device, self.dtype
        G = self.G
        chunk_size = int(cfg.streaming_chunk_size)
        max_length = int(cfg.streaming_max_length)
        min_new_frame = int(cfg.streaming_min_new_frame)
        base_seed = int(cfg.seed)
        rollout_step = self.step + 1
        seed_context = rollout_step * 1_000_003
        perturb_x = int(getattr(cfg, "perturb_within_first_x_chunks", 3))
        perturb_P = int(getattr(cfg, "perturb_num_blocks", 3))
        sink_size = int(cfg.model_kwargs.get("sink_size", 3))
        global_sink = bool(getattr(cfg, "global_sink", True))

        self.generator.eval()
        self.old_generator.eval()
        rollout_tag = self._make_rollout_tag()
        rollout_pbar = None
        pipeline_generator = self.pipeline.generator

        try:
            self._old_gen_to_gpu()
            self.pipeline.generator = self.old_generator
            self._last_rollout_progress_signature = None

            plan = self.pipeline.plan_perturbation(
                total_frames=max_length, chunk_size=chunk_size,
                min_new_frame=min_new_frame, K=G, base_seed=base_seed,
                sample_idx=sample_idx, seed_context=seed_context,
                perturb_within_first_x_chunks=perturb_x,
                perturb_num_blocks=perturb_P, sink_size=sink_size,
            )
            chunks = plan["chunks"]
            switch_raw = getattr(cfg, "switch_frame_indices", "57,93,129,165,201")
            switch_indices = self._resolve_switch_frame_indices(
                switch_raw,
                chunks=chunks,
                num_segments=len(prompts_list),
            )
            perturb_idx = plan["perturb_chunk_idx"]
            fork_abs = plan["perturb_block_abs_frame"]
            fork_chunk = chunks[perturb_idx]
            fork_local_block = plan["perturb_block_local_idx"]
            num_fpb = int(cfg.num_frame_per_block)
            fork_offset_in_chunk = fork_local_block * num_fpb
            fork_chunk_frames = int(fork_chunk["new_frames"])
            need_global = self.train_global
            need_local = self.train_local
            perturb_chunk_end_abs = int(fork_chunk["start_frame"]) + fork_chunk_frames
            use_per_step_log_prob = True
            policy_window_frames = self._resolve_policy_window_frames(max_length - fork_abs)
            policy_end_abs = min(max_length, fork_abs + policy_window_frames)
            generation_end_abs = max(
                policy_end_abs,
                perturb_chunk_end_abs if need_local else fork_abs,
                max_length if need_global else fork_abs,
            )

            with torch.no_grad():
                cond_list = []
                for p in prompts_list:
                    cd = self.text_encoder(text_prompts=[p])
                    cd = {
                        k: v.to(device=device, dtype=dtype) if isinstance(v, torch.Tensor) else v
                        for k, v in cd.items()
                    }
                    cond_list.append(cd)

            noise_seed = base_seed + sample_idx * 10000 + seed_context
            batch_size = 1
            self.pipeline._initialize_kv_cache(batch_size, dtype, device)
            self.pipeline._initialize_crossattn_cache(batch_size, dtype, device)
            self.pipeline.clear_kv_cache()
            self.pipeline.generator.model.local_attn_size = int(self.pipeline.local_attn_size)
            self.pipeline._set_all_modules_max_attention_size(int(self.pipeline.local_attn_size))

            rng = torch.Generator(device=device)
            rng.manual_seed(noise_seed)

            def _seg_for_frame(frame_idx: int) -> int:
                seg = 0
                for switch_idx in switch_indices:
                    if frame_idx >= switch_idx:
                        seg += 1
                    else:
                        break
                return seg

            rollout_ops = perturb_idx + (1 if fork_offset_in_chunk > 0 else 0) + G * (len(chunks) - perturb_idx)

            self._write_rollout_progress(
                rollout_tag,
                completed=0,
                total=rollout_ops,
                videos_done=0,
                finished=False,
            )
            rollout_pbar = tqdm(
                total=max(rollout_ops * max(self.world_size, 1), 1),
                desc=f"Rollout step {self.step + 1}",
                leave=False,
                disable=not self.show_progress,
            )
            self._refresh_rollout_pbar(rollout_pbar, rollout_tag)

            local_completed = 0
            local_videos_done = 0

            def _mark_rollout_progress(*, add_completed: int = 0, finished: bool = False) -> None:
                nonlocal local_completed, local_videos_done
                local_completed += add_completed
                self._write_rollout_progress(
                    rollout_tag,
                    completed=local_completed,
                    total=rollout_ops,
                    videos_done=local_videos_done,
                    finished=finished,
                )
                self._refresh_rollout_pbar(rollout_pbar, rollout_tag)

            prefix_latents = []
            current_length = 0
            seg_idx = 0
            with torch.no_grad():
                for ci, ch in enumerate(chunks):
                    if ci >= perturb_idx:
                        break
                    nf = ch["new_frames"]
                    if nf <= 0:
                        continue
                    new_seg = _seg_for_frame(current_length)
                    if new_seg != seg_idx:
                        self.pipeline.recache_after_switch(
                            prefix_latents,
                            current_length,
                            cond_list[new_seg],
                            global_sink,
                        )
                        seg_idx = new_seg
                    noise = torch.randn([1, nf, 16, 60, 104], generator=rng, device=device, dtype=dtype)
                    out = self.pipeline.generate_chunk_sampling(
                        noise=noise,
                        conditional_dict=cond_list[seg_idx],
                        current_start_frame=current_length,
                    )
                    prefix_latents.append(out.cpu())
                    current_length += nf
                    _mark_rollout_progress(add_completed=1)

            pre_rng = rng.get_state()
            self.pipeline.save_state()
            saved_cpu = torch.random.get_rng_state()
            saved_cuda = torch.cuda.get_rng_state(device)
            prefix_length = current_length

            shared_prefix_in_chunk = None

            targets = []
            completed_latents_all = []
            fork_noise_tensor = None
            kv_state_before_fork = None
            post_fork_rng = None
            saved_cpu_before_fork = saved_cpu
            saved_cuda_before_fork = saved_cuda

            shared_rng = torch.Generator(device=device)
            shared_rng.manual_seed(noise_seed)
            shared_rng.set_state(pre_rng)
            perturb_chunk_noise = torch.randn(
                [1, fork_chunk_frames, 16, 60, 104],
                generator=shared_rng,
                device=device,
                dtype=dtype,
            )
            post_fork_rng = shared_rng.get_state()
            perturb_P = int(getattr(cfg, "perturb_num_blocks", 3))
            fork_noise_tensor = perturb_chunk_noise[
                :, fork_offset_in_chunk:fork_offset_in_chunk + num_fpb * perturb_P
            ].clone()

            def _build_shared_nonperturb_noises() -> Dict[int, torch.Tensor]:
                """One RNG stream from post_fork: identical noise for every branch g on non-perturb chunks.

                KV perturbation only applies on ``perturb_idx``; all G groups must see the same
                latent noise afterward so differences come only from the perturbed chunk.
                """
                tail_rng = torch.Generator(device=device)
                tail_rng.manual_seed(noise_seed)
                tail_rng.set_state(post_fork_rng)
                out: Dict[int, torch.Tensor] = {}
                pnoise = perturb_chunk_noise[:, fork_offset_in_chunk:]
                if fork_abs + pnoise.shape[1] > generation_end_abs:
                    pnoise = pnoise[:, : generation_end_abs - fork_abs]
                cur_len = fork_abs + int(pnoise.shape[1])

                for ci in range(perturb_idx + 1, len(chunks)):
                    ch = chunks[ci]
                    nf = int(ch["new_frames"])
                    if nf <= 0:
                        continue
                    if not need_global and cur_len >= generation_end_abs:
                        break
                    if cur_len + nf > generation_end_abs:
                        nf = generation_end_abs - cur_len
                    if nf <= 0:
                        break
                    out[ci] = torch.randn(
                        [1, nf, 16, 60, 104],
                        generator=tail_rng,
                        device=device,
                        dtype=dtype,
                    )
                    cur_len += nf
                return out

            shared_nonperturb_noises = _build_shared_nonperturb_noises()

            if fork_offset_in_chunk > 0:
                self.pipeline.restore_state()
                pre_fork_noise = perturb_chunk_noise[:, :fork_offset_in_chunk]
                shared_prefix_in_chunk = self.pipeline.generate_chunk_sampling(
                    noise=pre_fork_noise,
                    conditional_dict=cond_list[seg_idx],
                    current_start_frame=prefix_length,
                ).detach().cpu()
                self.pipeline.save_state()
                kv_state_before_fork = copy.deepcopy(self.pipeline._saved_state)
                saved_cpu_before_fork = torch.random.get_rng_state()
                saved_cuda_before_fork = torch.cuda.get_rng_state(device)
                _mark_rollout_progress(add_completed=1)
            else:
                kv_state_before_fork = copy.deepcopy(self.pipeline._saved_state)

            pre_perturb_reference = None
            if fork_abs > 0:
                reference_parts = list(prefix_latents)
                if shared_prefix_in_chunk is not None:
                    reference_parts.append(shared_prefix_in_chunk)
                if reference_parts:
                    pre_perturb_reference = torch.cat(reference_parts, dim=1)
                    if pre_perturb_reference.shape[1] != fork_abs:
                        raise RuntimeError(
                            f"Pre-perturb reference length mismatch: expected {fork_abs}, "
                            f"got {pre_perturb_reference.shape[1]}"
                        )

            local_targets = []
            policy_targets = []
            all_branch_trajectories = []

            with torch.no_grad():
                for g in range(G):
                    branch_trajectories = []
                    if kv_state_before_fork is None:
                        raise RuntimeError("Missing pre-fork KV state")
                    self.pipeline._saved_state = kv_state_before_fork
                    self.pipeline.restore_state()
                    torch.random.set_rng_state(saved_cpu_before_fork)
                    torch.cuda.set_rng_state(saved_cuda_before_fork, device)
                    branch_latents = list(prefix_latents)
                    if shared_prefix_in_chunk is not None:
                        branch_latents.append(shared_prefix_in_chunk)
                    cur_len = fork_abs
                    cur_seg = _seg_for_frame(cur_len)
                    local_chunk_out = None

                    for ci in range(perturb_idx, len(chunks)):
                        if cur_len >= generation_end_abs:
                            break
                        ch = chunks[ci]
                        nf = ch["new_frames"]
                        if nf <= 0:
                            continue
                        new_seg = _seg_for_frame(cur_len)
                        if new_seg != cur_seg:
                            self.pipeline.recache_after_switch(
                                branch_latents,
                                cur_len,
                                cond_list[new_seg],
                                global_sink,
                            )
                            cur_seg = new_seg

                        is_perturb = ci == perturb_idx
                        if is_perturb:
                            noise = perturb_chunk_noise[:, fork_offset_in_chunk:]
                            if cur_len + noise.shape[1] > generation_end_abs:
                                keep_frames = generation_end_abs - cur_len
                                noise = noise[:, :keep_frames]
                            nf = noise.shape[1]
                            self.pipeline.activate_perturbation(
                                plan["branch_plans"][g],
                                target_abs_frame=plan["perturb_block_abs_frame"],
                                target_end_frame=plan["perturb_block_end_frame"],
                            )
                        else:
                            if not need_global and cur_len >= generation_end_abs:
                                break
                            if cur_len + nf > generation_end_abs:
                                nf = generation_end_abs - cur_len
                            if nf <= 0:
                                break
                            noise = shared_nonperturb_noises[ci]

                        if use_per_step_log_prob and is_perturb:
                            out, branch_trajectory = self.pipeline.generate_chunk_sampling_with_trajectory(
                                noise=noise,
                                conditional_dict=cond_list[cur_seg],
                                current_start_frame=cur_len,
                            )
                        else:
                            out = self.pipeline.generate_chunk_sampling(
                                noise=noise,
                                conditional_dict=cond_list[cur_seg],
                                current_start_frame=cur_len,
                            )
                            branch_trajectory = None

                        if is_perturb:
                            self.pipeline.deactivate_perturbation()
                            postfork_chunk = out.detach().cpu()
                            if shared_prefix_in_chunk is not None:
                                local_chunk_out = torch.cat(
                                    [shared_prefix_in_chunk, postfork_chunk], dim=1
                                )
                            else:
                                local_chunk_out = postfork_chunk
                        if branch_trajectory is not None:
                            branch_trajectories.append(branch_trajectory)

                        branch_latents.append(out.cpu())
                        cur_len += nf
                        _mark_rollout_progress(add_completed=1)

                    full_branch = torch.cat(branch_latents, dim=1)
                    if pre_perturb_reference is not None:
                        branch_prefix = full_branch[:, :fork_abs].cpu()
                        if not torch.equal(branch_prefix, pre_perturb_reference):
                            raise RuntimeError(
                                f"Branch {g} diverged before perturbation start at frame {fork_abs}"
                            )
                    completed_latents_all.append(full_branch)
                    policy_targets.append(
                        full_branch[:, fork_abs:policy_end_abs].detach().clone()
                    )
                    if need_global:
                        postfork = full_branch[:, fork_abs:].detach().clone()
                        targets.append(postfork)
                    if need_local and local_chunk_out is not None:
                        local_targets.append(local_chunk_out)
                    if use_per_step_log_prob and branch_trajectories:
                        all_branch_trajectories.append(branch_trajectories[0])
                    local_videos_done = g + 1
                    _mark_rollout_progress(add_completed=0)
                    print(
                        f"[Rank {self.rank}] rollout sample {sample_idx}: "
                        f"finished video group {local_videos_done}/{G}"
                    )

            _mark_rollout_progress(add_completed=0, finished=True)
            self._wait_for_rollout_completion(rollout_pbar, rollout_tag)

            targets_tensor = torch.cat(targets, dim=0).to(device=device, dtype=dtype) if targets else None
            local_targets_tensor = torch.cat(local_targets, dim=0).to(device=device, dtype=dtype) if local_targets else None
            policy_targets_tensor = torch.cat(policy_targets, dim=0).to(device=device, dtype=dtype)

            # Generate anchor (unperturbed) full video for reward baseline
            self.pipeline._saved_state = kv_state_before_fork
            self.pipeline.restore_state()
            torch.random.set_rng_state(saved_cpu_before_fork)
            torch.cuda.set_rng_state(saved_cuda_before_fork, device)
            anchor_rng = torch.Generator(device=device)
            anchor_rng.manual_seed(noise_seed)
            anchor_rng.set_state(post_fork_rng)

            anchor_latents = list(prefix_latents)
            if shared_prefix_in_chunk is not None:
                anchor_latents.append(shared_prefix_in_chunk)
            anchor_cur_len = fork_abs
            anchor_seg = _seg_for_frame(anchor_cur_len)
            anchor_local_chunk = None

            for ci in range(perturb_idx, len(chunks)):
                ch = chunks[ci]
                nf = ch["new_frames"]
                if nf <= 0:
                    continue
                new_seg = _seg_for_frame(anchor_cur_len)
                if new_seg != anchor_seg:
                    self.pipeline.recache_after_switch(
                        anchor_latents, anchor_cur_len,
                        cond_list[new_seg], global_sink)
                    anchor_seg = new_seg
                is_perturb = (ci == perturb_idx)
                if is_perturb:
                    noise = perturb_chunk_noise[:, fork_offset_in_chunk:]
                    nf = noise.shape[1]
                else:
                    if not need_global:
                        break
                    noise = torch.randn([1, nf, 16, 60, 104],
                                        generator=anchor_rng, device=device, dtype=dtype)
                out = self.pipeline.generate_chunk_sampling(
                    noise=noise, conditional_dict=cond_list[anchor_seg],
                    current_start_frame=anchor_cur_len)
                if is_perturb:
                    postfork_chunk = out.detach().cpu()
                    if shared_prefix_in_chunk is not None:
                        anchor_local_chunk = torch.cat(
                            [shared_prefix_in_chunk, postfork_chunk], dim=1
                        )
                    else:
                        anchor_local_chunk = postfork_chunk
                anchor_latents.append(out.cpu())
                anchor_cur_len += nf

            anchor_full_video = torch.cat(anchor_latents, dim=1)
            if pre_perturb_reference is not None:
                anchor_prefix = anchor_full_video[:, :fork_abs].cpu()
                if not torch.equal(anchor_prefix, pre_perturb_reference):
                    raise RuntimeError(
                        f"Anchor diverged before perturbation start at frame {fork_abs}"
                    )

            # Free GPU memory before reward scoring: neither generator is
            # needed until train_step, so offload both to CPU.
            self._old_gen_to_cpu()
            self.generator.to("cpu")
            torch.cuda.empty_cache()

            # --- Compute rewards ---
            global_rewards_raw = None
            global_anchor_reward = None
            global_rewards_components = None
            global_anchor_reward_components = None
            local_rewards_raw = None
            local_anchor_reward = None
            local_rewards_components = None
            local_anchor_reward_components = None

            if need_global:
                global_start_abs = perturb_chunk_end_abs
                full_video_latents = list(completed_latents_all) + [anchor_full_video]
                global_eval, global_debug_videos = self._compute_rewards_long(
                    full_video_latents, prompts_list, switch_indices,
                    global_start_abs, sample_idx,
                )
                global_scores = global_eval["aggregate_scores"]
                global_rewards_raw = torch.tensor(global_scores[:G], dtype=torch.float32, device=device)
                global_anchor_reward = torch.tensor(global_scores[G], dtype=torch.float32, device=device)
                global_rewards_components = {
                    name: torch.tensor(comp["scores"][:G], dtype=torch.float32, device=device)
                    for name, comp in global_eval["components"].items()
                }
                global_anchor_reward_components = {
                    name: torch.tensor(comp["scores"][G], dtype=torch.float32, device=device)
                    for name, comp in global_eval["components"].items()
                }

            if need_local:
                local_vids = [lt.cpu() for lt in local_targets]
                if anchor_local_chunk is not None:
                    local_vids.append(anchor_local_chunk)
                fork_seg = self._get_segment_for_frame(fork_abs, switch_indices)
                local_prompt = prompts_list[fork_seg]
                local_eval, local_debug_videos = self._compute_rewards_short(
                    local_vids, local_prompt, sample_idx,
                )
                local_scores = local_eval["aggregate_scores"]
                local_rewards_raw = torch.tensor(local_scores[:G], dtype=torch.float32, device=device)
                local_anchor_reward = torch.tensor(local_scores[G], dtype=torch.float32, device=device)
                local_rewards_components = {
                    name: torch.tensor(comp["scores"][:G], dtype=torch.float32, device=device)
                    for name, comp in local_eval["components"].items()
                }
                local_anchor_reward_components = {
                    name: torch.tensor(comp["scores"][G], dtype=torch.float32, device=device)
                    for name, comp in local_eval["components"].items()
                }

            debug_videos = None
            debug_source = "none"
            if self.debug_save_first_video_group and not self._debug_video_group_saved:
                debug_videos = self._decode_latents_to_pixels(
                    completed_latents_all + [anchor_full_video],
                    desc="VAE decode (debug full video group)",
                )
                debug_source = "full_video"
            else:
                debug_videos = global_debug_videos if need_global else (
                    local_debug_videos if need_local else None)
                debug_source = "postfork_reward" if need_global else (
                    "local_chunk_reward" if need_local else "none"
                )
            debug_rewards = (global_eval if need_global else local_eval).get(
                "aggregate_scores", []) if (need_global or need_local) else []
            if debug_videos is not None:
                self._maybe_save_debug_video_group(
                    debug_videos,
                    prompts_list,
                    switch_indices,
                    debug_rewards,
                    sample_idx,
                    reward_summary=(global_eval if need_global else local_eval),
                    debug_source=debug_source,
                    fork_abs_frame=fork_abs,
                )
        finally:
            self.pipeline.generator = pipeline_generator
            self._old_gen_to_cpu()
            self._reward_models_to_cpu()
            self.generator.to(device=self.device, dtype=self.dtype)
            torch.cuda.empty_cache()
            if rollout_pbar is not None:
                rollout_pbar.close()

        adv_global = adv_global_components = None
        if need_global:
            adv_global, adv_global_components = self._aggregate_component_advantages(
                global_rewards_components,
                global_anchor_reward_components,
            )

        adv_local = adv_local_components = None
        if need_local:
            adv_local, adv_local_components = self._aggregate_component_advantages(
                local_rewards_components,
                local_anchor_reward_components,
            )

        if self.use_raw_weighted_reward_advantage:
            if need_global:
                adv_global = self._anchor_advantage(global_rewards_raw, global_anchor_reward)
            if need_local:
                adv_local = self._anchor_advantage(local_rewards_raw, local_anchor_reward)

        cond_at_fork = cond_list[_seg_for_frame(fork_abs)]

        result = {
            "targets_global": targets_tensor.detach() if targets_tensor is not None else None,
            "targets_local": local_targets_tensor.detach() if local_targets_tensor is not None else None,
            "policy_targets": policy_targets_tensor.detach(),
            "policy_window_frames": policy_window_frames,
            "adv_global": adv_global.detach() if adv_global is not None else None,
            "adv_local": adv_local.detach() if adv_local is not None else None,
            "adv_global_components": {
                name: value.detach() for name, value in adv_global_components.items()
            } if adv_global_components is not None else None,
            "adv_local_components": {
                name: value.detach() for name, value in adv_local_components.items()
            } if adv_local_components is not None else None,
            "rewards_global": global_rewards_raw.detach() if global_rewards_raw is not None else None,
            "rewards_local": local_rewards_raw.detach() if local_rewards_raw is not None else None,
            "rewards_global_components": {
                name: value.detach() for name, value in global_rewards_components.items()
            } if global_rewards_components is not None else None,
            "rewards_local_components": {
                name: value.detach() for name, value in local_rewards_components.items()
            } if local_rewards_components is not None else None,
            "anchor_reward_global": global_anchor_reward.detach() if global_anchor_reward is not None else None,
            "anchor_reward_local": local_anchor_reward.detach() if local_anchor_reward is not None else None,
            "anchor_reward_global_components": {
                name: value.detach() for name, value in global_anchor_reward_components.items()
            } if global_anchor_reward_components is not None else None,
            "anchor_reward_local_components": {
                name: value.detach() for name, value in local_anchor_reward_components.items()
            } if local_anchor_reward_components is not None else None,
            "noise_block_b": fork_noise_tensor.detach().to(device),
            "kv_state_before_fork": kv_state_before_fork,
            "fork_abs_frame": fork_abs,
            "global_start_abs_frame": 0 if need_global else None,
            "conditional_dict": cond_at_fork,
            "plan": plan,
            "switch_frame_indices": switch_indices,
            "cond_list": cond_list,
            "chunks": chunks,
            "perturb_idx": perturb_idx,
            "prefix_latents": prefix_latents,
            "shared_prefix_in_chunk": shared_prefix_in_chunk.detach().cpu()
            if shared_prefix_in_chunk is not None else None,
            "perturb_chunk_noise": perturb_chunk_noise.detach().cpu(),
            "fork_offset_in_chunk": fork_offset_in_chunk,
            "post_fork_rng_state": post_fork_rng,
            "noise_seed": noise_seed,
            "seed_context": seed_context,
            "global_sink": global_sink,
            "saved_cpu_before_fork": saved_cpu_before_fork,
            "saved_cuda_before_fork": saved_cuda_before_fork,
            "branch_trajectories": all_branch_trajectories if use_per_step_log_prob else None,
        }

        self.generator.train()
        return result

    # ------------------------------------------------------------------
    # Reward computation: utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _get_segment_for_frame(
        frame_idx: int,
        switch_indices: List[int],
    ) -> int:
        """Return the prompt segment index for a given latent frame."""
        seg = 0
        for si in switch_indices:
            if frame_idx >= si:
                seg += 1
            else:
                break
        return seg

    def _latent_to_pixel_frame(self, latent_frame: int) -> int:
        """Convert a latent frame index to the corresponding pixel frame index.

        Valid for latent_frame >= 1.  Latent frame 0 maps to pixel frame 0
        (handled as a special case since the general formula yields a negative value).
        """
        if latent_frame <= 0:
            return 0
        return latent_frame * self.vae_temporal_stride - (self.vae_temporal_stride - 1)

    @staticmethod
    def _make_clip_window(
        center: int,
        total_frames: int,
        clip_len: int,
    ) -> Tuple[int, int]:
        """Compute a [start, end) clip window centred at *center*, clamped to [0, total)."""
        start = max(0, center - clip_len // 2)
        end = start + clip_len
        if end > total_frames:
            end = total_frames
            start = max(0, end - clip_len)
        return int(start), int(end)

    def _build_global_segment_clip_specs(
        self,
        full_pixel_len: int,
        prompts_list: List[str],
        switch_indices: List[int],
    ) -> Tuple[List[ClipSpec], List[ClipSpec]]:
        """Build fixed per-segment ClipSpecs for whole-video global scoring."""
        switch_count = max(0, len(prompts_list) - 1)
        switch_pixels: List[int] = []
        for si in switch_indices[:switch_count]:
            pixel_idx = self._latent_to_pixel_frame(int(si))
            if 0 < pixel_idx < full_pixel_len:
                switch_pixels.append(pixel_idx)

        boundaries = [0] + switch_pixels + [full_pixel_len]
        normal_specs: List[ClipSpec] = []
        transition_specs: List[ClipSpec] = []

        if self.long_reward_enable_normal:
            clip_len = max(1, self.long_reward_normal_clip_len)
            for i in range(min(len(prompts_list), len(boundaries) - 1)):
                seg_start, seg_end = boundaries[i], boundaries[i + 1]
                if seg_end - seg_start < 2:
                    continue
                center = (seg_start + seg_end) // 2
                clip_start, clip_end = self._make_clip_window(center, full_pixel_len, clip_len)
                normal_specs.append(ClipSpec(
                    name=f"normal_seg_{i}",
                    start_frame=clip_start,
                    end_frame=clip_end,
                    prompt=prompts_list[i],
                    group="normal",
                ))

        return normal_specs, transition_specs

    # ------------------------------------------------------------------
    # Reward computation: VAE decode helper
    # ------------------------------------------------------------------

    def _decode_latents_to_pixels(
        self,
        latent_list: List[torch.Tensor],
        desc: str = "VAE decode",
    ) -> List[torch.Tensor]:
        """Decode a list of latent tensors to pixel-space [T, C, H, W] float [0, 1]."""
        self._vae_to_gpu()
        videos: List[torch.Tensor] = []
        decode_iter = enumerate(latent_list)
        if self.is_main:
            decode_iter = tqdm(
                decode_iter, total=len(latent_list),
                desc=f"  [Rank {self.rank}] {desc}",
                leave=False,
            )
        for _gi, lat_cpu in decode_iter:
            lat = lat_cpu.to(device=self.device, dtype=self.dtype)
            vid = self.vae.decode_to_pixel(lat)
            vid_float = (vid * 0.5 + 0.5).clamp(0, 1)[0]  # [T, C, H, W]
            videos.append(vid_float.cpu())
            self.vae.model.clear_cache()
            del lat
            torch.cuda.empty_cache()
        self._vae_to_cpu()
        return videos

    # ------------------------------------------------------------------
    # Reward computation: short video (local signal)
    # ------------------------------------------------------------------

    def _compute_rewards_short(
        self,
        latent_list: List[torch.Tensor],
        prompt: str,
        sample_idx: int,
        include_aggregate: bool = True,
        reward_components: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[Dict[str, Any], List[torch.Tensor]]:
        """Score short videos (perturb chunk) directly as a whole clip.

        Args:
            latent_list: G+1 latent tensors, each [1, T, 16, 60, 104].
            prompt: the segment prompt at the fork point.
            sample_idx: data sample index (for debug).

        Returns:
            (reward_eval, videos) where reward_eval has per-component raw scores.
        """
        reward_components = reward_components or self.reward_components
        B = len(latent_list)
        videos = self._decode_latents_to_pixels(latent_list, desc="VAE decode (local)")

        pixel_frames = videos[0].shape[0]
        if self.is_main:
            print(f"[KVPO] Short reward (local): B={B}, "
                  f"pixel_frames={pixel_frames}, prompt='{prompt[:80]}...'")

        aggregate_scores = torch.zeros(B, dtype=torch.float32) if include_aggregate else None
        reward_eval: Dict[str, Any] = {"aggregate_scores": None, "components": {}}
        reward_batch_size = int(getattr(self.config, "reward_batch_size", 0)) or B

        for comp in reward_components:
            self._reward_component_to_gpu(comp)
            try:
                all_scores: List[float] = []
                for batch_start in range(0, B, reward_batch_size):
                    batch_end = min(batch_start + reward_batch_size, B)
                    batch_tensor = torch.stack(
                        videos[batch_start:batch_end], dim=0,
                    )  # [bs, T, C, H, W]
                    scores, _ = comp["fn"](batch_tensor, [prompt] * (batch_end - batch_start), None)
                    if isinstance(scores, torch.Tensor):
                        all_scores.extend(scores.cpu().tolist())
                    else:
                        all_scores.extend([float(s) for s in scores])
                    del batch_tensor
                score_tensor = torch.tensor(all_scores, dtype=torch.float32)
                if include_aggregate and aggregate_scores is not None:
                    aggregate_scores += score_tensor * float(comp["weight"])
                reward_eval["components"][comp["key"]] = {
                    "name": comp["name"],
                    "weight": float(comp["weight"]),
                    "scores": score_tensor.tolist(),
                }
                if self.is_main:
                    s_str = ", ".join(f"{s:.4f}" for s in score_tensor.tolist())
                    print(f"  [SHORT] {comp['key']} (w={comp['weight']:.2f}): "
                          f"scores=[{s_str}]")
                torch.cuda.empty_cache()
            finally:
                self._reward_component_to_cpu(comp)

        if include_aggregate and aggregate_scores is not None:
            reward_eval["aggregate_scores"] = aggregate_scores.tolist()
        if include_aggregate and aggregate_scores is not None and self.is_main:
            agg_str = ", ".join(f"{s:.4f}" for s in aggregate_scores.tolist())
            print(f"  [SHORT] aggregate: [{agg_str}]")
        return reward_eval, videos

    # ------------------------------------------------------------------
    # Reward computation: long video (global signal)
    # ------------------------------------------------------------------

    def _compute_rewards_long(
        self,
        latent_list: List[torch.Tensor],
        prompts_list: List[str],
        switch_indices: List[int],
        global_start_abs: int,
        sample_idx: int,
        include_aggregate: bool = True,
        reward_components: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[Dict[str, Any], List[torch.Tensor]]:
        """Score long videos via fixed per-segment clips over the full video.

        Args:
            latent_list: G+1 full-video latent tensors, each [1, T, 16, 60, 104].
            prompts_list: all segment prompts.
            switch_indices: latent-space switch frame indices (absolute).
            global_start_abs: kept for metadata compatibility; no longer affects scoring.
            sample_idx: data sample index (for debug).

        Returns:
            (reward_eval, videos) where reward_eval has per-component raw scores
            after weighted aggregation of per-segment clip scores.
        """
        reward_components = reward_components or self.reward_components
        B = len(latent_list)
        videos = self._decode_latents_to_pixels(latent_list, desc="VAE decode (global)")

        full_pixel_len = videos[0].shape[0]
        normal_specs, transition_specs = self._build_global_segment_clip_specs(
            full_pixel_len, prompts_list, switch_indices,
        )
        all_specs = normal_specs + transition_specs
        n_normal = len(normal_specs)
        n_transition = len(transition_specs)

        reward_batch_size = int(getattr(self.config, "reward_batch_size", 0)) or B

        if not all_specs:
            if self.is_main:
                print("[KVPO] WARNING: no clip specs generated for long reward, "
                      "falling back to whole-video scoring with first segment prompt")
            fallback_prompt = prompts_list[0] if prompts_list else ""
            aggregate_scores = torch.zeros(B, dtype=torch.float32) if include_aggregate else None
            reward_eval: Dict[str, Any] = {"aggregate_scores": None, "components": {}}
            for comp in reward_components:
                self._reward_component_to_gpu(comp)
                try:
                    all_scores: List[float] = []
                    for bs in range(0, B, reward_batch_size):
                        be = min(bs + reward_batch_size, B)
                        batch_t = torch.stack(videos[bs:be], dim=0)
                        sc, _ = comp["fn"](batch_t, [fallback_prompt] * (be - bs), None)
                        if isinstance(sc, torch.Tensor):
                            all_scores.extend(sc.cpu().tolist())
                        else:
                            all_scores.extend([float(s) for s in sc])
                        del batch_t
                    score_tensor = torch.tensor(all_scores, dtype=torch.float32)
                    if include_aggregate and aggregate_scores is not None:
                        aggregate_scores += score_tensor * float(comp["weight"])
                    reward_eval["components"][comp["key"]] = {
                        "name": comp["name"],
                        "weight": float(comp["weight"]),
                        "scores": score_tensor.tolist(),
                    }
                    torch.cuda.empty_cache()
                finally:
                    self._reward_component_to_cpu(comp)
            if include_aggregate and aggregate_scores is not None:
                reward_eval["aggregate_scores"] = aggregate_scores.tolist()
            return reward_eval, videos

        if self.is_main:
            print(f"[KVPO] Long reward (global): B={B}, "
                  f"{n_normal} normal clips, {n_transition} transition clips, "
                  f"full_pixel_len={full_pixel_len}")
            for i, spec in enumerate(all_specs):
                tag = "N" if spec.group == "normal" else "T"
                print(f"  [{tag}] {spec.name}: pixel[{spec.start_frame}:{spec.end_frame}] "
                      f"({spec.end_frame - spec.start_frame}f) "
                      f"prompt='{spec.prompt[:60]}...'")

        clip_scores: Dict[str, List[torch.Tensor]] = {
            comp["key"]: [] for comp in reward_components
        }

        reward_batch_size = int(getattr(self.config, "reward_batch_size", 0)) or B

        for comp in reward_components:
            self._reward_component_to_gpu(comp)
            try:
                for spec in all_specs:
                    all_video_scores: List[float] = []
                    for batch_start in range(0, B, reward_batch_size):
                        batch_end = min(batch_start + reward_batch_size, B)
                        clips = torch.stack([
                            videos[vi][spec.start_frame:spec.end_frame]
                            for vi in range(batch_start, batch_end)
                        ], dim=0)
                        prompts_batch = [spec.prompt] * (batch_end - batch_start)
                        scores, _ = comp["fn"](clips, prompts_batch, None)
                        if isinstance(scores, torch.Tensor):
                            all_video_scores.extend(scores.cpu().tolist())
                        else:
                            all_video_scores.extend(
                                [float(s) for s in scores])
                        del clips
                    score_t = torch.tensor(all_video_scores, dtype=torch.float32)
                    clip_scores[comp["key"]].append(score_t)
                    if self.is_main:
                        s_str = ", ".join(f"{s:.4f}" for s in score_t.tolist())
                        print(f"  [LONG] clip={spec.name} | {comp['key']}: [{s_str}]")
                torch.cuda.empty_cache()
            finally:
                self._reward_component_to_cpu(comp)

        w_n = self.long_reward_normal_weight if n_normal > 0 else 0.0
        w_t = self.long_reward_transition_weight if n_transition > 0 else 0.0
        total_w = w_n + w_t

        aggregate_scores = torch.zeros(B, dtype=torch.float32) if include_aggregate else None
        reward_eval = {"aggregate_scores": None, "components": {}}

        for comp in reward_components:
            all_clip = clip_scores[comp["key"]]

            normal_mean = torch.zeros(B, dtype=torch.float32)
            transition_mean = torch.zeros(B, dtype=torch.float32)
            if n_normal > 0:
                normal_mean = torch.stack(all_clip[:n_normal]).mean(dim=0)
            if n_transition > 0:
                transition_mean = torch.stack(all_clip[n_normal:]).mean(dim=0)

            if total_w > 0:
                comp_score = (w_n * normal_mean + w_t * transition_mean) / total_w
            else:
                comp_score = torch.zeros(B, dtype=torch.float32)

            if include_aggregate and aggregate_scores is not None:
                aggregate_scores += comp_score * float(comp["weight"])
            if self.is_main:
                n_str = ", ".join(f"{s:.4f}" for s in normal_mean.tolist()) if n_normal > 0 else "-"
                t_str = ", ".join(f"{s:.4f}" for s in transition_mean.tolist()) if n_transition > 0 else "-"
                f_str = ", ".join(f"{s:.4f}" for s in comp_score.tolist())
                print(f"  [LONG] {comp['key']} (w={comp['weight']:.2f}): "
                      f"normal_mean=[{n_str}] trans_mean=[{t_str}] → final=[{f_str}]")
            reward_eval["components"][comp["key"]] = {
                "name": comp["name"],
                "weight": float(comp["weight"]),
                "scores": comp_score.tolist(),
                "normal_clip_scores": [s.tolist() for s in all_clip[:n_normal]] if n_normal > 0 else [],
                "transition_clip_scores": [s.tolist() for s in all_clip[n_normal:]] if n_transition > 0 else [],
            }

        if include_aggregate and aggregate_scores is not None:
            reward_eval["aggregate_scores"] = aggregate_scores.tolist()
        reward_eval["n_normal_clips"] = n_normal
        reward_eval["n_transition_clips"] = n_transition
        if include_aggregate and aggregate_scores is not None and self.is_main:
            agg_str = ", ".join(f"{s:.4f}" for s in aggregate_scores.tolist())
            print(f"  [LONG] aggregate: [{agg_str}]")
        return reward_eval, videos

    def _materialize_cache_list(self, saved_list):
        materialized = []
        for blk in saved_list:
            item = {
                "k": blk["k"].to(device=self.device).clone(),
                "v": blk["v"].to(device=self.device).clone(),
                "global_end_index": blk["global_end_index"].to(device=self.device).clone(),
                "local_end_index": blk["local_end_index"].to(device=self.device).clone(),
            }
            if "k_new" in blk:
                item["k_new"] = blk["k_new"].to(device=self.device).clone()
            if "v_new" in blk:
                item["v_new"] = blk["v_new"].to(device=self.device).clone()
            materialized.append(item)
        return materialized

    def _materialize_crossattn_cache(self, saved_list):
        materialized = []
        for blk in saved_list:
            materialized.append({
                "k": blk["k"].to(device=self.device).clone(),
                "v": blk["v"].to(device=self.device).clone(),
                "is_init": blk["is_init"],
            })
        return materialized

    # ------------------------------------------------------------------
    # Single-block 4-step denoising (shared by anchor / ref_anchor)
    # ------------------------------------------------------------------

    def _denoise_blocks(
        self,
        generator,
        noise_blocks: torch.Tensor,
        conditional_dict: dict,
        kv_cache,
        crossattn_cache,
        start_token: int,
        num_blocks: int,
        grad_steps: int = 4,
        enable_grad: bool = True,
    ) -> torch.Tensor:
        """Denoise P consecutive blocks with M grad-enabled steps each.

        Args:
            noise_blocks: [B, P*num_frame_per_block, C, H, W] noise for all P blocks
            num_blocks: P — how many consecutive blocks to process
            grad_steps: M — the **first** M of the 4 denoising steps carry gradient
                (early steps establish composition/structure and are more important)
            enable_grad: master switch; if False, everything is no_grad

        Each block is 3 frames. After each block's denoising, a context_noise
        cache update is performed (no_grad) so subsequent blocks see proper
        KV context — mirroring ``generate_chunk_with_cache``.

        When ``self.gradient_checkpointing`` is True, grad-enabled forward
        passes are wrapped with ``torch.utils.checkpoint`` to discard
        intermediate activations (recomputed during backward).
        """
        num_fpb = self.pipeline.num_frame_per_block
        B = noise_blocks.shape[0]
        device = noise_blocks.device
        fsl = self.pipeline.frame_seq_length
        num_steps = len(self.denoising_step_list)
        use_ckpt = self.gradient_checkpointing and enable_grad

        def _generator_forward(noisy_in, cond, ts, kv, xattn, cur_start):
            """Thin wrapper whose positional args are checkpoint-friendly."""
            _, pred = generator(
                noisy_image_or_video=noisy_in,
                conditional_dict=cond,
                timestep=ts,
                kv_cache=kv,
                crossattn_cache=xattn,
                current_start=cur_start,
                update_cache=False,
            )
            return pred

        block_outputs = []
        for bi in range(num_blocks):
            frame_offset = bi * num_fpb
            block_noise = noise_blocks[:, frame_offset:frame_offset + num_fpb]
            block_token_start = start_token + frame_offset * fsl
            noisy_input = block_noise

            for step_idx, ts_val in enumerate(self.denoising_step_list):
                timestep = torch.ones(
                    [B, num_fpb], device=device, dtype=torch.int64) * ts_val

                step_needs_grad = enable_grad and (step_idx < grad_steps)

                if step_needs_grad and use_ckpt:
                    denoised_pred = torch.utils.checkpoint.checkpoint(
                        _generator_forward,
                        noisy_input, conditional_dict, timestep,
                        kv_cache, crossattn_cache, block_token_start,
                        use_reentrant=False,
                    )
                elif step_needs_grad:
                    with torch.enable_grad():
                        denoised_pred = _generator_forward(
                            noisy_input, conditional_dict, timestep,
                            kv_cache, crossattn_cache, block_token_start,
                        )
                else:
                    with torch.no_grad():
                        denoised_pred = _generator_forward(
                            noisy_input, conditional_dict, timestep,
                            kv_cache, crossattn_cache, block_token_start,
                        )

                is_last_step = (step_idx == num_steps - 1)
                if not is_last_step:
                    next_ts = self.denoising_step_list[step_idx + 1]
                    if step_needs_grad:
                        noisy_input = self.scheduler.add_noise(
                            denoised_pred.flatten(0, 1),
                            torch.randn_like(denoised_pred.flatten(0, 1)),
                            next_ts * torch.ones(
                                [B * num_fpb], device=device, dtype=torch.long),
                        ).unflatten(0, (B, num_fpb))
                    else:
                        with torch.no_grad():
                            noisy_input = self.scheduler.add_noise(
                                denoised_pred.flatten(0, 1),
                                torch.randn_like(denoised_pred.flatten(0, 1)),
                                next_ts * torch.ones(
                                    [B * num_fpb], device=device, dtype=torch.long),
                            ).unflatten(0, (B, num_fpb))

            block_outputs.append(denoised_pred)

            if bi < num_blocks - 1:
                with torch.no_grad():
                    ctx_ts = torch.ones(
                        [B, num_fpb], device=device, dtype=torch.int64
                    ) * self.pipeline.context_noise
                    ctx_noisy = self.scheduler.add_noise(
                        denoised_pred.detach().flatten(0, 1),
                        torch.randn_like(denoised_pred.flatten(0, 1)),
                        ctx_ts.flatten(0, 1),
                    ).unflatten(0, (B, num_fpb))
                    generator(
                        noisy_image_or_video=ctx_noisy,
                        conditional_dict=conditional_dict,
                        timestep=ctx_ts,
                        kv_cache=kv_cache,
                        crossattn_cache=crossattn_cache,
                        current_start=block_token_start,
                        update_cache=True,
                    )

        return torch.cat(block_outputs, dim=1)

    def _generate_postfork_anchor(
        self,
        generator,
        rollout_result: Dict[str, Any],
        grad_blocks: int,
        grad_steps: int,
        enable_grad: bool,
        max_postfork_frames: Optional[int] = None,
    ) -> torch.Tensor:
        """Generate the full post-fork latent sequence (fork → end of video).

        The first ``grad_blocks`` blocks carry gradient (via ``_denoise_blocks``).
        The remaining chunks are generated with the pipeline (no grad).
        When ``max_postfork_frames`` is set, generation stops once that many
        post-fork latent frames have been produced and returns only that window.
        """
        device, dtype = self.device, self.dtype
        kv_saved = rollout_result["kv_state_before_fork"]
        fork_abs = rollout_result["fork_abs_frame"]
        noise_block = rollout_result["noise_block_b"]
        cond_list = rollout_result["cond_list"]
        chunks = rollout_result["chunks"]
        perturb_idx = rollout_result["perturb_idx"]
        perturb_chunk_noise = rollout_result["perturb_chunk_noise"].to(device=device, dtype=dtype)
        fork_offset = rollout_result["fork_offset_in_chunk"]
        post_fork_rng_state = rollout_result["post_fork_rng_state"]
        noise_seed = rollout_result["noise_seed"]
        switch_indices = rollout_result["switch_frame_indices"]
        global_sink = rollout_result["global_sink"]
        cond_at_fork = rollout_result["conditional_dict"]

        torch.random.set_rng_state(rollout_result["saved_cpu_before_fork"])
        torch.cuda.set_rng_state(rollout_result["saved_cuda_before_fork"], device)

        num_fpb = self.pipeline.num_frame_per_block
        fsl = self.pipeline.frame_seq_length
        fork_token = fork_abs * fsl
        P = grad_blocks

        pipeline_generator_backup = self.pipeline.generator
        self.pipeline.generator = generator
        try:
            self.pipeline.deactivate_perturbation()
            kv_cache = self._materialize_cache_list(kv_saved["kv_cache1"])
            xattn_cache = self._materialize_crossattn_cache(kv_saved["crossattn_cache"])

            anchor_head = self._denoise_blocks(
                generator, noise_block, cond_at_fork,
                kv_cache, xattn_cache, fork_token,
                num_blocks=P, grad_steps=grad_steps, enable_grad=enable_grad,
            )
            if max_postfork_frames is not None and anchor_head.shape[1] >= max_postfork_frames:
                return anchor_head[:, :max_postfork_frames]

            with torch.no_grad():
                last_block = anchor_head[:, -num_fpb:].detach()
                ctx_ts = torch.ones([1, num_fpb], device=device, dtype=torch.int64) * self.pipeline.context_noise
                ctx_noisy = self.scheduler.add_noise(
                    last_block.flatten(0, 1),
                    torch.randn_like(last_block.flatten(0, 1)),
                    ctx_ts.flatten(0, 1),
                ).unflatten(0, (1, num_fpb))
                last_token = fork_token + (P - 1) * num_fpb * fsl
                generator(
                    noisy_image_or_video=ctx_noisy,
                    conditional_dict=cond_at_fork,
                    timestep=ctx_ts,
                    kv_cache=kv_cache, crossattn_cache=xattn_cache,
                    current_start=last_token,
                    update_cache=True,
                )

            self.pipeline._restore_cache_list(self.pipeline.kv_cache1, kv_cache)
            self.pipeline._restore_crossattn_cache(self.pipeline.crossattn_cache, xattn_cache)
            del kv_cache, xattn_cache

            tail_parts = []
            cur_len = fork_abs + P * num_fpb
            remaining_in_chunk = perturb_chunk_noise[:, fork_offset + P * num_fpb:]
            if max_postfork_frames is not None:
                keep_frames = max(fork_abs + max_postfork_frames - cur_len, 0)
                remaining_in_chunk = remaining_in_chunk[:, :keep_frames]
            if remaining_in_chunk.shape[1] > 0:
                out = self.pipeline.generate_chunk_sampling(
                    noise=remaining_in_chunk,
                    conditional_dict=cond_at_fork,
                    current_start_frame=cur_len,
                )
                tail_parts.append(out.detach())
                cur_len += remaining_in_chunk.shape[1]

            if max_postfork_frames is not None and cur_len >= fork_abs + max_postfork_frames:
                all_parts = [anchor_head] + tail_parts
                return torch.cat(all_parts, dim=1)[:, :max_postfork_frames]

            def _seg_for_frame(f):
                s = 0
                for si in switch_indices:
                    if f >= si:
                        s += 1
                    else:
                        break
                return s

            prefix_latents = rollout_result.get("prefix_latents", [])

            branch_rng = torch.Generator(device=device)
            branch_rng.manual_seed(noise_seed)
            branch_rng.set_state(post_fork_rng_state)
            seg_idx = _seg_for_frame(cur_len)

            with torch.no_grad():
                for ci in range(perturb_idx + 1, len(chunks)):
                    if max_postfork_frames is not None and cur_len >= fork_abs + max_postfork_frames:
                        break
                    ch = chunks[ci]
                    nf = ch["new_frames"]
                    if nf <= 0:
                        continue
                    if max_postfork_frames is not None:
                        nf = min(nf, fork_abs + max_postfork_frames - cur_len)
                        if nf <= 0:
                            break
                    new_seg = _seg_for_frame(cur_len)
                    if new_seg != seg_idx:
                        all_so_far = (
                            list(prefix_latents)
                            + [anchor_head.detach().cpu()]
                            + [t.cpu() for t in tail_parts]
                        )
                        self.pipeline.recache_after_switch(
                            all_so_far, cur_len, cond_list[new_seg], global_sink)
                        seg_idx = new_seg
                    noise = torch.randn([1, nf, 16, 60, 104],
                                        generator=branch_rng, device=device, dtype=dtype)
                    out = self.pipeline.generate_chunk_sampling(
                        noise=noise,
                        conditional_dict=cond_list[seg_idx],
                        current_start_frame=cur_len,
                    )
                    tail_parts.append(out.detach())
                    cur_len += nf

            all_parts = [anchor_head] + tail_parts
            anchor_postfork = torch.cat(all_parts, dim=1)
            if max_postfork_frames is not None:
                anchor_postfork = anchor_postfork[:, :max_postfork_frames]
            return anchor_postfork
        finally:
            self.pipeline.generator = pipeline_generator_backup

    # ------------------------------------------------------------------
    # Train step
    # ------------------------------------------------------------------

    def _get_sigma_for_timestep(self, ts_val) -> float:
        """Get the effective sigma for log-probability weighting.

        Uses a floor of ``log_prob_sigma_floor`` (default 1.0) to prevent
        the 1/(2*sigma^2) amplification from blowing up the log-probability.
        A floor of 1.0 means all steps are weighted equally (log_prob = -MSE/2).
        """
        sched = self.scheduler
        ts_float = float(ts_val)
        ts_tensor = torch.tensor([ts_float], dtype=torch.float32, device=sched.timesteps.device)
        ts_id = torch.argmin((sched.timesteps.unsqueeze(0) - ts_tensor.unsqueeze(1)).abs(), dim=1)
        sigma = float(sched.sigmas[ts_id].item())
        floor = float(getattr(self.config, "log_prob_sigma_floor", 1.0))
        return max(sigma, floor)

    def _compute_trajectory_log_prob(
        self,
        generator: torch.nn.Module,
        rollout_result: Dict[str, Any],
        branch_trajectories: List[Dict[str, Any]],
        grad_steps: int = 2,
        enable_grad: bool = True,
        branch_indices: Optional[Sequence[int]] = None,
    ) -> torch.Tensor:
        """Compute per-step diffusion log-probability by replaying branch trajectories.

        For each branch, feeds the branch's saved noisy inputs into the given
        generator and computes:
            log p(branch_x0 | branch_xt, t) = -||branch_x0 - gen_pred||^2 / (2*sigma_t^2)
        summed over all blocks and denoising steps.

        Args:
            generator: the generator to evaluate (current or old).
            rollout_result: dict from rollout containing KV state etc.
            branch_trajectories: list of G trajectory dicts, each with
                'noisy_inputs' and 'x0_preds' (list-of-lists on CPU).
            grad_steps: how many of the denoising steps carry gradient.
            enable_grad: master grad switch.

        Returns:
            log_probs: tensor [G] of total log-probabilities per branch.
        """
        device, dtype = self.device, self.dtype
        G = len(branch_trajectories)
        active_branch_ids = (
            list(range(G))
            if branch_indices is None
            else [int(idx) for idx in branch_indices]
        )
        num_fpb = self.pipeline.num_frame_per_block
        fsl = self.pipeline.frame_seq_length
        default_kv_saved = rollout_result["kv_state_before_fork"]
        fork_abs = rollout_result["fork_abs_frame"]
        cond_at_fork = rollout_result["conditional_dict"]
        fork_token = fork_abs * fsl
        use_ckpt = self.gradient_checkpointing and enable_grad

        num_blocks = len(branch_trajectories[0]["noisy_inputs"])
        num_steps = len(branch_trajectories[0]["noisy_inputs"][0])

        total_log_prob = torch.zeros(len(active_branch_ids), device=device, dtype=torch.float32)

        def _gen_forward_single(noisy_in, cond, ts, kv, xattn, cur_start):
            _, pred = generator(
                noisy_image_or_video=noisy_in,
                conditional_dict=cond,
                timestep=ts,
                kv_cache=kv,
                crossattn_cache=xattn,
                current_start=cur_start,
                update_cache=False,
            )
            return pred

        for out_idx, g in enumerate(active_branch_ids):
            kv_cache = self._materialize_cache_list(default_kv_saved["kv_cache1"])
            xattn_cache = self._materialize_crossattn_cache(default_kv_saved["crossattn_cache"])

            branch_log_prob = torch.tensor(0.0, device=device, dtype=torch.float32)

            for bi in range(num_blocks):
                block_token_start = fork_token + bi * num_fpb * fsl

                for step_idx in range(num_steps):
                    ts_val = self.denoising_step_list[step_idx]
                    sigma_t = self._get_sigma_for_timestep(int(ts_val))
                    sigma_t_sq = max(sigma_t ** 2, 1e-10)

                    b_input = branch_trajectories[g]["noisy_inputs"][bi][step_idx].to(
                        device=device, dtype=dtype)
                    b_target = branch_trajectories[g]["x0_preds"][bi][step_idx].to(
                        device=device, dtype=dtype)

                    timestep = torch.ones(
                        [1, num_fpb], device=device, dtype=torch.int64,
                    ) * ts_val

                    step_needs_grad = enable_grad and (step_idx < grad_steps)

                    if step_needs_grad and use_ckpt:
                        gen_pred = torch.utils.checkpoint.checkpoint(
                            _gen_forward_single,
                            b_input, cond_at_fork, timestep,
                            kv_cache, xattn_cache, block_token_start,
                            use_reentrant=False,
                        )
                    elif step_needs_grad:
                        with torch.enable_grad():
                            gen_pred = _gen_forward_single(
                                b_input, cond_at_fork, timestep,
                                kv_cache, xattn_cache, block_token_start,
                            )
                    else:
                        with torch.no_grad():
                            gen_pred = _gen_forward_single(
                                b_input, cond_at_fork, timestep,
                                kv_cache, xattn_cache, block_token_start,
                            )

                    diff = (b_target - gen_pred).float()
                    mse = diff.pow(2).mean()
                    step_lp = -mse / (2.0 * sigma_t_sq)
                    branch_log_prob = branch_log_prob + step_lp

                    if self.is_main and g == 0 and bi == 0 and enable_grad:
                        print(f"    [TRAJ] step={step_idx} sigma={sigma_t:.6f} "
                              f"mse={mse.item():.6f} step_lp={step_lp.item():.4f}")

                if bi < num_blocks - 1:
                    with torch.no_grad():
                        ctx_ts = torch.ones(
                            [1, num_fpb], device=device, dtype=torch.int64,
                        ) * self.pipeline.context_noise
                        ctx_noisy = self.scheduler.add_noise(
                            gen_pred.detach().flatten(0, 1),
                            torch.randn_like(gen_pred.flatten(0, 1)),
                            ctx_ts.flatten(0, 1),
                        ).unflatten(0, (1, num_fpb))
                        generator(
                            noisy_image_or_video=ctx_noisy,
                            conditional_dict=cond_at_fork,
                            timestep=ctx_ts,
                            kv_cache=kv_cache,
                            crossattn_cache=xattn_cache,
                            current_start=block_token_start,
                            update_cache=True,
                        )

            total_log_prob[out_idx] = branch_log_prob
            del kv_cache, xattn_cache

        return total_log_prob

    def _compute_policy_log_scores(
        self,
        generator: torch.nn.Module,
        rollout_result: Dict[str, Any],
        branch_trajectories: List[Dict[str, Any]],
        grad_steps: int = 2,
        enable_grad: bool = True,
        branch_indices: Optional[Sequence[int]] = None,
    ) -> torch.Tensor:
        """Compute per-branch policy logits for the configured policy mode."""
        if self.policy_log_prob_mode == "per_step_x":
            return self._compute_trajectory_log_prob(
                generator,
                rollout_result,
                branch_trajectories,
                grad_steps=grad_steps,
                enable_grad=enable_grad,
                branch_indices=branch_indices,
            )
        raise RuntimeError(
            f"Unsupported policy_log_prob_mode={self.policy_log_prob_mode!r}"
        )

    def _discrete_kl_pi_vs_log_ref(
        self, pi_theta: torch.Tensor, log_pi_ref: torch.Tensor
    ) -> torch.Tensor:
        """KL(π_θ || π_ref) for discrete π_θ with log π_ref on the same G-simplex."""
        log_pi_theta = torch.log(pi_theta + 1e-10)
        return (pi_theta * (log_pi_theta - log_pi_ref)).sum()

    def _compute_try_kl_anchor_l2(
        self,
        rollout_result: Dict[str, Any],
        policy_window_frames: int,
        grad_steps: int,
        current_generator: torch.nn.Module,
        ref_generator: torch.nn.Module,
        current_anchor: Optional[torch.Tensor] = None,
        current_anchor_grad_blocks: int = 0,
    ) -> Tuple[torch.Tensor, Dict[str, Any], Optional[torch.Tensor]]:
        """Regularize replay by summing current/ref trajectory divergence.

        ``try_kl`` is intentionally not a strict KL. Instead, starting from the
        same post-fork anchor noise, current/ref generators each roll forward on
        their own predicted trajectory and we penalize the per-step MSE between
        their x0 predictions. This makes the regularizer sensitive to both
        instantaneous prediction differences and accumulated trajectory drift.

        Only the first ``grad_steps`` denoising steps carry gradient on the
        current branch; the remaining terms are constants, mirroring
        ``_compute_trajectory_log_prob``.
        """
        if policy_window_frames <= 0:
            raise ValueError(
                f"try_kl requires positive policy_window_frames, got {policy_window_frames}"
            )

        num_fpb = int(self.pipeline.num_frame_per_block)
        max_window_blocks = max(1, (int(policy_window_frames) + num_fpb - 1) // num_fpb)
        anchor_grad_blocks = max(1, min(int(current_anchor_grad_blocks), max_window_blocks))
        num_steps = len(self.denoising_step_list)
        use_ckpt = self.gradient_checkpointing
        device, dtype = self.device, self.dtype

        kv_saved = rollout_result["kv_state_before_fork"]
        fork_abs = rollout_result["fork_abs_frame"]
        noise_block = rollout_result["noise_block_b"].to(device=device, dtype=dtype)
        cond_at_fork = rollout_result["conditional_dict"]
        fork_token = fork_abs * self.pipeline.frame_seq_length

        ref_to_gpu = None
        if ref_generator is self.initial_generator:
            self._initial_gen_to_gpu()
            ref_to_gpu = self._initial_gen_to_cpu
        elif ref_generator is self.old_generator:
            self._old_gen_to_gpu()
            ref_to_gpu = self._old_gen_to_cpu

        saved_cpu_rng_state = torch.random.get_rng_state()
        saved_cuda_rng_state = None
        if torch.cuda.is_available():
            saved_cuda_rng_state = torch.cuda.get_rng_state(self.device)

        try:
            del current_anchor

            current_kv_cache = self._materialize_cache_list(kv_saved["kv_cache1"])
            current_xattn_cache = self._materialize_crossattn_cache(kv_saved["crossattn_cache"])
            ref_kv_cache = self._materialize_cache_list(kv_saved["kv_cache1"])
            ref_xattn_cache = self._materialize_crossattn_cache(kv_saved["crossattn_cache"])

            total_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
            n_terms = 0

            def _current_forward(noisy_in, ts, kv_cache, xattn_cache, cur_start):
                _, pred = current_generator(
                    noisy_image_or_video=noisy_in,
                    conditional_dict=cond_at_fork,
                    timestep=ts,
                    kv_cache=kv_cache,
                    crossattn_cache=xattn_cache,
                    current_start=cur_start,
                    update_cache=False,
                )
                return pred

            for bi in range(anchor_grad_blocks):
                frame_offset = bi * num_fpb
                block_token_start = fork_token + frame_offset * self.pipeline.frame_seq_length
                base_noisy = noise_block[:, frame_offset:frame_offset + num_fpb]
                current_noisy = base_noisy
                ref_noisy = base_noisy
                current_pred = None
                ref_pred = None

                for step_idx, ts_val in enumerate(self.denoising_step_list):
                    timestep = torch.ones(
                        [1, num_fpb], device=device, dtype=torch.int64
                    ) * ts_val
                    step_needs_grad = step_idx < grad_steps

                    if step_needs_grad and use_ckpt:
                        current_pred = torch.utils.checkpoint.checkpoint(
                            _current_forward,
                            current_noisy,
                            timestep,
                            current_kv_cache,
                            current_xattn_cache,
                            block_token_start,
                            use_reentrant=False,
                        )
                    elif step_needs_grad:
                        with torch.enable_grad():
                            current_pred = _current_forward(
                                current_noisy,
                                timestep,
                                current_kv_cache,
                                current_xattn_cache,
                                block_token_start,
                            )
                    else:
                        with torch.no_grad():
                            current_pred = _current_forward(
                                current_noisy,
                                timestep,
                                current_kv_cache,
                                current_xattn_cache,
                                block_token_start,
                            )

                    with torch.no_grad():
                        _, ref_pred = ref_generator(
                            noisy_image_or_video=ref_noisy,
                            conditional_dict=cond_at_fork,
                            timestep=timestep,
                            kv_cache=ref_kv_cache,
                            crossattn_cache=ref_xattn_cache,
                            current_start=block_token_start,
                            update_cache=False,
                        )

                    step_mse = F.mse_loss(
                        current_pred.float(),
                        ref_pred.float(),
                        reduction="mean",
                    )
                    total_loss = total_loss + step_mse
                    n_terms += 1

                    is_last_step = (step_idx == num_steps - 1)
                    if not is_last_step:
                        next_ts = self.denoising_step_list[step_idx + 1]
                        next_timestep = next_ts * torch.ones(
                            [num_fpb], device=device, dtype=torch.long
                        )
                        step_noise = torch.randn_like(current_pred.flatten(0, 1))
                        # Let current/ref each continue on their own replay path
                        # so the regularizer captures accumulated trajectory drift.
                        if step_needs_grad:
                            current_noisy = self.scheduler.add_noise(
                                current_pred.flatten(0, 1),
                                step_noise,
                                next_timestep,
                            ).unflatten(0, (1, num_fpb))
                        else:
                            with torch.no_grad():
                                current_noisy = self.scheduler.add_noise(
                                    current_pred.flatten(0, 1),
                                    step_noise,
                                    next_timestep,
                                ).unflatten(0, (1, num_fpb))
                        with torch.no_grad():
                            ref_noisy = self.scheduler.add_noise(
                                ref_pred.flatten(0, 1),
                                step_noise,
                                next_timestep,
                            ).unflatten(0, (1, num_fpb))

                if bi < anchor_grad_blocks - 1:
                    with torch.no_grad():
                        ctx_ts = torch.ones(
                            [1, num_fpb], device=device, dtype=torch.int64
                        ) * self.pipeline.context_noise
                        ctx_noise = torch.randn_like(current_pred.flatten(0, 1))
                        current_ctx_noisy = self.scheduler.add_noise(
                            current_pred.detach().flatten(0, 1),
                            ctx_noise,
                            ctx_ts.flatten(0, 1),
                        ).unflatten(0, (1, num_fpb))
                        ref_ctx_noisy = self.scheduler.add_noise(
                            ref_pred.detach().flatten(0, 1),
                            ctx_noise,
                            ctx_ts.flatten(0, 1),
                        ).unflatten(0, (1, num_fpb))
                        current_generator(
                            noisy_image_or_video=current_ctx_noisy,
                            conditional_dict=cond_at_fork,
                            timestep=ctx_ts,
                            kv_cache=current_kv_cache,
                            crossattn_cache=current_xattn_cache,
                            current_start=block_token_start,
                            update_cache=True,
                        )
                        ref_generator(
                            noisy_image_or_video=ref_ctx_noisy,
                            conditional_dict=cond_at_fork,
                            timestep=ctx_ts,
                            kv_cache=ref_kv_cache,
                            crossattn_cache=ref_xattn_cache,
                            current_start=block_token_start,
                            update_cache=True,
                        )
        finally:
            torch.random.set_rng_state(saved_cpu_rng_state)
            if saved_cuda_rng_state is not None:
                torch.cuda.set_rng_state(saved_cuda_rng_state, self.device)
            if ref_to_gpu is not None:
                ref_to_gpu()

        if n_terms <= 0:
            raise ValueError("try_kl replay produced no denoising terms")

        loss_l2 = total_loss / float(n_terms)
        info = {
            "try_kl_anchor_mse": loss_l2.detach().item(),
            "try_kl_policy_window_frames": int(policy_window_frames),
            "try_kl_anchor_grad_blocks": int(anchor_grad_blocks),
            "try_kl_denoise_grad_steps": int(grad_steps),
            "try_kl_denoise_terms": int(n_terms),
            "try_kl_trajectory": "independent_rollout_divergence",
            "try_kl_ref": (
                "initial"
                if (self.kl_reference_initial and ref_generator is self.initial_generator)
                else "old"
            ),
        }
        return loss_l2, info, None

    def _compute_kl_regularizer(
        self,
        rollout_result: Optional[Dict[str, Any]],
        policy_window_frames: Optional[int],
        grad_steps: int,
        pi_theta: Optional[torch.Tensor] = None,
        log_pi_ref: Optional[torch.Tensor] = None,
        current_anchor: Optional[torch.Tensor] = None,
        current_anchor_grad_blocks: int = 0,
        current_generator: Optional[torch.nn.Module] = None,
        ref_generator: Optional[torch.nn.Module] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any], Optional[torch.Tensor]]:
        """Compute the configured KL-style regularizer for KVPO replay."""
        if self.kl_loss_mode == "discrete_kl":
            del rollout_result, policy_window_frames, grad_steps, current_anchor, current_anchor_grad_blocks
            del current_generator, ref_generator
            if pi_theta is None or log_pi_ref is None:
                raise ValueError("discrete_kl requires pi_theta and log_pi_ref")
            return self._discrete_kl_pi_vs_log_ref(pi_theta, log_pi_ref), {}, None

        if self.kl_loss_mode == "try_kl":
            if rollout_result is None or policy_window_frames is None:
                raise ValueError("try_kl requires rollout_result and policy_window_frames")
            if current_generator is None or ref_generator is None:
                raise ValueError("try_kl requires current_generator and ref_generator")
            return self._compute_try_kl_anchor_l2(
                rollout_result,
                int(policy_window_frames),
                grad_steps,
                current_generator=current_generator,
                ref_generator=ref_generator,
                current_anchor=current_anchor,
                current_anchor_grad_blocks=current_anchor_grad_blocks,
            )

        raise RuntimeError(f"Unsupported kl_loss_mode={self.kl_loss_mode!r}")

    def _iter_branch_backward_chunks(self, num_branches: int) -> List[List[int]]:
        chunk_size = int(self.branch_backward_chunk_size)
        if chunk_size <= 0 or chunk_size >= num_branches:
            return [list(range(num_branches))]
        return [
            list(range(start, min(start + chunk_size, num_branches)))
            for start in range(0, num_branches, chunk_size)
        ]

    def _compute_policy_objective_from_logits(
        self,
        log_pi_theta: torch.Tensor,
        log_pi_old: torch.Tensor,
        fused_adv: torch.Tensor,
        *,
        rank_is_safe_policy: bool,
        rollout_result: Dict[str, Any],
        policy_window_frames: int,
        log_pi_initial: Optional[torch.Tensor] = None,
        current_generator: Optional[torch.nn.Module] = None,
        current_anchor_grad_blocks: int = 0,
        current_anchor_grad_steps: int = 0,
        try_kl_term: Optional[torch.Tensor] = None,
        try_kl_info: Optional[Dict[str, Any]] = None,
        detach_try_kl_term: bool = False,
    ) -> Dict[str, Any]:
        log_pi_old_det = log_pi_old.detach()
        log_z_theta = torch.logsumexp(log_pi_theta, dim=0)
        log_z_old = torch.logsumexp(log_pi_old_det, dim=0)
        log_ratio = (log_pi_theta - log_z_theta) - (log_pi_old_det - log_z_old)
        log_ratio = torch.clamp(log_ratio, -20.0, 20.0)
        ratio = torch.exp(log_ratio)

        loss_kl_contrib = 0.0
        loss_grpo_item = 0.0
        kl_info: Dict[str, Any] = {}
        m2_entropy = None
        m2_pi_anchor = None
        m2_scale = None
        loss_kl_term = None

        def _resolve_loss_kl_term() -> Optional[torch.Tensor]:
            nonlocal kl_info
            if self.kl_coef == 0.0:
                return None

            if self.kl_loss_mode == "try_kl":
                resolved = try_kl_term
                if resolved is None:
                    use_initial_ref = (
                        self.kl_reference_initial and self.initial_generator is not None
                    )
                    kl_ref_generator = (
                        self.initial_generator if use_initial_ref else self.old_generator
                    )
                    resolved, kl_info_local, _ = self._compute_kl_regularizer(
                        rollout_result,
                        policy_window_frames,
                        grad_steps=current_anchor_grad_steps,
                        current_anchor_grad_blocks=current_anchor_grad_blocks,
                        current_generator=current_generator,
                        ref_generator=kl_ref_generator,
                    )
                    kl_info = dict(kl_info_local)
                else:
                    kl_info = dict(try_kl_info or {})
                return resolved.detach() if detach_try_kl_term else resolved

            use_initial_ref = (
                self.kl_reference_initial and self.initial_generator is not None
            )
            log_pi_kl_ref = (
                log_pi_initial
                if use_initial_ref and log_pi_initial is not None
                else log_pi_old
            )
            log_pi_branch_theta = self._per_step_branch_log_pi(log_pi_theta)
            pi_theta_branch = log_pi_branch_theta.exp()
            log_pi_branch_ref = self._per_step_branch_log_pi(log_pi_kl_ref.detach())
            resolved, kl_info_local, _ = self._compute_kl_regularizer(
                rollout_result,
                policy_window_frames,
                grad_steps=current_anchor_grad_steps,
                pi_theta=pi_theta_branch,
                log_pi_ref=log_pi_branch_ref,
                current_anchor_grad_blocks=current_anchor_grad_blocks,
                current_generator=current_generator,
            )
            kl_info = dict(kl_info_local)
            return resolved

        if self.per_step_protect_mode == "mechanism_1":
            if not rank_is_safe_policy:
                total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
                clip_frac = torch.tensor(0.0, device=self.device, dtype=torch.float32)
            else:
                ratio_clipped = torch.clamp(ratio, self.ratio_clip_low, self.ratio_clip_high)
                surrogate = torch.minimum(ratio * fused_adv, ratio_clipped * fused_adv)
                loss_grpo_tensor = -surrogate.mean()
                loss_grpo_item = loss_grpo_tensor.detach().item()
                total_loss = loss_grpo_tensor
                loss_kl_term = _resolve_loss_kl_term()
                if loss_kl_term is not None:
                    total_loss = total_loss + self.kl_coef * loss_kl_term
                    loss_kl_contrib = (self.kl_coef * loss_kl_term).detach().item()
                clip_frac = (
                    (ratio - torch.clamp(ratio, self.ratio_clip_low, self.ratio_clip_high)).abs() > 1e-6
                ).float().mean()
        else:
            log_pi_branch_theta = self._per_step_branch_log_pi(log_pi_theta)
            m2_scale, m2_entropy, m2_pi_anchor = self._per_step_mechanism2_anchor_scale_entropy(
                log_pi_branch_theta, rank_is_safe_policy
            )
            ratio_clipped = torch.clamp(ratio, self.ratio_clip_low, self.ratio_clip_high)
            surrogate = torch.minimum(ratio * fused_adv, ratio_clipped * fused_adv)
            loss_grpo_tensor = -surrogate.mean()
            loss_kl_term = _resolve_loss_kl_term()
            if loss_kl_term is not None:
                total_loss = m2_scale * (loss_grpo_tensor + self.kl_coef * loss_kl_term)
                loss_kl_contrib = (m2_scale * self.kl_coef * loss_kl_term).detach().item()
            else:
                total_loss = m2_scale * loss_grpo_tensor
            loss_grpo_item = (m2_scale * loss_grpo_tensor).detach().item()
            clip_frac = (
                (ratio - torch.clamp(ratio, self.ratio_clip_low, self.ratio_clip_high)).abs() > 1e-6
            ).float().mean()

        log_ratio_std = log_ratio.std().item() if log_ratio.numel() > 1 else 0.0
        return {
            "total_loss": total_loss,
            "ratio": ratio,
            "log_ratio": log_ratio,
            "clip_fraction": clip_frac,
            "log_ratio_std": log_ratio_std,
            "loss_grpo_item": loss_grpo_item,
            "loss_kl_contrib": loss_kl_contrib,
            "kl_info": kl_info,
            "m2_entropy": m2_entropy,
            "m2_pi_anchor": m2_pi_anchor,
            "m2_scale": m2_scale,
            "loss_kl_term": loss_kl_term,
        }

    def train_step(self, rollout_result: Dict[str, Any]) -> Dict[str, Any]:
        """Execute N PPO epochs with dual-signal (local + global) combined loss.

        When ``gradient_accumulation_steps > 1``, multiple rollouts share the
        same old-policy reference (multi-rollout off-policy window).  The
        optimizer steps every PPO epoch so the generator evolves continuously,
        but ``_sync_old_policy`` and the LR scheduler only advance at the
        accumulation boundary.  This is intentionally NOT standard gradient
        accumulation (which would produce identical gradients since the anchor
        is regenerated each epoch); instead it exposes the generator to more
        diverse prompt experiences per effective training step.
        """
        self._accum_count += 1
        is_accum_boundary = (self._accum_count >= self.gradient_accumulation_steps)
        G = int(self.G)
        N = self.ppo_epochs
        P = self.perturb_num_blocks
        M = self.anchor_grad_steps
        num_fpb = self.pipeline.num_frame_per_block

        policy_targets = rollout_result.get("policy_targets")
        adv_global = rollout_result.get("adv_global")
        adv_local = rollout_result.get("adv_local")
        use_global = self.train_global and adv_global is not None
        use_local = self.train_local and adv_local is not None

        if policy_targets is None:
            raise ValueError("rollout_result is missing policy_targets for single-pi update")

        tgt_policy_flat = policy_targets.reshape(G, -1).float()
        adv_g = adv_global.float() if use_global else None
        adv_l = adv_local.float() if use_local else None
        fused_adv_raw, fusion_info = self._fuse_signal_advantages(adv_l, adv_g)
        fused_adv = fused_adv_raw
        if self.adv_clip_max > 0:
            fused_adv = torch.clamp(fused_adv, -self.adv_clip_max, self.adv_clip_max)

        if self.is_main:
            parts = []
            if use_local:
                parts.append(f"local_adv=[{', '.join(f'{a:.3f}' for a in adv_l.cpu().tolist())}]")
            if use_global:
                parts.append(f"global_adv=[{', '.join(f'{a:.3f}' for a in adv_g.cpu().tolist())}]")
            parts.append(f"fused_adv=[{', '.join(f'{a:.3f}' for a in fused_adv.cpu().tolist())}]")
            parts.append(f"policy_frames={int(rollout_result['policy_window_frames'])}")
            print(f"[KVPO][Step {self.step + 1}] {' | '.join(parts)}")

        rewards_g_raw = rollout_result.get("rewards_global")
        rewards_l_raw = rollout_result.get("rewards_local")
        anchor_g_raw = rollout_result.get("anchor_reward_global")
        anchor_l_raw = rollout_result.get("anchor_reward_local")
        if rewards_g_raw is not None:
            rewards_g_raw = rewards_g_raw.to(device=self.device, dtype=torch.float32)
        if rewards_l_raw is not None:
            rewards_l_raw = rewards_l_raw.to(device=self.device, dtype=torch.float32)
        if anchor_g_raw is not None:
            anchor_g_raw = anchor_g_raw.to(device=self.device, dtype=torch.float32)
        if anchor_l_raw is not None:
            anchor_l_raw = anchor_l_raw.to(device=self.device, dtype=torch.float32)
        local_signal_safe = False
        global_signal_safe = False
        if rewards_g_raw is not None and anchor_g_raw is not None:
            global_signal_safe = bool((rewards_g_raw > anchor_g_raw).any())
        if rewards_l_raw is not None and anchor_l_raw is not None:
            local_signal_safe = bool((rewards_l_raw > anchor_l_raw).any())

        # Align with memflow: OR across available signals — any branch beats anchor on global or local.
        any_branch_better = False
        if rewards_g_raw is not None and anchor_g_raw is not None:
            any_branch_better = any_branch_better or bool((rewards_g_raw > anchor_g_raw).any())
        if rewards_l_raw is not None and anchor_l_raw is not None:
            any_branch_better = any_branch_better or bool((rewards_l_raw > anchor_l_raw).any())
        rank_is_safe_policy = any_branch_better

        global_has_safe_policy = self._distributed_bool_any(rank_is_safe_policy)
        if self.is_main:
            g_best = f"{rewards_g_raw.max().item():.4f}" if rewards_g_raw is not None else "-"
            g_anc = f"{anchor_g_raw.item():.4f}" if anchor_g_raw is not None else "-"
            l_best = f"{rewards_l_raw.max().item():.4f}" if rewards_l_raw is not None else "-"
            l_anc = f"{anchor_l_raw.item():.4f}" if anchor_l_raw is not None else "-"
            print(
                f"  [PROTECT] local_signal_safe={local_signal_safe if use_local else 'n/a'} "
                f"global_signal_safe={global_signal_safe if use_global else 'n/a'} "
                f"rank_is_safe={rank_is_safe_policy} global_has_safe={global_has_safe_policy} "
                f"(global: best={g_best} vs anchor={g_anc}, "
                f"local: best={l_best} vs anchor={l_anc})"
            )

        # --- Compute pi_old and run PPO epochs (per-step trajectory replay only) ---
        branch_trajectories = rollout_result.get("branch_trajectories")
        if branch_trajectories is None:
            raise ValueError(
                "longlive KVPO requires rollout_result['branch_trajectories'] for per-step training."
            )
        policy_window_frames = int(rollout_result["policy_window_frames"])
        self.generator.train()
        epoch_logs: List[Dict[str, Any]] = []
        log_prob_M = self.log_prob_grad_steps

        self._old_gen_to_gpu()
        pipeline_gen_backup = self.pipeline.generator
        self.pipeline.generator = self.old_generator
        with torch.no_grad():
            log_pi_old = self._compute_policy_log_scores(
                self.old_generator, rollout_result, branch_trajectories,
                grad_steps=0, enable_grad=False,
            ).detach()
        self.pipeline.generator = pipeline_gen_backup
        self._old_gen_to_cpu()

        log_pi_initial = None
        if self.kl_reference_initial and self.initial_generator is not None:
            self._initial_gen_to_gpu()
            self.pipeline.generator = self.initial_generator
            with torch.no_grad():
                log_pi_initial = self._compute_policy_log_scores(
                    self.initial_generator, rollout_result, branch_trajectories,
                    grad_steps=0, enable_grad=False,
                ).detach()
            self.pipeline.generator = pipeline_gen_backup
            self._initial_gen_to_cpu()

        if self.is_main:
            lp_str = ", ".join(f"{v:.2f}" for v in log_pi_old.tolist())
            print(f"  [PI_OLD] per_step log_pi_old=[{lp_str}]")
            if log_pi_initial is not None:
                li_str = ", ".join(f"{v:.2f}" for v in log_pi_initial.tolist())
                print(f"  [PI_INIT] per_step log_pi_initial=[{li_str}] (KL ref)")

        use_branch_chunked_backward = (
            self.branch_backward_chunk_size > 0 and self.branch_backward_chunk_size < G
        )
        branch_backward_chunks = self._iter_branch_backward_chunks(G)
        if self.is_main and use_branch_chunked_backward:
            print(
                f"  [BRANCH_CHUNK] enabled: size={self.branch_backward_chunk_size}, "
                f"G={G}, chunks={len(branch_backward_chunks)}"
            )

        for epoch in range(N):
                self.optimizer.zero_grad()

                if use_branch_chunked_backward:
                    local_entropy_skip = False
                    pipeline_gen_backup = self.pipeline.generator
                    self.pipeline.generator = self.generator

                    with torch.no_grad():
                        log_pi_theta = self._compute_policy_log_scores(
                            self.generator, rollout_result, branch_trajectories,
                            grad_steps=log_prob_M, enable_grad=False,
                        ).detach()

                    try_kl_live = None
                    try_kl_info_live: Dict[str, Any] = {}
                    if self.kl_coef != 0.0 and self.kl_loss_mode == "try_kl":
                        use_initial_ref = (
                            self.kl_reference_initial and self.initial_generator is not None
                        )
                        kl_ref_generator = (
                            self.initial_generator if use_initial_ref else self.old_generator
                        )
                        try_kl_live, try_kl_info_live, _ = self._compute_kl_regularizer(
                            rollout_result,
                            policy_window_frames,
                            grad_steps=M,
                            current_anchor_grad_blocks=P,
                            current_generator=self.generator,
                            ref_generator=kl_ref_generator,
                        )

                    chunk_eval = self._compute_policy_objective_from_logits(
                        log_pi_theta,
                        log_pi_old,
                        fused_adv,
                        rank_is_safe_policy=rank_is_safe_policy,
                        rollout_result=rollout_result,
                        policy_window_frames=policy_window_frames,
                        log_pi_initial=log_pi_initial,
                        current_generator=self.generator,
                        current_anchor_grad_blocks=P,
                        current_anchor_grad_steps=M,
                        try_kl_term=try_kl_live,
                        try_kl_info=try_kl_info_live,
                        detach_try_kl_term=True,
                    )

                    total_loss = chunk_eval["total_loss"]
                    log_ratio = chunk_eval["log_ratio"]
                    ratio = chunk_eval["ratio"]
                    clip_frac = chunk_eval["clip_fraction"]
                    log_ratio_std = chunk_eval["log_ratio_std"]
                    loss_grpo_item = chunk_eval["loss_grpo_item"]
                    loss_kl_contrib = chunk_eval["loss_kl_contrib"]
                    kl_info = dict(chunk_eval["kl_info"])
                    m2_entropy = chunk_eval["m2_entropy"]
                    m2_pi_anchor = chunk_eval["m2_pi_anchor"]

                    info_policy = {
                        "loss_grpo": loss_grpo_item,
                        "loss_kl": loss_kl_contrib,
                        "log_pi_theta": log_pi_theta.detach().cpu().tolist(),
                        "log_pi_old": log_pi_old.detach().cpu().tolist(),
                        "log_ratio": log_ratio.detach().cpu().tolist(),
                        "ratio": ratio.detach().cpu().tolist(),
                        "clip_fraction": clip_frac.item() if isinstance(clip_frac, torch.Tensor) else clip_frac,
                        "log_ratio_std": log_ratio_std,
                        "is_safe_rank": rank_is_safe_policy,
                        "is_safe_local_signal": local_signal_safe,
                        "is_safe_global_signal": global_signal_safe,
                        "is_safe_global": global_has_safe_policy,
                        "per_step_protect_mode": self.per_step_protect_mode,
                        "mode": self.policy_log_prob_mode,
                        "kl_loss_mode": self.kl_loss_mode,
                        "branch_backward_chunk_size": self.branch_backward_chunk_size,
                    }
                    info_policy.update(kl_info)
                    if self.per_step_protect_mode == "mechanism_2":
                        info_policy["entropy"] = m2_entropy.detach().item()
                        info_policy["pi_anchor"] = m2_pi_anchor.detach().item()
                        info_policy["m2_loss_scale"] = (1.0 - m2_pi_anchor.detach()).item()

                    import math as _math
                    if self.per_step_protect_mode == "mechanism_1":
                        skip_update = not global_has_safe_policy
                    else:
                        max_entropy = _math.log(G)
                        cur_ent = m2_entropy.item()
                        local_entropy_skip = (
                            self.min_update_entropy_ratio > 0
                            and max_entropy > 0
                            and cur_ent > max_entropy * (1.0 - self.min_update_entropy_ratio)
                        )
                        if self.world_size > 1:
                            skip_update = self._distributed_bool_any(local_entropy_skip)
                        else:
                            skip_update = local_entropy_skip

                    if not skip_update:
                        full_log_pi_theta_det = log_pi_theta.detach()
                        for branch_chunk in branch_backward_chunks:
                            chunk_logits = self._compute_policy_log_scores(
                                self.generator, rollout_result, branch_trajectories,
                                grad_steps=log_prob_M, enable_grad=True,
                                branch_indices=branch_chunk,
                            )
                            mixed_log_pi_theta = full_log_pi_theta_det.clone()
                            mixed_log_pi_theta[branch_chunk] = chunk_logits
                            chunk_loss = self._compute_policy_objective_from_logits(
                                mixed_log_pi_theta,
                                log_pi_old,
                                fused_adv,
                                rank_is_safe_policy=rank_is_safe_policy,
                                rollout_result=rollout_result,
                                policy_window_frames=policy_window_frames,
                                log_pi_initial=log_pi_initial,
                                current_generator=self.generator,
                                current_anchor_grad_blocks=P,
                                current_anchor_grad_steps=M,
                                try_kl_term=try_kl_live,
                                try_kl_info=try_kl_info_live,
                                detach_try_kl_term=(self.kl_loss_mode == "try_kl"),
                            )["total_loss"]
                            chunk_loss.backward()

                        if (
                            self.kl_coef != 0.0
                            and self.kl_loss_mode == "try_kl"
                            and try_kl_live is not None
                        ):
                            if self.per_step_protect_mode == "mechanism_1":
                                if rank_is_safe_policy:
                                    (self.kl_coef * try_kl_live).backward()
                            else:
                                (chunk_eval["m2_scale"].detach() * self.kl_coef * try_kl_live).backward()

                        self._average_trainable_grads()

                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            [p for p in self.generator.parameters() if p.requires_grad],
                            self.max_grad_norm,
                        )

                        self.optimizer.step()
                        grad_norm_val = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
                        if self.ema is not None:
                            self.ema.update(self.generator)
                    else:
                        self.optimizer.zero_grad()
                        grad_norm_val = 0.0

                    self.pipeline.generator = pipeline_gen_backup
                else:
                    pipeline_gen_backup = self.pipeline.generator
                    self.pipeline.generator = self.generator
                    log_pi_theta = self._compute_policy_log_scores(
                        self.generator, rollout_result, branch_trajectories,
                        grad_steps=log_prob_M, enable_grad=True,
                    )
                    self.pipeline.generator = pipeline_gen_backup

                    # PPO ratio = π_θ(g)/π_old(g) with π = softmax(trajectory scores); invariant to global score shift.
                    log_pi_old_det = log_pi_old.detach()
                    log_z_theta = torch.logsumexp(log_pi_theta, dim=0)
                    log_z_old = torch.logsumexp(log_pi_old_det, dim=0)
                    log_ratio = (log_pi_theta - log_z_theta) - (log_pi_old_det - log_z_old)
                    log_ratio = torch.clamp(log_ratio, -20.0, 20.0)
                    ratio = torch.exp(log_ratio)

                    loss_kl_contrib = 0.0
                    loss_grpo_item = 0.0
                    local_entropy_skip = False
                    kl_info: Dict[str, Any] = {}

                    if self.per_step_protect_mode == "mechanism_1":
                        if not rank_is_safe_policy:
                            total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
                            if self.is_main and epoch == 0:
                                print(
                                    "  [PROTECT] not all enabled signals beat anchor → zero policy loss "
                                    "(still all_reduce grads; update if any rank global_has_safe)"
                                )
                        else:
                            ratio_clipped = torch.clamp(ratio, self.ratio_clip_low, self.ratio_clip_high)
                            surrogate = torch.minimum(ratio * fused_adv, ratio_clipped * fused_adv)
                            loss_grpo_tensor = -surrogate.mean()
                            loss_grpo_item = loss_grpo_tensor.detach().item()
                            total_loss = loss_grpo_tensor
                            if self.kl_coef != 0.0:
                                use_initial_ref = (
                                    self.kl_reference_initial and self.initial_generator is not None
                                )
                                kl_ref_generator = (
                                    self.initial_generator if use_initial_ref else self.old_generator
                                )
                                log_pi_kl_ref = (
                                    log_pi_initial
                                    if use_initial_ref and log_pi_initial is not None
                                    else log_pi_old
                                )
                                log_pi_branch_theta = self._per_step_branch_log_pi(log_pi_theta)
                                pi_theta_branch = log_pi_branch_theta.exp()
                                log_pi_branch_ref = self._per_step_branch_log_pi(log_pi_kl_ref.detach())
                                loss_kl_term, kl_info, _ = self._compute_kl_regularizer(
                                    rollout_result,
                                    policy_window_frames,
                                    grad_steps=M,
                                    pi_theta=pi_theta_branch,
                                    log_pi_ref=log_pi_branch_ref,
                                    current_anchor_grad_blocks=P,
                                    current_generator=self.generator,
                                    ref_generator=kl_ref_generator,
                                )
                                total_loss = total_loss + self.kl_coef * loss_kl_term
                                loss_kl_contrib = (self.kl_coef * loss_kl_term).detach().item()

                        clip_frac = (
                            (ratio - torch.clamp(ratio, self.ratio_clip_low, self.ratio_clip_high)).abs() > 1e-6
                        ).float().mean() if rank_is_safe_policy else torch.tensor(0.0)
                    else:
                        # mechanism_2: same reward-based unsafe trigger; anchor slot on π_branch + entropy skip
                        m2_scale, m2_entropy, m2_pi_anchor = self._per_step_mechanism2_anchor_scale_entropy(
                            self._per_step_branch_log_pi(log_pi_theta), rank_is_safe_policy
                        )
                        ratio_clipped = torch.clamp(ratio, self.ratio_clip_low, self.ratio_clip_high)
                        surrogate = torch.minimum(ratio * fused_adv, ratio_clipped * fused_adv)
                        loss_grpo_tensor = -surrogate.mean()
                        if self.kl_coef != 0.0:
                            use_initial_ref = (
                                self.kl_reference_initial and self.initial_generator is not None
                            )
                            kl_ref_generator = (
                                self.initial_generator if use_initial_ref else self.old_generator
                            )
                            log_pi_kl_ref = (
                                log_pi_initial
                                if use_initial_ref and log_pi_initial is not None
                                else log_pi_old
                            )
                            log_pi_branch_theta = self._per_step_branch_log_pi(log_pi_theta)
                            pi_theta_branch = log_pi_branch_theta.exp()
                            log_pi_branch_ref = self._per_step_branch_log_pi(log_pi_kl_ref.detach())
                            loss_kl_term, kl_info, _ = self._compute_kl_regularizer(
                                rollout_result,
                                policy_window_frames,
                                grad_steps=M,
                                pi_theta=pi_theta_branch,
                                log_pi_ref=log_pi_branch_ref,
                                current_anchor_grad_blocks=P,
                                current_generator=self.generator,
                                ref_generator=kl_ref_generator,
                            )
                            total_loss = m2_scale * (loss_grpo_tensor + self.kl_coef * loss_kl_term)
                            loss_kl_contrib = (m2_scale * self.kl_coef * loss_kl_term).detach().item()
                        else:
                            total_loss = m2_scale * loss_grpo_tensor
                        loss_grpo_item = (m2_scale * loss_grpo_tensor).detach().item()
                        clip_frac = (
                            (ratio - torch.clamp(ratio, self.ratio_clip_low, self.ratio_clip_high)).abs() > 1e-6
                        ).float().mean()

                    log_ratio_std = log_ratio.std().item() if G > 1 else 0.0

                    info_policy = {
                        "loss_grpo": loss_grpo_item,
                        "loss_kl": loss_kl_contrib,
                        "log_pi_theta": log_pi_theta.detach().cpu().tolist(),
                        "log_pi_old": log_pi_old.detach().cpu().tolist(),
                        "log_ratio": log_ratio.detach().cpu().tolist(),
                        "ratio": ratio.detach().cpu().tolist(),
                        "clip_fraction": clip_frac.item() if isinstance(clip_frac, torch.Tensor) else clip_frac,
                        "log_ratio_std": log_ratio_std,
                        "is_safe_rank": rank_is_safe_policy,
                        "is_safe_local_signal": local_signal_safe,
                        "is_safe_global_signal": global_signal_safe,
                        "is_safe_global": global_has_safe_policy,
                        "per_step_protect_mode": self.per_step_protect_mode,
                        "mode": self.policy_log_prob_mode,
                        "kl_loss_mode": self.kl_loss_mode,
                    }
                    info_policy.update(kl_info)
                    if self.per_step_protect_mode == "mechanism_2":
                        info_policy["entropy"] = m2_entropy.detach().item()
                        info_policy["pi_anchor"] = m2_pi_anchor.detach().item()
                        info_policy["m2_loss_scale"] = (1.0 - m2_pi_anchor.detach()).item()

                # -- shared epoch tail (per_step branch) --
                epoch_log = {
                    "epoch": epoch + 1,
                    "policy": info_policy,
                    "loss_total": total_loss.item(),
                }
                if not use_branch_chunked_backward:
                    import math as _math
                    if self.per_step_protect_mode == "mechanism_1":
                        skip_update = not global_has_safe_policy
                    else:
                        max_entropy = _math.log(G)
                        cur_ent = m2_entropy.item()
                        local_entropy_skip = (
                            self.min_update_entropy_ratio > 0
                            and max_entropy > 0
                            and cur_ent > max_entropy * (1.0 - self.min_update_entropy_ratio)
                        )
                        if self.world_size > 1:
                            skip_update = self._distributed_bool_any(local_entropy_skip)
                        else:
                            skip_update = local_entropy_skip

                    total_loss.backward()
                    self._average_trainable_grads()

                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        [p for p in self.generator.parameters() if p.requires_grad],
                        self.max_grad_norm,
                    )

                    if skip_update:
                        self.optimizer.zero_grad()
                        grad_norm_val = 0.0
                    else:
                        self.optimizer.step()
                        grad_norm_val = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
                        if self.ema is not None:
                            self.ema.update(self.generator)

                epoch_log["grad_norm"] = grad_norm_val
                epoch_log["skipped_update"] = skip_update
                epoch_logs.append(epoch_log)

                if self.is_main:
                    r_str = ", ".join(f"{r:.4f}" for r in info_policy["ratio"])
                    lr_str = ", ".join(f"{v:.4f}" for v in info_policy["log_ratio"])
                    kl_s = (
                        f" kl={info_policy['loss_kl']:.6f}"
                        if self.kl_coef != 0.0
                        else ""
                    )
                    m2_dbg = ""
                    if self.per_step_protect_mode == "mechanism_2":
                        m2_dbg = (
                            f" ent={info_policy.get('entropy', 0.0):.5f} "
                            f"pi_anc={info_policy.get('pi_anchor', 0.0):.5f}"
                        )
                    print(f"  epoch {epoch+1}/{N} [PER_STEP]: "
                          f"loss={info_policy['loss_grpo']:.5f}{kl_s} clip={info_policy['clip_fraction']:.5f} "
                          f"log_ratio_std={log_ratio_std:.5f} "
                          f"protect={self.per_step_protect_mode} "
                          f"safe_rank={rank_is_safe_policy} safe_global={global_has_safe_policy}{m2_dbg}")
                    print(f"    log_ratio=[{lr_str}]")
                    print(f"    ratio    =[{r_str}]")
                    sk_extra = ""
                    if self.per_step_protect_mode == "mechanism_2":
                        sk_extra = f" (entropy_skip_local={local_entropy_skip})"
                    print(f"  epoch {epoch+1}/{N} COMBINED: loss={epoch_log['loss_total']:.5f} "
                          f"grad={grad_norm_val:.5f}"
                          f"{' [SKIPPED]' if skip_update else ''}{sk_extra}")

                del total_loss
                torch.cuda.empty_cache()

        if is_accum_boundary:
            self.lr_scheduler.step()
            self.step += 1
            self._sync_old_policy()
            self._accum_count = 0
            if self.is_main:
                print(f"  [ACCUM] boundary reached ({self.gradient_accumulation_steps} rollouts) "
                      f"→ old_policy synced, step={self.step}")

        last = epoch_logs[-1]
        rewards_g = rollout_result.get("rewards_global")
        rewards_l = rollout_result.get("rewards_local")
        rewards_g_components = rollout_result.get("rewards_global_components")
        rewards_l_components = rollout_result.get("rewards_local_components")
        adv_g_components = rollout_result.get("adv_global_components")
        adv_l_components = rollout_result.get("adv_local_components")
        anchor_g_components = rollout_result.get("anchor_reward_global_components")
        anchor_l_components = rollout_result.get("anchor_reward_local_components")
        metrics = {
            "loss_total": last["loss_total"],
            "grad_norm": last["grad_norm"],
            "step": self.step,
            "ppo_epochs": N,
            "epoch_logs": epoch_logs,
            "policy_window_frames": int(rollout_result["policy_window_frames"]),
            "advantage_fused": fused_adv.detach().cpu().tolist(),
            "advantage_fusion": fusion_info,
            "loss_grpo_policy": last.get("policy", {}).get("loss_grpo", 0),
            "clip_policy": last.get("policy", {}).get("clip_fraction", 0),
            "dist_sq_policy": last.get("policy", {}).get("dist_sq", []),
        }
        if rewards_g is not None:
            metrics["mean_reward_global"] = rewards_g.mean().item()
            metrics["reward_std_global"] = rewards_g.std().item()
            metrics["rewards_per_video_global"] = rewards_g.detach().cpu().tolist()
            metrics["anchor_reward_global"] = rollout_result["anchor_reward_global"].item()
            if rewards_g_components is not None:
                metrics["reward_components_global"] = {
                    name: {
                        "rewards_per_video": tensor.detach().cpu().tolist(),
                        "mean_reward": tensor.mean().item(),
                        "reward_std": tensor.std().item(),
                        "anchor_reward": anchor_g_components[name].item(),
                        "advantages": adv_g_components[name].detach().cpu().tolist(),
                    }
                    for name, tensor in rewards_g_components.items()
                }
        if rewards_l is not None:
            metrics["mean_reward_local"] = rewards_l.mean().item()
            metrics["reward_std_local"] = rewards_l.std().item()
            metrics["rewards_per_video_local"] = rewards_l.detach().cpu().tolist()
            metrics["anchor_reward_local"] = rollout_result["anchor_reward_local"].item()
            if rewards_l_components is not None:
                metrics["reward_components_local"] = {
                    name: {
                        "rewards_per_video": tensor.detach().cpu().tolist(),
                        "mean_reward": tensor.mean().item(),
                        "reward_std": tensor.std().item(),
                        "anchor_reward": anchor_l_components[name].item(),
                        "advantages": adv_l_components[name].detach().cpu().tolist(),
                    }
                    for name, tensor in rewards_l_components.items()
                }
        return metrics

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _get_eval_reward_setup(self) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
        eval_reward_components = getattr(self, "eval_reward_components", self.reward_components)
        eval_component_weights = {
            comp["key"]: float(comp.get("weight", 1.0)) for comp in eval_reward_components
        }
        return eval_reward_components, eval_component_weights

    def _build_eval_sampling_pipeline(self, generator: WanDiffusionWrapper) -> DiversitySamplingPipeline:
        cfg = self.config
        model_kwargs = OmegaConf.to_container(cfg.model_kwargs, resolve=True)
        return DiversitySamplingPipeline(
            denoising_step_list=self.denoising_step_list,
            scheduler=generator.get_scheduler(),
            generator=generator,
            num_frame_per_block=getattr(cfg, "num_frame_per_block", 3),
            same_step_across_blocks=getattr(cfg, "same_step_across_blocks", True),
            last_step_only=getattr(cfg, "last_step_only", True),
            context_noise=getattr(cfg, "context_noise", 0),
            local_attn_size=model_kwargs.get("local_attn_size", 12),
            slice_last_frames=getattr(cfg, "slice_last_frames", 21),
            m_nearest_frames=getattr(cfg, "m_nearest_frames", 2),
            recache_full_kv_cache_after_switch=bool(
                getattr(cfg, "recache_full_kv_cache_after_switch", False)
            ),
        )

    @torch.no_grad()
    def _create_eval_ema_context(self) -> Dict[str, Any]:
        if self.ema is None:
            raise ValueError("EMA eval context requires EMA to be enabled")
        ema_generator = copy.deepcopy(self.generator)
        ema_generator.to(device=self.device, dtype=self.dtype)
        ema_generator.requires_grad_(False)
        ema_generator.eval()
        self.ema.copy_to(ema_generator)
        return {
            "generator": ema_generator,
            "pipeline": self._build_eval_sampling_pipeline(ema_generator),
        }

    @torch.no_grad()
    def _destroy_eval_ema_context(self, ema_context: Optional[Dict[str, Any]]) -> None:
        if ema_context is None:
            return
        ema_pipeline = ema_context.get("pipeline")
        ema_generator = ema_context.get("generator")
        if ema_pipeline is not None:
            del ema_pipeline
        if ema_generator is not None:
            del ema_generator
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @torch.no_grad()
    def _prepare_eval_prompt_bundle(
        self,
        eval_items: List[Dict[str, Any]],
        prompt_idx: int,
        eval_seed: int,
    ) -> Dict[str, Any]:
        cfg = self.config
        device, dtype = self.device, self.dtype
        chunk_size = int(cfg.streaming_chunk_size)
        max_length = int(cfg.streaming_max_length)
        min_new_frame = int(cfg.streaming_min_new_frame)
        item = eval_items[prompt_idx]
        prompts_list = item["prompts"]

        cond_list = []
        for prompt in prompts_list:
            cond_dict = self.text_encoder(text_prompts=[prompt])
            cond_dict = {
                k: v.to(device=device, dtype=dtype) if isinstance(v, torch.Tensor) else v
                for k, v in cond_dict.items()
            }
            cond_list.append(cond_dict)

        plan = self.pipeline.plan_perturbation(
            total_frames=max_length,
            chunk_size=chunk_size,
            min_new_frame=min_new_frame,
            K=1,
            base_seed=eval_seed,
            sample_idx=prompt_idx,
            seed_context=0,
            perturb_within_first_x_chunks=1,
            perturb_num_blocks=1,
            sink_size=int(cfg.model_kwargs.get("sink_size", 3)),
        )
        chunks = plan["chunks"]
        switch_raw = getattr(cfg, "switch_frame_indices", "57,93,129,165,201")
        switch_indices = self._resolve_switch_frame_indices(
            switch_raw, chunks=chunks, num_segments=len(prompts_list),
        )
        perturb_idx = int(plan["perturb_chunk_idx"])
        fork_abs = int(plan["perturb_block_abs_frame"])
        fork_chunk = chunks[perturb_idx]
        local_chunk_start = int(fork_chunk["start_frame"])
        local_chunk_end = local_chunk_start + int(fork_chunk["new_frames"])
        fork_seg = self._get_segment_for_frame(fork_abs, switch_indices)
        local_prompt = prompts_list[min(fork_seg, len(prompts_list) - 1)] if prompts_list else ""

        return {
            "prompt_idx": int(prompt_idx),
            "prompts_list": prompts_list,
            "cond_list": cond_list,
            "chunks": chunks,
            "switch_indices": switch_indices,
            "fork_abs": fork_abs,
            "local_chunk_start": local_chunk_start,
            "local_chunk_end": local_chunk_end,
            "local_prompt": local_prompt,
            "noise_seed": int(eval_seed + prompt_idx * 10000),
            "global_sink": bool(getattr(cfg, "global_sink", True)),
        }

    @torch.no_grad()
    def _generate_eval_latent(
        self,
        *,
        pipeline: DiversitySamplingPipeline,
        generator: WanDiffusionWrapper,
        prompt_bundle: Dict[str, Any],
        stream: Optional[torch.cuda.Stream] = None,
    ) -> torch.Tensor:
        device, dtype = self.device, self.dtype
        chunks = prompt_bundle["chunks"]
        cond_list = prompt_bundle["cond_list"]
        switch_indices = prompt_bundle["switch_indices"]
        noise_seed = int(prompt_bundle["noise_seed"])
        global_sink = bool(prompt_bundle["global_sink"])

        batch_size = 1
        pipeline.generator = generator
        pipeline._initialize_kv_cache(batch_size, dtype, device)
        pipeline._initialize_crossattn_cache(batch_size, dtype, device)
        pipeline.clear_kv_cache()
        pipeline.generator.model.local_attn_size = int(pipeline.local_attn_size)
        pipeline._set_all_modules_max_attention_size(int(pipeline.local_attn_size))

        rng = torch.Generator(device=device)
        rng.manual_seed(noise_seed)

        def _seg_for_frame(frame_idx: int, _si=switch_indices) -> int:
            seg = 0
            for si in _si:
                if frame_idx >= si:
                    seg += 1
                else:
                    break
            return seg

        def _run_generation() -> torch.Tensor:
            latents = []
            cur_len = 0
            seg_idx = 0
            for chunk in chunks:
                new_frames = int(chunk["new_frames"])
                if new_frames <= 0:
                    continue
                new_seg = _seg_for_frame(cur_len)
                if new_seg != seg_idx:
                    pipeline.recache_after_switch(
                        latents, cur_len, cond_list[new_seg], global_sink,
                    )
                    seg_idx = new_seg
                noise = torch.randn(
                    [1, new_frames, 16, 60, 104],
                    generator=rng,
                    device=device,
                    dtype=dtype,
                )
                out = pipeline.generate_chunk_sampling(
                    noise=noise,
                    conditional_dict=cond_list[seg_idx],
                    current_start_frame=cur_len,
                    randn_generator=rng,
                )
                latents.append(out.cpu())
                cur_len += new_frames
            return torch.cat(latents, dim=1)

        if stream is not None and device.type == "cuda":
            with torch.cuda.stream(stream):
                return _run_generation()
        return _run_generation()

    @torch.no_grad()
    def _evaluate_prompt_pair_same_rank(
        self,
        eval_items: List[Dict[str, Any]],
        prompt_idx: int,
        eval_seed: int,
        eval_reward_components: List[Dict[str, Any]],
        eval_component_weights: Dict[str, float],
        ema_context: Optional[Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        prompt_bundle = self._prepare_eval_prompt_bundle(eval_items, prompt_idx, eval_seed)
        model_contexts: List[Tuple[str, WanDiffusionWrapper, DiversitySamplingPipeline]] = [
            ("online", self.generator, self.pipeline),
        ]
        if ema_context is not None:
            model_contexts.append(
                ("ema", ema_context["generator"], ema_context["pipeline"])
            )

        latent_by_model: Dict[str, torch.Tensor] = {}
        errors: Dict[str, Exception] = {}
        if len(model_contexts) > 1 and self.device.type == "cuda":
            streams = {
                model_type: torch.cuda.Stream(device=self.device)
                for model_type, _, _ in model_contexts
            }
            threads = []

            def _thread_worker(
                model_type: str,
                generator: WanDiffusionWrapper,
                pipeline: DiversitySamplingPipeline,
                stream: torch.cuda.Stream,
            ) -> None:
                try:
                    latent_by_model[model_type] = self._generate_eval_latent(
                        pipeline=pipeline,
                        generator=generator,
                        prompt_bundle=prompt_bundle,
                        stream=stream,
                    )
                except Exception as exc:  # pragma: no cover - surfaced immediately below
                    errors[model_type] = exc

            for model_type, generator, pipeline in model_contexts:
                thread = threading.Thread(
                    target=_thread_worker,
                    args=(model_type, generator, pipeline, streams[model_type]),
                    daemon=False,
                )
                thread.start()
                threads.append(thread)

            for thread in threads:
                thread.join()

            current_stream = torch.cuda.current_stream(device=self.device)
            for stream in streams.values():
                current_stream.wait_stream(stream)
        else:
            for model_type, generator, pipeline in model_contexts:
                try:
                    latent_by_model[model_type] = self._generate_eval_latent(
                        pipeline=pipeline,
                        generator=generator,
                        prompt_bundle=prompt_bundle,
                    )
                except Exception as exc:  # pragma: no cover - surfaced immediately below
                    errors[model_type] = exc

        if errors:
            raise next(iter(errors.values()))

        model_order = [model_type for model_type, _, _ in model_contexts]
        prompt_results: Dict[str, Dict[str, Any]] = {
            model_type: {
                "prompt_idx": int(prompt_bundle["prompt_idx"]),
                "fork_abs_frame": int(prompt_bundle["fork_abs"]),
                "global_start_abs_frame": 0,
                "local_chunk_latent_start": int(prompt_bundle["local_chunk_start"]),
                "local_chunk_latent_end": int(prompt_bundle["local_chunk_end"]),
            }
            for model_type in model_order
        }

        local_chunk_start = int(prompt_bundle["local_chunk_start"])
        local_chunk_end = int(prompt_bundle["local_chunk_end"])
        prompts_list = prompt_bundle["prompts_list"]
        switch_indices = prompt_bundle["switch_indices"]

        if self.train_local:
            local_eval, _ = self._compute_rewards_short(
                [latent_by_model[model_type][:, local_chunk_start:local_chunk_end].cpu() for model_type in model_order],
                prompt_bundle["local_prompt"],
                prompt_idx,
                include_aggregate=False,
                reward_components=eval_reward_components,
            )
            for model_offset, model_type in enumerate(model_order):
                prompt_result = prompt_results[model_type]
                prompt_result["local_prompt"] = prompt_bundle["local_prompt"]
                prompt_result["local_component_scores"] = {
                    name: float(comp["scores"][model_offset])
                    for name, comp in local_eval["components"].items()
                }
                prompt_result["local_total"] = sum(
                    eval_component_weights.get(name, 1.0) * score
                    for name, score in prompt_result["local_component_scores"].items()
                )

        if self.train_global:
            global_eval, _ = self._compute_rewards_long(
                [latent_by_model[model_type].cpu() for model_type in model_order],
                prompts_list,
                switch_indices,
                local_chunk_end,
                prompt_idx,
                include_aggregate=False,
                reward_components=eval_reward_components,
            )
            for model_offset, model_type in enumerate(model_order):
                prompt_result = prompt_results[model_type]
                prompt_result["global_component_scores"] = {
                    name: float(comp["scores"][model_offset])
                    for name, comp in global_eval["components"].items()
                }
                prompt_result["global_total"] = sum(
                    eval_component_weights.get(name, 1.0) * score
                    for name, score in prompt_result["global_component_scores"].items()
                )
                prompt_result["global_n_normal_clips"] = int(global_eval.get("n_normal_clips", 0))
                prompt_result["global_n_transition_clips"] = int(global_eval.get("n_transition_clips", 0))

        for model_type in model_order:
            prompt_result = prompt_results[model_type]
            parts = [f"prompt {prompt_idx}", f"fork={prompt_bundle['fork_abs']}"]
            if "local_component_scores" in prompt_result:
                local_str = " ".join(
                    f"{k}={v:.4f}" for k, v in prompt_result["local_component_scores"].items()
                )
                local_total = prompt_result.get("local_total")
                if local_total is not None:
                    parts.append(f"local_total={local_total:.4f}")
                parts.append(f"local[{local_str}]")
            if "global_component_scores" in prompt_result:
                global_str = " ".join(
                    f"{k}={v:.4f}" for k, v in prompt_result["global_component_scores"].items()
                )
                global_total = prompt_result.get("global_total")
                if global_total is not None:
                    parts.append(f"global_total={global_total:.4f}")
                parts.append(f"global[{global_str}]")
            print(f"  [EVAL:{model_type}][Rank {self.rank}] " + " ".join(parts))

        return prompt_results

    def _evaluate_both_same_rank(
        self,
        eval_items: List[Dict[str, Any]],
        eval_seed: int,
        eval_reward_components: List[Dict[str, Any]],
        eval_component_weights: Dict[str, float],
    ) -> List[Dict[str, Any]]:
        model_specs: List[Tuple[str, bool]] = [("online", False)]
        if self.ema is not None:
            model_specs.append(("ema", True))

        local_results_by_model: Dict[str, List[Dict[str, Any]]] = {
            model_type: [] for model_type, _ in model_specs
        }
        if not eval_items:
            return []

        active_ranks = min(self.world_size, len(eval_items))
        prompt_indices = []
        if self.rank < active_ranks:
            prompt_indices = list(range(self.rank, len(eval_items), active_ranks))

        ema_context = None
        generator_was_training = self.generator.training
        self.generator.eval()
        try:
            if any(use_ema for _, use_ema in model_specs) and prompt_indices:
                ema_context = self._create_eval_ema_context()

            for prompt_idx in prompt_indices:
                prompt_results = self._evaluate_prompt_pair_same_rank(
                    eval_items=eval_items,
                    prompt_idx=prompt_idx,
                    eval_seed=eval_seed,
                    eval_reward_components=eval_reward_components,
                    eval_component_weights=eval_component_weights,
                    ema_context=ema_context,
                )
                for model_type, result in prompt_results.items():
                    local_results_by_model[model_type].append(result)
        finally:
            self._destroy_eval_ema_context(ema_context)
            if generator_was_training:
                self.generator.train()

        if dist.is_initialized():
            gathered: List[Optional[Dict[str, List[Dict[str, Any]]]]] = [None] * self.world_size
            dist.all_gather_object(gathered, local_results_by_model)
            all_results_by_model: Dict[str, List[Dict[str, Any]]] = {
                model_type: [] for model_type, _ in model_specs
            }
            for rank_result in gathered:
                if not rank_result:
                    continue
                for model_type, results in rank_result.items():
                    all_results_by_model[model_type].extend(results)
        else:
            all_results_by_model = local_results_by_model

        for results in all_results_by_model.values():
            results.sort(key=lambda x: x["prompt_idx"])

        if not self.is_main:
            return []

        return [
            self._summarize_eval_results(
                all_results=all_results_by_model[model_type],
                model_type=model_type,
                eval_reward_components=eval_reward_components,
            )
            for model_type, _ in model_specs
            if all_results_by_model[model_type]
        ]

    def _evaluate_prompt_indices(
        self,
        eval_items: List[Dict[str, Any]],
        prompt_indices: Sequence[int],
        eval_seed: int,
        use_ema: bool,
        eval_reward_components: List[Dict[str, Any]],
        eval_component_weights: Dict[str, float],
    ) -> List[Dict[str, Any]]:
        cfg = self.config
        device, dtype = self.device, self.dtype
        chunk_size = int(cfg.streaming_chunk_size)
        max_length = int(cfg.streaming_max_length)
        min_new_frame = int(cfg.streaming_min_new_frame)
        global_sink = bool(getattr(cfg, "global_sink", True))
        model_type = "ema" if use_ema else "online"

        ema_backup = None
        if use_ema:
            if self.ema is None:
                raise ValueError("use_ema=True requires EMA to be enabled")
            ema_backup = {}
            for n, p in self.generator.named_parameters():
                if n in self.ema.shadow:
                    ema_backup[n] = p.data.clone()
            self.ema.copy_to(self.generator)

        self.generator.eval()
        pipeline_gen_backup = self.pipeline.generator
        self.pipeline.generator = self.generator

        local_results: List[Dict[str, Any]] = []

        try:
            for ei in sorted(int(idx) for idx in prompt_indices):
                item = eval_items[ei]
                prompts_list = item["prompts"]
                noise_seed = eval_seed + ei * 10000
                saved_cpu_rng = torch.random.get_rng_state()
                saved_cuda_rng = None
                if device.type == "cuda":
                    saved_cuda_rng = torch.cuda.get_rng_state(device=device)
                    seeded_cuda_rng = torch.Generator(device=device)
                    seeded_cuda_rng.manual_seed(noise_seed)
                    torch.cuda.set_rng_state(seeded_cuda_rng.get_state(), device=device)
                torch.manual_seed(noise_seed)

                cond_list = []
                for p in prompts_list:
                    cd = self.text_encoder(text_prompts=[p])
                    cd = {
                        k: v.to(device=device, dtype=dtype) if isinstance(v, torch.Tensor) else v
                        for k, v in cd.items()
                    }
                    cond_list.append(cd)

                plan = self.pipeline.plan_perturbation(
                    total_frames=max_length, chunk_size=chunk_size,
                    min_new_frame=min_new_frame, K=1, base_seed=eval_seed,
                    sample_idx=ei, seed_context=0,
                    perturb_within_first_x_chunks=1,
                    perturb_num_blocks=1,
                    sink_size=int(cfg.model_kwargs.get("sink_size", 3)),
                )
                chunks = plan["chunks"]
                switch_raw = getattr(cfg, "switch_frame_indices", "57,93,129,165,201")
                switch_indices = self._resolve_switch_frame_indices(
                    switch_raw, chunks=chunks, num_segments=len(prompts_list),
                )
                perturb_idx = int(plan["perturb_chunk_idx"])
                fork_abs = int(plan["perturb_block_abs_frame"])
                fork_chunk = chunks[perturb_idx]
                local_chunk_start = int(fork_chunk["start_frame"])
                local_chunk_end = local_chunk_start + int(fork_chunk["new_frames"])

                batch_size = 1
                self.pipeline._initialize_kv_cache(batch_size, dtype, device)
                self.pipeline._initialize_crossattn_cache(batch_size, dtype, device)
                self.pipeline.clear_kv_cache()
                self.pipeline.generator.model.local_attn_size = int(self.pipeline.local_attn_size)
                self.pipeline._set_all_modules_max_attention_size(int(self.pipeline.local_attn_size))

                rng = torch.Generator(device=device)
                rng.manual_seed(noise_seed)

                def _seg_for_frame(frame_idx: int, _si=switch_indices) -> int:
                    seg = 0
                    for si in _si:
                        if frame_idx >= si:
                            seg += 1
                        else:
                            break
                    return seg

                try:
                    latents = []
                    cur_len = 0
                    seg_idx = 0
                    for ch in chunks:
                        nf = ch["new_frames"]
                        if nf <= 0:
                            continue
                        new_seg = _seg_for_frame(cur_len)
                        if new_seg != seg_idx:
                            self.pipeline.recache_after_switch(
                                latents, cur_len, cond_list[new_seg], global_sink,
                            )
                            seg_idx = new_seg
                        noise = torch.randn(
                            [1, nf, 16, 60, 104], generator=rng, device=device, dtype=dtype,
                        )
                        out = self.pipeline.generate_chunk_sampling(
                            noise=noise,
                            conditional_dict=cond_list[seg_idx],
                            current_start_frame=cur_len,
                        )
                        latents.append(out.cpu())
                        cur_len += nf
                finally:
                    torch.random.set_rng_state(saved_cpu_rng)
                    if saved_cuda_rng is not None:
                        torch.cuda.set_rng_state(saved_cuda_rng, device=device)

                full_latent = torch.cat(latents, dim=1)

                prompt_result: Dict[str, Any] = {
                    "prompt_idx": ei,
                    "fork_abs_frame": fork_abs,
                    "global_start_abs_frame": 0,
                    "local_chunk_latent_start": local_chunk_start,
                    "local_chunk_latent_end": local_chunk_end,
                }

                if self.train_local:
                    local_chunk = full_latent[:, local_chunk_start:local_chunk_end].cpu()
                    fork_seg = self._get_segment_for_frame(fork_abs, switch_indices)
                    local_prompt = prompts_list[min(fork_seg, len(prompts_list) - 1)] if prompts_list else ""
                    local_eval, _ = self._compute_rewards_short(
                        [local_chunk],
                        local_prompt,
                        ei,
                        include_aggregate=False,
                        reward_components=eval_reward_components,
                    )
                    prompt_result["local_prompt"] = local_prompt
                    prompt_result["local_component_scores"] = {
                        name: float(comp["scores"][0])
                        for name, comp in local_eval["components"].items()
                    }
                    prompt_result["local_total"] = sum(
                        eval_component_weights.get(name, 1.0) * score
                        for name, score in prompt_result["local_component_scores"].items()
                    )

                if self.train_global:
                    global_eval, _ = self._compute_rewards_long(
                        [full_latent.cpu()], prompts_list, switch_indices,
                        local_chunk_end,
                        ei,
                        include_aggregate=False,
                        reward_components=eval_reward_components,
                    )
                    prompt_result["global_component_scores"] = {
                        name: float(comp["scores"][0])
                        for name, comp in global_eval["components"].items()
                    }
                    prompt_result["global_total"] = sum(
                        eval_component_weights.get(name, 1.0) * score
                        for name, score in prompt_result["global_component_scores"].items()
                    )
                    prompt_result["global_n_normal_clips"] = int(global_eval.get("n_normal_clips", 0))
                    prompt_result["global_n_transition_clips"] = int(global_eval.get("n_transition_clips", 0))

                local_results.append(prompt_result)

                parts = [f"prompt {ei}", f"fork={fork_abs}"]
                if "local_component_scores" in prompt_result:
                    local_str = " ".join(
                        f"{k}={v:.4f}" for k, v in prompt_result["local_component_scores"].items()
                    )
                    local_total = prompt_result.get("local_total")
                    if local_total is not None:
                        parts.append(f"local_total={local_total:.4f}")
                    parts.append(f"local[{local_str}]")
                if "global_component_scores" in prompt_result:
                    global_str = " ".join(
                        f"{k}={v:.4f}" for k, v in prompt_result["global_component_scores"].items()
                    )
                    global_total = prompt_result.get("global_total")
                    if global_total is not None:
                        parts.append(f"global_total={global_total:.4f}")
                    parts.append(f"global[{global_str}]")
                print(f"  [EVAL:{model_type}][Rank {self.rank}] " + " ".join(parts))
        finally:
            self.pipeline.generator = pipeline_gen_backup
            if ema_backup is not None:
                for n, p in self.generator.named_parameters():
                    if n in ema_backup:
                        p.data.copy_(ema_backup[n])
            self.generator.train()

        return local_results

    def _summarize_eval_results(
        self,
        all_results: List[Dict[str, Any]],
        model_type: str,
        eval_reward_components: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        def _mean_component_scores(score_key: str) -> Dict[str, float]:
            result: Dict[str, float] = {}
            for comp in eval_reward_components:
                k = comp["key"]
                vals = [
                    r[score_key][k]
                    for r in all_results
                    if score_key in r and k in r[score_key]
                ]
                if vals:
                    result[k] = sum(vals) / len(vals)
            return result

        mean_local_components = _mean_component_scores("local_component_scores")
        mean_global_components = _mean_component_scores("global_component_scores")
        mean_local_total = None
        mean_global_total = None
        local_totals = [r["local_total"] for r in all_results if "local_total" in r]
        global_totals = [r["global_total"] for r in all_results if "global_total" in r]
        if local_totals:
            mean_local_total = sum(local_totals) / len(local_totals)
        if global_totals:
            mean_global_total = sum(global_totals) / len(global_totals)

        summary = {
            "eval_type": "reward_eval",
            "model_type": model_type,
            "scoring_mode": "train_aligned_no_advantage",
            "step": self.step,
            "n_prompts": len(all_results),
            "eval_component_keys": [comp["key"] for comp in eval_reward_components],
            "mean_local_total": mean_local_total,
            "mean_global_total": mean_global_total,
            "mean_local_components": mean_local_components,
            "mean_global_components": mean_global_components,
            "per_prompt": all_results,
        }
        parts = [f"[EVAL:{model_type}] MEAN ({len(all_results)} prompts):"]
        if mean_local_total is not None:
            parts.append(f"local_total={mean_local_total:.4f}")
        if mean_global_total is not None:
            parts.append(f"global_total={mean_global_total:.4f}")
        if mean_local_components:
            parts.append(
                "local[" + " ".join(f"{k}={v:.4f}" for k, v in mean_local_components.items()) + "]"
            )
        if mean_global_components:
            parts.append(
                "global[" + " ".join(f"{k}={v:.4f}" for k, v in mean_global_components.items()) + "]"
            )
        print("  " + " ".join(parts))
        return summary

    def evaluate(
        self,
        eval_items: List[Dict[str, Any]],
        eval_seed: int = 99999,
        use_ema: bool = False,
    ) -> Dict[str, Any]:
        """Generate one unperturbed video per eval prompt and compute rewards.

        Each rank generates and scores its own subset of eval prompts in
        parallel. Rank 0 gathers all results and returns the summary;
        other ranks return an empty dict.
        """
        eval_reward_components, eval_component_weights = self._get_eval_reward_setup()
        model_type = "ema" if use_ema else "online"
        my_indices = list(range(self.rank, len(eval_items), self.world_size))
        local_results = self._evaluate_prompt_indices(
            eval_items=eval_items,
            prompt_indices=my_indices,
            eval_seed=eval_seed,
            use_ema=use_ema,
            eval_reward_components=eval_reward_components,
            eval_component_weights=eval_component_weights,
        )

        if dist.is_initialized():
            gathered: List[Optional[List[Dict[str, Any]]]] = [None] * self.world_size
            dist.all_gather_object(gathered, local_results)
            all_results = [r for rank_list in gathered if rank_list for r in rank_list]
            all_results.sort(key=lambda x: x["prompt_idx"])
        else:
            all_results = local_results

        if not self.is_main:
            return {}

        return self._summarize_eval_results(all_results, model_type, eval_reward_components)

    def evaluate_both(
        self,
        eval_items: List[Dict[str, Any]],
        eval_seed: int = 99999,
    ) -> List[Dict[str, Any]]:
        """Evaluate online and EMA policies in one distributed pass.

        Supports two modes:
        - ``task_sharded``: assign tasks over the Cartesian product of
          {online, ema} x eval prompts across all ranks.
        - ``same_rank``: shard prompts across ranks, and for each active rank
          evaluate online + EMA for the same prompt pairwise.
        """
        eval_reward_components, eval_component_weights = self._get_eval_reward_setup()
        eval_mode = str(
            getattr(self.config, "eval_parallel_online_ema_mode", "task_sharded")
        ).strip().lower()
        if eval_mode == "same_rank":
            return self._evaluate_both_same_rank(
                eval_items=eval_items,
                eval_seed=eval_seed,
                eval_reward_components=eval_reward_components,
                eval_component_weights=eval_component_weights,
            )
        if eval_mode != "task_sharded":
            raise ValueError(
                "eval_parallel_online_ema_mode must be 'task_sharded' or 'same_rank' "
                f"(got {eval_mode!r})"
            )
        model_specs: List[Tuple[str, bool]] = [("online", False)]
        if self.ema is not None:
            model_specs.append(("ema", True))

        task_specs: List[Tuple[str, bool, int]] = []
        for model_type, use_ema in model_specs:
            for prompt_idx in range(len(eval_items)):
                task_specs.append((model_type, use_ema, prompt_idx))

        local_task_specs = [task_specs[i] for i in range(self.rank, len(task_specs), self.world_size)]
        local_results_by_model: Dict[str, List[Dict[str, Any]]] = {
            model_type: [] for model_type, _ in model_specs
        }

        for model_type, use_ema in model_specs:
            prompt_indices = [
                prompt_idx for task_model_type, task_use_ema, prompt_idx in local_task_specs
                if task_model_type == model_type and task_use_ema == use_ema
            ]
            if not prompt_indices:
                continue
            local_results_by_model[model_type].extend(
                self._evaluate_prompt_indices(
                    eval_items=eval_items,
                    prompt_indices=prompt_indices,
                    eval_seed=eval_seed,
                    use_ema=use_ema,
                    eval_reward_components=eval_reward_components,
                    eval_component_weights=eval_component_weights,
                )
            )

        if dist.is_initialized():
            gathered: List[Optional[Dict[str, List[Dict[str, Any]]]]] = [None] * self.world_size
            dist.all_gather_object(gathered, local_results_by_model)
            all_results_by_model: Dict[str, List[Dict[str, Any]]] = {
                model_type: [] for model_type, _ in model_specs
            }
            for rank_result in gathered:
                if not rank_result:
                    continue
                for model_type, results in rank_result.items():
                    all_results_by_model[model_type].extend(results)
        else:
            all_results_by_model = local_results_by_model

        for results in all_results_by_model.values():
            results.sort(key=lambda x: x["prompt_idx"])

        if not self.is_main:
            return []

        return [
            self._summarize_eval_results(
                all_results=all_results_by_model[model_type],
                model_type=model_type,
                eval_reward_components=eval_reward_components,
            )
            for model_type, _ in model_specs
            if all_results_by_model[model_type]
        ]

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str, extra_state: Optional[Dict[str, Any]] = None):
        if not self.is_main:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state = {
            "step": self.step,
            "_accum_count": self._accum_count,
            "samples_seen": self.samples_seen,
            "rollout_count": self.rollout_count,
        }
        if extra_state:
            state.update(extra_state)
        model = self.generator.model
        if hasattr(model, 'base_model'):
            import peft
            state["generator_lora"] = peft.get_peft_model_state_dict(model)
        else:
            state["generator"] = model.state_dict()
        state["optimizer"] = self.optimizer.state_dict()
        state["lr_scheduler"] = self.lr_scheduler.state_dict()
        if self.ema is not None:
            state["ema"] = self.ema.state_dict()
        if self.initial_generator is not None:
            im = self.initial_generator.model
            if hasattr(im, "base_model"):
                import peft
                state["initial_generator_lora"] = peft.get_peft_model_state_dict(im)
            else:
                state["initial_generator"] = im.state_dict()
        torch.save(state, path)
        print(f"[KVPO] Checkpoint saved: {path}")

    def load_checkpoint(self, path: str):
        """Resume training from a saved checkpoint."""
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self.step = ckpt.get("step", 0)
        self._accum_count = int(ckpt.get("_accum_count", 0))
        self.samples_seen = int(ckpt.get("samples_seen", 0))
        self.rollout_count = int(ckpt.get("rollout_count", 0))
        if self._accum_count < 0 or self._accum_count >= self.gradient_accumulation_steps:
            if self.is_main:
                print(
                    f"[KVPO] Invalid _accum_count={self._accum_count} in checkpoint; "
                    "resetting to 0"
                )
            self._accum_count = 0
        if self.rollout_count < 0:
            self.rollout_count = 0
        if self.samples_seen < 0:
            self.samples_seen = 0
        if self.rollout_count == 0 and (self.step > 0 or self._accum_count > 0):
            self.rollout_count = self.step * self.gradient_accumulation_steps + self._accum_count
        if self.samples_seen == 0 and self.rollout_count > 0:
            self.samples_seen = self.rollout_count * self.world_size

        model = self.generator.model
        if "generator_lora" in ckpt and hasattr(model, 'base_model'):
            import peft
            peft.set_peft_model_state_dict(model, ckpt["generator_lora"])
        elif "generator" in ckpt:
            model.load_state_dict(ckpt["generator"])

        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)

        if "lr_scheduler" in ckpt:
            self.lr_scheduler.load_state_dict(ckpt["lr_scheduler"])

        if "ema" in ckpt and self.ema is not None:
            self.ema.load_state_dict(ckpt["ema"])

        if self.initial_generator is not None:
            im = self.initial_generator.model
            if "initial_generator_lora" in ckpt and hasattr(im, "base_model"):
                import peft
                peft.set_peft_model_state_dict(im, ckpt["initial_generator_lora"])
            elif "initial_generator" in ckpt:
                im.load_state_dict(ckpt["initial_generator"])
            else:
                self.initial_generator.load_state_dict(self.generator.state_dict(), strict=True)
            self.initial_generator.requires_grad_(False)
            self.initial_generator.eval()

        self._sync_old_policy()

        if self.is_main:
            print(
                f"[KVPO] Resumed from {path}, step={self.step}, "
                f"_accum_count={self._accum_count}, "
                f"samples_seen={self.samples_seen}, rollout_count={self.rollout_count}"
            )

    def save_ema_checkpoint(self, path: str, extra_state: Optional[Dict[str, Any]] = None):
        """Save EMA weights as a standalone LoRA/model checkpoint for inference."""
        if not self.is_main or self.ema is None:
            return
        backup = {}
        for n, p in self.generator.named_parameters():
            if n in self.ema.shadow:
                backup[n] = p.data.clone()
        self.ema.copy_to(self.generator)
        self.save_checkpoint(path, extra_state=extra_state)
        for n, p in self.generator.named_parameters():
            if n in backup:
                p.data.copy_(backup[n])
        print(f"[KVPO] EMA checkpoint saved: {path}")
