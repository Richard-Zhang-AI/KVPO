# Adopted from https://github.com/guandeh17/Self-Forcing
# SPDX-License-Identifier: CC-BY-NC-SA-4.0
import os
import types
from pathlib import Path
from typing import List, Optional
import torch
from torch import nn

from models.longlive.utils.scheduler import SchedulerInterface, FlowMatchScheduler
from models.longlive.wan.modules.tokenizers import HuggingfaceTokenizer
from models.longlive.wan.modules.model import WanModel, RegisterTokens, GanAttentionBlock
from models.longlive.wan.modules.vae import _video_vae
from models.longlive.wan.modules.t5 import umt5_xxl
from models.longlive.wan.modules.causal_model import CausalWanModel
from models.longlive.wan.modules.causal_model_infinity import CausalWanModel as CausalWanModelInfinity

# KVPO repo root: .../KVPO/models/longlive/utils/wan_wrapper.py -> parents[3] == KVPO
_KVPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_WAN_MODEL_NAME = "Wan2.1-T2V-1.3B"


def _resolve_wan_checkpoint_dir(model_name: str) -> str:
    explicit = os.environ.get("WAN_MODEL_DIR", "").strip()
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if (p / "config.json").is_file():
            return str(p)

    candidates: list[Path] = []
    root = os.environ.get("WAN_MODEL_ROOT", "").strip()
    if root:
        r = Path(root).expanduser()
        if not r.is_absolute():
            r = (_KVPO_ROOT / r).resolve()
        else:
            r = r.resolve()
        candidates.append(r / model_name)
        if (r / "config.json").is_file():
            candidates.append(r)

    candidates.append((_KVPO_ROOT / "wan_models" / model_name).resolve())
    candidates.append((Path.cwd() / "wan_models" / model_name).resolve())

    seen: set[str] = set()
    for c in candidates:
        key = str(c)
        if key in seen:
            continue
        seen.add(key)
        if (c / "config.json").is_file():
            return str(c)

    msg = (
        f"Wan checkpoint not found for model_name={model_name!r} "
        f"(need config.json). Tried:\n  "
        + "\n  ".join(str(x) for x in candidates)
        + f"\nExpected default: {_KVPO_ROOT / 'wan_models' / model_name}\n"
        "Set WAN_MODEL_DIR to the directory containing config.json, or "
        "WAN_MODEL_ROOT to the parent of model folders (e.g. .../wan_models)."
    )
    raise FileNotFoundError(msg)


def _wan_dir_for_encoder() -> Path:
    return Path(_resolve_wan_checkpoint_dir(_DEFAULT_WAN_MODEL_NAME))


class WanTextEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.text_encoder = umt5_xxl(
            encoder_only=True,
            return_tokenizer=False,
            dtype=torch.float32,
            device=torch.device('cpu')
        ).eval().requires_grad_(False)
        _wan = _wan_dir_for_encoder()
        t5_path = _wan / "models_t5_umt5-xxl-enc-bf16.pth"
        tok_path = _wan / "google" / "umt5-xxl"
        if not t5_path.is_file():
            raise FileNotFoundError(f"T5 weights not found: {t5_path}")
        if not tok_path.is_dir():
            raise FileNotFoundError(f"T5 tokenizer dir not found: {tok_path}")
        self.text_encoder.load_state_dict(
            torch.load(str(t5_path), map_location='cpu', weights_only=False)
        )
        
        # Move text encoder to GPU if available
        if torch.cuda.is_available():
            self.text_encoder = self.text_encoder.cuda()

        self.tokenizer = HuggingfaceTokenizer(
            name=str(tok_path), seq_len=512, clean='whitespace')

    @property
    def device(self):
        # Assume we are always on GPU
        return torch.cuda.current_device()

    def forward(self, text_prompts: List[str]) -> dict:
        ids, mask = self.tokenizer(
            text_prompts, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        context = self.text_encoder(ids, mask)
        # ids = ids.to(torch.device('cpu'))
        # mask = mask.to(torch.device('cpu'))
        for u, v in zip(context, seq_lens):
            u[v:] = 0.0  # set padding to 0.0

        return {
            "prompt_embeds": context
        }


class WanVAEWrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        mean = [
            -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
            0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
        ]
        std = [
            2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
            3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
        ]
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)

        _wan = _wan_dir_for_encoder()
        vae_path = _wan / "Wan2.1_VAE.pth"
        if not vae_path.is_file():
            raise FileNotFoundError(f"VAE weights not found: {vae_path}")
        self.model = _video_vae(
            pretrained_path=str(vae_path),
            z_dim=16,
        ).eval().requires_grad_(False)

    def encode_to_latent(self, pixel: torch.Tensor) -> torch.Tensor:
        # pixel: [batch_size, num_channels, num_frames, height, width]
        device, dtype = pixel.device, pixel.dtype
        scale = [self.mean.to(device=device, dtype=dtype),
                 1.0 / self.std.to(device=device, dtype=dtype)]

        output = [
            self.model.encode(u.unsqueeze(0), scale).float().squeeze(0)
            for u in pixel
        ]
        output = torch.stack(output, dim=0)
        # from [batch_size, num_channels, num_frames, height, width]
        # to [batch_size, num_frames, num_channels, height, width]
        output = output.permute(0, 2, 1, 3, 4)
        return output

    def decode_to_pixel(self, latent: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
        zs = latent.permute(0, 2, 1, 3, 4)
        if use_cache:
            assert latent.shape[0] == 1, "Batch size must be 1 when using cache"

        device, dtype = latent.device, latent.dtype
        scale = [self.mean.to(device=device, dtype=dtype),
                 1.0 / self.std.to(device=device, dtype=dtype)]

        if use_cache:
            decode_function = self.model.cached_decode
        else:
            decode_function = self.model.decode

        output = []
        for u in zs:
            output.append(decode_function(u.unsqueeze(0), scale).float().clamp_(-1, 1).squeeze(0))
        output = torch.stack(output, dim=0)
        # from [batch_size, num_channels, num_frames, height, width]
        # to [batch_size, num_frames, num_channels, height, width]
        output = output.permute(0, 2, 1, 3, 4)
        return output

    def decode_to_pixel_chunk(self, latent: torch.Tensor, use_cache: bool = False, chunk_size: int = 120) -> torch.Tensor:
        zs = latent.permute(0, 2, 1, 3, 4)
        if use_cache:
            assert latent.shape[0] == 1, "Batch size must be 1 when using cache"

        device, dtype = latent.device, latent.dtype
        scale = [self.mean.to(device=device, dtype=dtype),
                 1.0 / self.std.to(device=device, dtype=dtype)]

        decode_function = self.model.cached_decode if use_cache else self.model.decode

        output = []
        for u in zs:
            num_frames = u.shape[1]
            if num_frames <= chunk_size:
                decoded = decode_function(u.unsqueeze(0), scale).float().clamp_(-1, 1).squeeze(0)
                decoded = decoded.cpu()
            else:
                decoded_chunks = []
                for start_idx in range(0, num_frames, chunk_size):
                    end_idx = min(start_idx + chunk_size, num_frames)
                    chunk = u[:, start_idx:end_idx, :, :]
                    self.model.clear_cache()
                    decoded_chunk = decode_function(chunk.unsqueeze(0), scale).float().clamp_(-1, 1).squeeze(0)
                    decoded_chunks.append(decoded_chunk.cpu())
                    del decoded_chunk
                    torch.cuda.empty_cache()
                decoded = torch.cat(decoded_chunks, dim=1)
                self.model.clear_cache()
            output.append(decoded)

        output = torch.stack(output, dim=0)
        output = output.permute(0, 2, 1, 3, 4)
        return output


class WanDiffusionWrapper(torch.nn.Module):
    def __init__(
            self,
            model_name="Wan2.1-T2V-1.3B",
            timestep_shift=8.0,
            is_causal=False,
            local_attn_size=-1,
            sink_size=0,
            use_infinite_attention=False,
    ):
        super().__init__()
        self.use_infinite_attention = bool(use_infinite_attention)

        pretrained_dir = _resolve_wan_checkpoint_dir(model_name)
        if is_causal:
            if self.use_infinite_attention:
                self.model = CausalWanModelInfinity.from_pretrained(
                    pretrained_dir, local_attn_size=local_attn_size, sink_size=sink_size)
            else:
                self.model = CausalWanModel.from_pretrained(
                    pretrained_dir,
                    local_attn_size=local_attn_size,
                    sink_size=sink_size,
                )
        else:
            self.model = WanModel.from_pretrained(pretrained_dir)
        self.model.eval()

        # For non-causal diffusion, all frames share the same timestep
        self.uniform_timestep = not is_causal

        self.scheduler = FlowMatchScheduler(
            shift=timestep_shift, sigma_min=0.0, extra_one_step=True
        )
        self.scheduler.set_timesteps(1000, training=True)

        # self.seq_len = 1560 * local_attn_size if local_attn_size != -1 else 32760 # [1, 21, 16, 60, 104]
        self.seq_len = 1560 * local_attn_size if local_attn_size > 21 else 32760 # [1, 21, 16, 60, 104]
        self.post_init()

    def enable_gradient_checkpointing(self) -> None:
        self.model.enable_gradient_checkpointing()

    def adding_cls_branch(self, atten_dim=1536, num_class=4, time_embed_dim=0) -> None:
        # NOTE: This is hard coded for WAN2.1-T2V-1.3B for now!!!!!!!!!!!!!!!!!!!!
        self._cls_pred_branch = nn.Sequential(
            # Input: [B, 384, 21, 60, 104]
            nn.LayerNorm(atten_dim * 3 + time_embed_dim),
            nn.Linear(atten_dim * 3 + time_embed_dim, 1536),
            nn.SiLU(),
            nn.Linear(atten_dim, num_class)
        )
        self._cls_pred_branch.requires_grad_(True)
        num_registers = 3
        self._register_tokens = RegisterTokens(num_registers=num_registers, dim=atten_dim)
        self._register_tokens.requires_grad_(True)

        gan_ca_blocks = []
        for _ in range(num_registers):
            block = GanAttentionBlock()
            gan_ca_blocks.append(block)
        self._gan_ca_blocks = nn.ModuleList(gan_ca_blocks)
        self._gan_ca_blocks.requires_grad_(True)
        # self.has_cls_branch = True

    def _convert_flow_pred_to_x0(self, flow_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        Convert flow matching's prediction to x0 prediction.
        flow_pred: the prediction with shape [B, C, H, W]
        xt: the input noisy data with shape [B, C, H, W]
        timestep: the timestep with shape [B]

        pred = noise - x0
        x_t = (1-sigma_t) * x0 + sigma_t * noise
        we have x0 = x_t - sigma_t * pred
        see derivations https://chatgpt.com/share/67bf8589-3d04-8008-bc6e-4cf1a24e2d0e
        """
        # use higher precision for calculations
        original_dtype = flow_pred.dtype
        flow_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(flow_pred.device), [flow_pred, xt,
                                                        self.scheduler.sigmas,
                                                        self.scheduler.timesteps]
        )

        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        x0_pred = xt - sigma_t * flow_pred
        return x0_pred.to(original_dtype)

    @staticmethod
    def _convert_x0_to_flow_pred(scheduler, x0_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        Convert x0 prediction to flow matching's prediction.
        x0_pred: the x0 prediction with shape [B, C, H, W]
        xt: the input noisy data with shape [B, C, H, W]
        timestep: the timestep with shape [B]

        pred = (x_t - x_0) / sigma_t
        """
        # use higher precision for calculations
        original_dtype = x0_pred.dtype
        x0_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(x0_pred.device), [x0_pred, xt,
                                                      scheduler.sigmas,
                                                      scheduler.timesteps]
        )
        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        flow_pred = (xt - x0_pred) / sigma_t
        return flow_pred.to(original_dtype)

    def forward(
        self,
        noisy_image_or_video: torch.Tensor, conditional_dict: dict,
        timestep: torch.Tensor, kv_cache: Optional[List[dict]] = None,
        crossattn_cache: Optional[List[dict]] = None,
        current_start: Optional[int] = None,
        classify_mode: Optional[bool] = False,
        concat_time_embeddings: Optional[bool] = False,
        clean_x: Optional[torch.Tensor] = None,
        aug_t: Optional[torch.Tensor] = None,
        cache_start: Optional[int] = None,
        sink_recache_after_switch: Optional[bool] = False,
        update_cache: Optional[bool] = True,
    ) -> torch.Tensor:
        prompt_embeds = conditional_dict["prompt_embeds"]

        # [B, F] -> [B]
        if self.uniform_timestep:
            input_timestep = timestep[:, 0]
        else:
            input_timestep = timestep

        logits = None
        # X0 prediction
        if kv_cache is not None:
            if self.use_infinite_attention:
                flow_pred = self.model(
                    noisy_image_or_video.permute(0, 2, 1, 3, 4),
                    t=input_timestep, context=prompt_embeds,
                    seq_len=self.seq_len,
                    kv_cache=kv_cache,
                    crossattn_cache=crossattn_cache,
                    current_start=current_start,
                    cache_start=cache_start,
                    sink_recache_after_switch=sink_recache_after_switch,
                    update_cache=update_cache,
                ).permute(0, 2, 1, 3, 4)
            else:
                flow_pred = self.model(
                    noisy_image_or_video.permute(0, 2, 1, 3, 4),
                    t=input_timestep, context=prompt_embeds,
                    seq_len=self.seq_len,
                    kv_cache=kv_cache,
                    crossattn_cache=crossattn_cache,
                    current_start=current_start,
                    cache_start=cache_start,
                    sink_recache_after_switch=sink_recache_after_switch,
                    update_cache=update_cache,
                ).permute(0, 2, 1, 3, 4)
        else:
            if clean_x is not None:
                # teacher forcing
                model_kwargs = dict(
                    t=input_timestep,
                    context=prompt_embeds,
                    seq_len=self.seq_len,
                    clean_x=clean_x.permute(0, 2, 1, 3, 4),
                    aug_t=aug_t,
                )
                if hasattr(self.model, "_forward_inference"):
                    model_kwargs["sink_recache_after_switch"] = sink_recache_after_switch
                flow_pred = self.model(
                    noisy_image_or_video.permute(0, 2, 1, 3, 4),
                    **model_kwargs,
                ).permute(0, 2, 1, 3, 4)
            else:
                if classify_mode:
                    model_kwargs = dict(
                        t=input_timestep,
                        context=prompt_embeds,
                        seq_len=self.seq_len,
                        classify_mode=True,
                        register_tokens=self._register_tokens,
                        cls_pred_branch=self._cls_pred_branch,
                        gan_ca_blocks=self._gan_ca_blocks,
                        concat_time_embeddings=concat_time_embeddings,
                    )
                    if hasattr(self.model, "_forward_inference"):
                        model_kwargs["sink_recache_after_switch"] = sink_recache_after_switch
                    flow_pred, logits = self.model(
                        noisy_image_or_video.permute(0, 2, 1, 3, 4),
                        **model_kwargs,
                    )
                    flow_pred = flow_pred.permute(0, 2, 1, 3, 4)
                else:
                    model_kwargs = dict(
                        t=input_timestep,
                        context=prompt_embeds,
                        seq_len=self.seq_len,
                    )
                    if hasattr(self.model, "_forward_inference"):
                        model_kwargs["sink_recache_after_switch"] = sink_recache_after_switch
                    flow_pred = self.model(
                        noisy_image_or_video.permute(0, 2, 1, 3, 4),
                        **model_kwargs,
                    ).permute(0, 2, 1, 3, 4)

        pred_x0 = self._convert_flow_pred_to_x0(
            flow_pred=flow_pred.flatten(0, 1),
            xt=noisy_image_or_video.flatten(0, 1),
            timestep=timestep.flatten(0, 1)
        ).unflatten(0, flow_pred.shape[:2])

        if logits is not None:
            return flow_pred, pred_x0, logits

        return flow_pred, pred_x0

    def get_scheduler(self) -> SchedulerInterface:
        """
        Update the current scheduler with the interface's static method
        """
        scheduler = self.scheduler
        scheduler.convert_x0_to_noise = types.MethodType(
            SchedulerInterface.convert_x0_to_noise, scheduler)
        scheduler.convert_noise_to_x0 = types.MethodType(
            SchedulerInterface.convert_noise_to_x0, scheduler)
        scheduler.convert_velocity_to_x0 = types.MethodType(
            SchedulerInterface.convert_velocity_to_x0, scheduler)
        self.scheduler = scheduler
        return scheduler

    def post_init(self):
        """
        A few custom initialization steps that should be called after the object is created.
        Currently, the only one we have is to bind a few methods to scheduler.
        We can gradually add more methods here if needed.
        """
        self.get_scheduler()
