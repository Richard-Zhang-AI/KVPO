from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn
from easydict import EasyDict as edict

from ... import CKPT_PATH, SEA_RAFT_CKPT_PATH

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SEA_RAFT_CORE_PATH = os.path.abspath(
    os.path.join(CURRENT_DIR, "..", "..", "reward_models", "sea_raft", "core")
)
if SEA_RAFT_CORE_PATH not in sys.path:
    sys.path.insert(0, SEA_RAFT_CORE_PATH)

from raft import RAFT
from raft_utils.utils import InputPadder, load_ckpt


class OpticalFlowSmoothnessScorer(nn.Module):
    """Reward smoother, more coherent motion fields.

    This scorer intentionally does not reward motion magnitude. It only penalizes
    noisy, broken, or temporally inconsistent optical-flow fields.
    """

    def __init__(
        self,
        device,
        model_path=None,
        dtype=torch.float32,
        iters: int = 4,
        motion_threshold: float = 0.02,
        spatial_weight: float = 0.6,
        temporal_weight: float = 0.4,
        reward_scale: float = 8.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.motion_threshold = float(motion_threshold)
        self.spatial_weight = float(spatial_weight)
        self.temporal_weight = float(temporal_weight)
        self.reward_scale = float(reward_scale)
        self.eps = float(eps)

        args = edict(
            {
                "dim": 128,
                "radius": 4,
                "use_var": True,
                "var_min": 0,
                "var_max": 10,
                "scale": 0,
                "model": SEA_RAFT_CKPT_PATH,
                "initial_dim": 64,
                "block_dims": [64, 128, 256],
                "pretrain": "resnet34",
                "num_blocks": 2,
                "iters": iters,
                "url": "MemorySlices/Tartan-C-T-TSKH-spring540x960-M",
                "init_backbone_with_imagenet": False,
            }
        )
        self.args = args

        resolved_model_path = model_path or args.model
        if not os.path.exists(resolved_model_path):
            raise FileNotFoundError(
                "SEA-RAFT checkpoint not found. Expected a local checkpoint under "
                f"'{resolved_model_path}'. Put the shared weight file in checkpoints/sea-raft/."
            )

        self.model = RAFT(args)
        load_ckpt(self.model, resolved_model_path)

        self.model = self.model.to(device=self.device, dtype=self.dtype)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def _compute_flows(self, videos: torch.Tensor, max_pairs_per_chunk: int = 32) -> torch.Tensor:
        """Return flows with shape [B, T-1, 2, H, W].

        Frame pairs are processed in chunks of *max_pairs_per_chunk* to
        avoid OOM on high-resolution long clips.
        """
        batch, channels, steps, height, width = videos.shape
        if steps < 2:
            return torch.zeros(batch, 0, 2, height, width, device=self.device, dtype=self.dtype)

        frames1 = videos[:, :, :-1].permute(0, 2, 1, 3, 4).reshape(-1, channels, height, width).contiguous()
        frames2 = videos[:, :, 1:].permute(0, 2, 1, 3, 4).reshape(-1, channels, height, width).contiguous()

        total_pairs = frames1.shape[0]
        flow_chunks = []
        for start in range(0, total_pairs, max_pairs_per_chunk):
            end = min(start + max_pairs_per_chunk, total_pairs)
            f1_chunk = frames1[start:end]
            f2_chunk = frames2[start:end]
            padder = InputPadder(f1_chunk.shape)
            f1_p, f2_p = padder.pad(f1_chunk, f2_chunk)
            output = self.model(f1_p, f2_p, iters=self.args.iters, test_mode=True)
            flow_chunks.append(padder.unpad(output["final"]))
            del f1_p, f2_p, output

        flow = torch.cat(flow_chunks, dim=0)
        del frames1, frames2, flow_chunks
        return flow.view(batch, steps - 1, 2, height, width)

    def _masked_mean(self, values: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        if mask is None:
            return values.reshape(values.shape[0], -1).mean(dim=1)
        weights = mask.to(dtype=values.dtype)
        summed = (values * weights).reshape(values.shape[0], -1).sum(dim=1)
        denom = weights.reshape(weights.shape[0], -1).sum(dim=1).clamp_min(self.eps)
        return summed / denom

    def _spatial_jitter(self, flows: torch.Tensor, motion_mask: torch.Tensor) -> torch.Tensor:
        dx = flows[..., :, 1:] - flows[..., :, :-1]
        dy = flows[..., 1:, :] - flows[..., :-1, :]

        dx_mag = torch.sqrt((dx ** 2).sum(dim=2) + self.eps)
        dy_mag = torch.sqrt((dy ** 2).sum(dim=2) + self.eps)

        dx_mask = motion_mask[..., :, 1:] & motion_mask[..., :, :-1]
        dy_mask = motion_mask[..., 1:, :] & motion_mask[..., :-1, :]

        dx_score = self._masked_mean(dx_mag, dx_mask)
        dy_score = self._masked_mean(dy_mag, dy_mask)
        return 0.5 * (dx_score + dy_score)

    def _temporal_jitter(self, flows: torch.Tensor, motion_mask: torch.Tensor) -> torch.Tensor:
        if flows.shape[1] < 2:
            return torch.zeros(flows.shape[0], device=flows.device, dtype=flows.dtype)

        delta = flows[:, 1:] - flows[:, :-1]
        delta_mag = torch.sqrt((delta ** 2).sum(dim=2) + self.eps)
        delta_mask = motion_mask[:, 1:] & motion_mask[:, :-1]
        return self._masked_mean(delta_mag, delta_mask)

    @torch.no_grad()
    def forward(self, videos: torch.Tensor, prompts=None, return_details: bool = False):
        del prompts
        batch, _, steps, _, _ = videos.shape
        if steps < 2:
            zeros = torch.zeros(batch, device=self.device, dtype=torch.float32)
            if return_details:
                return {
                    "reward": zeros,
                    "spatial_jitter": zeros,
                    "temporal_jitter": zeros,
                    "motion_ratio": zeros,
                }
            return zeros

        videos = videos.to(device=self.device, dtype=self.dtype)
        flows = self._compute_flows(videos)
        flow_mag = torch.sqrt((flows ** 2).sum(dim=2) + self.eps)
        motion_mask = flow_mag > self.motion_threshold

        spatial_jitter = self._spatial_jitter(flows, motion_mask)
        temporal_jitter = self._temporal_jitter(flows, motion_mask)
        jitter = self.spatial_weight * spatial_jitter + self.temporal_weight * temporal_jitter
        reward = torch.exp(-self.reward_scale * jitter).to(dtype=torch.float32)

        if return_details:
            motion_ratio = motion_mask.to(dtype=torch.float32).reshape(batch, -1).mean(dim=1)
            return {
                "reward": reward,
                "spatial_jitter": spatial_jitter.to(dtype=torch.float32),
                "temporal_jitter": temporal_jitter.to(dtype=torch.float32),
                "motion_ratio": motion_ratio,
            }
        return reward

    def __call__(self, videos, prompts=None, return_details: bool = False):
        return self.forward(videos, prompts=prompts, return_details=return_details)
