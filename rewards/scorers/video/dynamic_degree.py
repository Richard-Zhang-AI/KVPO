
import os
import sys
import torch
import torch.nn as nn
from typing import Dict, Tuple
from easydict import EasyDict as edict

from ... import CKPT_PATH, SEA_RAFT_CKPT_PATH

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SEA_RAFT_CORE_PATH = os.path.abspath(
    os.path.join(CURRENT_DIR, "..", "..", "reward_models", "sea_raft", "core")
)
if SEA_RAFT_CORE_PATH not in sys.path:
    sys.path.insert(0, SEA_RAFT_CORE_PATH)

from raft import RAFT
from raft_utils.utils import load_ckpt, InputPadder


class DynamicDegreeScorer(nn.Module):
    """
    Lightweight dynamic-degree scorer used as a training reward.

    Normalization formula:
    m = min{1, sqrt(u² + v²) / (sigma * sqrt(H² + W²))}

    Evaluated dimensions:
    1. Motion magnitude: moderate motion scores high; static or excessive motion scores low.
    2. Temporal consistency: smooth transitions score high; flicker or abrupt changes score low.

    Final reward = magnitude_weight * magnitude_reward + temporal_weight * temporal_reward.
    """
    
    def __init__(
        self,
        device,
        model_path=None,
        dtype=torch.float32,
        sigma: float = 0.15,
        clip_normalized: bool = True,
        magnitude_weight: float = 1.5,
        temporal_weight: float = 1,
        magnitude_target: float = 0.6,
        magnitude_tolerance: float = 0.2,
        magnitude_max: float = 1.2,
        temporal_smoothness_sigma: float = 0.1,
        iters: int = 4,
    ):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.sigma = sigma
        self.clip_normalized = clip_normalized
        
        self.magnitude_weight = magnitude_weight
        self.temporal_weight = temporal_weight
        
        self.magnitude_target = magnitude_target
        self.magnitude_tolerance = magnitude_tolerance
        self.magnitude_max = magnitude_max
        
        self.temporal_smoothness_sigma = temporal_smoothness_sigma
        
        args = edict({
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
        })
        
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

    def normalize_flow_magnitude(self, magnitude, H, W):
        """
        Normalize optical-flow magnitude by the frame diagonal.

        Formula: m = min{1, sqrt(u² + v²) / (sigma * sqrt(H² + W²))}

        Args:
            magnitude: [..., H, W] optical-flow magnitude sqrt(u² + v²).
            H, W: frame height and width.

        Returns:
            normalized_magnitude: normalized magnitude.
        """
        diagonal = torch.sqrt(torch.tensor(H ** 2 + W ** 2, dtype=torch.float32, device=magnitude.device))
        
        normalization_factor = self.sigma * diagonal
        
        normalized = magnitude / normalization_factor
        
        if self.clip_normalized:
            normalized = torch.clamp(normalized, min=0, max=1)
        
        return normalized

    def get_top_percentile_mean(self, magnitude, percentile=95):
        """
        Compute the mean magnitude over the top percentile to suppress noise.

        Args:
            magnitude: [B, T, H, W]
            percentile: percentile threshold; 95 means the top 5%.

        Returns:
            top_mean: [B, T]
        """
        B, T, H, W = magnitude.shape
        mag_flat = magnitude.view(B, T, -1)
        
        total_pixels = H * W
        k = max(1, int(total_pixels * (100 - percentile) / 100))
        
        top_k_mag, _ = torch.topk(mag_flat, k, dim=-1)  # [B, T, k]
        top_mean = top_k_mag.mean(dim=-1)  # [B, T]
        
        return top_mean

    def compute_optical_flow(self, videos):
        """
        Compute optical flow and return normalized magnitudes.

        Args:
            videos: [B, C, T, H, W], value range [0, 1].

        Returns:
            normalized_magnitude: [B, T-1, H, W] normalized magnitude.
        """
        B, C, T, H, W = videos.shape
        
        if T < 2:
            return torch.zeros(B, 0, H, W, device=self.device, dtype=self.dtype)
        
        videos = videos.to(dtype=self.dtype)
    
        frames1 = videos[:, :, :-1, :, :].permute(0, 2, 1, 3, 4).reshape(-1, C, H, W).contiguous()
        frames2 = videos[:, :, 1:, :, :].permute(0, 2, 1, 3, 4).reshape(-1, C, H, W).contiguous()
        
        padder = InputPadder(frames1.shape)
        f1, f2 = padder.pad(frames1, frames2)

        # with torch.cuda.amp.autocast(enabled=False):
        output = self.model(f1, f2, iters=self.args.iters, test_mode=True)

        flow_up = padder.unpad(output['final'])
        
        flows = flow_up.view(B, T-1, 2, H, W)
        
        magnitude = torch.sqrt(flows[:, :, 0]**2 + flows[:, :, 1]**2)  # [B, T-1, H, W]
        
        normalized_magnitude = self.normalize_flow_magnitude(magnitude, H, W)
        
        return normalized_magnitude

    def compute_magnitude_reward(self, normalized_magnitude):
        """
        Motion-magnitude reward.

        This rewards moderate motion and penalizes static or excessive motion
        with a Gaussian-shaped reward curve.

        Args:
            normalized_magnitude: [B, T, H, W] normalized optical-flow magnitude.

        Returns:
            reward: [B] motion-magnitude reward in [0, 1].
        """
        B, T, H, W = normalized_magnitude.shape
        
        avg_magnitude = self.get_top_percentile_mean(normalized_magnitude, percentile=95)  # [B, T]
        
        video_magnitude = avg_magnitude.mean(dim=-1)  # [B]
        
        # reward = exp(-((m - target)² / (2 * tolerance²)))
        reward = torch.exp(
            -((video_magnitude - self.magnitude_target) ** 2) / 
            (2 * self.magnitude_tolerance ** 2)
        )
        
        if self.magnitude_max > 0:
            over_motion_penalty = torch.clamp(
                (video_magnitude - self.magnitude_max) / self.magnitude_max, 
                min=0, max=1
            )
            reward = reward * (1 - over_motion_penalty)
        
        static_penalty = torch.exp(-video_magnitude * 3.0)
        reward = reward * (1 - 0.5 * static_penalty)
        
        return reward

    def compute_temporal_reward(self, normalized_magnitude):
        """
        Temporal-consistency reward.

        This rewards smooth motion and penalizes flicker or abrupt changes.

        Args:
            normalized_magnitude: [B, T, H, W]

        Returns:
            reward: [B] temporal-consistency reward in [0, 1].
        """
        B, T, H, W = normalized_magnitude.shape
        
        if T < 2:
            return torch.ones(B, device=normalized_magnitude.device)
        
        frame_magnitude = normalized_magnitude.mean(dim=(2, 3))  # [B, T]
        
        mag_diff = torch.abs(frame_magnitude[:, 1:] - frame_magnitude[:, :-1])  # [B, T-1]
        avg_acceleration = mag_diff.mean(dim=-1)  # [B]
        
        # reward = exp(-acceleration / sigma)
        reward = torch.exp(-avg_acceleration / self.temporal_smoothness_sigma)
        
        return reward

    @torch.no_grad()
    def forward(self, videos, return_details=False):
        """
        Run the scorer.

        Args:
            videos: [B, C, T, H, W], value range [0, 1].
            return_details: whether to return per-dimension scores.

        Returns:
            If return_details is False, returns [B] combined rewards.
            If return_details is True, returns a dict with per-dimension scores.
        """
        B, C, T, H, W = videos.shape
        
        if T < 2:
            if return_details:
                return {
                    'reward': torch.zeros(B, device=self.device),
                    'magnitude_reward': torch.zeros(B, device=self.device),
                    'temporal_reward': torch.zeros(B, device=self.device),
                    'normalized_magnitude_mean': torch.zeros(B, device=self.device),
                }
            return torch.zeros(B, device=self.device)
        
        normalized_magnitude = self.compute_optical_flow(videos)
        
        mag_reward = self.compute_magnitude_reward(normalized_magnitude)
        temp_reward = self.compute_temporal_reward(normalized_magnitude)
        
        total_reward = (
            self.magnitude_weight * mag_reward + 
            self.temporal_weight * temp_reward
        )
        
        if return_details:
            avg_normalized_mag = self.get_top_percentile_mean(
                normalized_magnitude, percentile=95
            ).mean(dim=1)  # [B]
            
            return {
                'reward': total_reward,
                'magnitude_reward': mag_reward,
                'temporal_reward': temp_reward,
                'normalized_magnitude_mean': avg_normalized_mag,
            }
        else:
            return total_reward

    def __call__(self, videos, return_details=False):
        """Convenience wrapper around forward()."""
        return self.forward(videos, return_details)



if __name__ == "__main__":
    example_usage()
