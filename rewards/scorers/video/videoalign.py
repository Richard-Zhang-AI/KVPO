import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torchvision.transforms.functional as F

from ... import VIDEOALIGN_CKPT_PATH

class VideoAlignScorer(nn.Module):
    def __init__(self, model_path=VIDEOALIGN_CKPT_PATH, device='cuda', dtype=torch.bfloat16, reward_type="MQ", use_grayscale=False):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.model_path = model_path
        self.reward_type = reward_type
        self.use_grayscale = use_grayscale
        self.scorer = None
        
        valid_types = ["VQ", "MQ", "TA", "Overall"]
        assert self.reward_type in valid_types, f"reward_type must be one of {valid_types}"

    def _ensure_scorer(self):
        if self.scorer is not None:
            return

        try:
            from ...reward_models.VideoAlign.wan_inference import VideoVLMRewardInference
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "VideoAlign inference code is not present under rewards.reward_models.VideoAlign. "
                f"Current checkpoint directory is '{self.model_path}'."
            ) from exc

        self.scorer = VideoVLMRewardInference(
            load_from_pretrained=self.model_path,
            device=self.device,
            dtype=self.dtype,
        )

    def __call__(
        self,
        videos,
        prompts,
        *,
        fps=None,
        num_frames=None,
        max_pixels=None,
        source_fps=None,
        sample_type=None,
    ):
        """
        Args:
            videos: List[torch.Tensor] or torch.Tensor.
            prompts: List[str]
        """
        if isinstance(videos, torch.Tensor):
            if videos.dtype != torch.uint8:
                videos = (videos * 255).round().clamp(0, 255).to(torch.uint8)
            videos = [v for v in videos]
        
        if self.use_grayscale:
            processed_videos = []
            for v in videos:
                processed_videos.append(F.rgb_to_grayscale(v, num_output_channels=3))
            videos = processed_videos
            

        self._ensure_scorer()

        results = self.scorer.reward_from_frames(
            videos,
            prompts,
            fps=fps,
            num_frames=num_frames,
            max_pixels=max_pixels,
            source_fps=source_fps,
            sample_type=sample_type,
            use_norm=True,
        )
        
        scores = results[self.reward_type]
        
        if not isinstance(scores, torch.Tensor):
            scores = torch.tensor(scores, device=self.device, dtype=self.dtype)
        else:
            scores = scores.to(dtype=self.dtype, device=self.device)
            
        return scores



def main():
    import torchvision
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scorer = VideoAlignScorer(device=device, dtype=torch.bfloat16)
    
    video_path = "./videos/self_forcing_dmd/A 3D animation of a small, round, fluffy creature with big, expressive eyes exploring a vibrant, enc-0.mp4"
    
    if not os.path.exists(video_path):
        print(f"Warning: test video not found at {video_path}; using a random tensor test.")
        videos = torch.randint(0, 256, (1, 10, 3, 480, 832), dtype=torch.uint8).float() / 255.0
        prompts = ["A test video prompt"]
    else:
        filename = os.path.basename(video_path)
        prompt = filename.split(", enc-")[0]
        prompts = [prompt]
        
        video_frames, _, _ = torchvision.io.read_video(video_path, pts_unit='sec', output_format='THWC')
        # [F, H, W, C] -> [F, C, H, W]
        video_frames = video_frames.permute(0, 3, 1, 2)
        videos = video_frames.unsqueeze(0).float() / 255.0
        print(f"Loaded video: {filename}")
        print(f"Prompt: {prompt}")
        print(f"Video shape: {videos.shape}")
    
    with torch.no_grad():
        scores = scorer(videos, prompts)
        print(f"VideoAlign Overall Score: {scores.item():.4f}")


if __name__ == "__main__":
    main()
