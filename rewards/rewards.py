from PIL import Image
import os
import numpy as np
import torch

from . import CKPT_PATH, HPSV3_CKPT_PATH


def _ensure_btchw(videos: torch.Tensor) -> torch.Tensor:
    """Ensure video tensor is in [B, T, C, H, W] layout.

    Detects [B, T, H, W, C] (C-last) by checking ``shape[4] == 3`` and
    ``shape[2] != 3`` and permutes to [B, T, C, H, W] when necessary.
    """
    if (
        videos.ndim == 5
        and videos.shape[4] in (1, 3)
        and videos.shape[2] not in (1, 3)
    ):
        videos = videos.permute(0, 1, 4, 2, 3)
    return videos


def _move_reward_object_to_device(obj, device, dtype=None, _visited=None):
    """Best-effort device move for reward wrappers and nested scorer objects."""
    if obj is None:
        return

    if _visited is None:
        _visited = set()
    obj_id = id(obj)
    if obj_id in _visited:
        return
    _visited.add(obj_id)

    if hasattr(obj, "device"):
        try:
            obj.device = device
        except (AttributeError, TypeError):
            # Some wrapped models expose read-only `.device`.
            pass
    if dtype is not None and hasattr(obj, "dtype"):
        try:
            obj.dtype = dtype
        except (AttributeError, TypeError):
            # Some wrapped models expose read-only `.dtype`.
            pass

    if isinstance(obj, torch.nn.Module):
        try:
            if dtype is not None:
                obj.to(device=device, dtype=dtype)
            else:
                obj.to(device=device)
        except TypeError:
            obj.to(device)
    elif hasattr(obj, "to") and callable(getattr(obj, "to")):
        try:
            if dtype is not None:
                obj.to(device=device, dtype=dtype)
            else:
                obj.to(device=device)
        except Exception:
            try:
                obj.to(device)
            except Exception:
                pass

    for attr_name in ("model", "scorer", "predictor"):
        child = getattr(obj, attr_name, None)
        if child is not None and child is not obj:
            _move_reward_object_to_device(child, device, dtype=dtype, _visited=_visited)


def _attach_reward_device_hook(fn, scorer, *, dtype=None):
    """Attach a device-move hook so trainers can offload reward scorers."""
    def _move(target_device):
        _move_reward_object_to_device(scorer, target_device, dtype=dtype)

    fn._kvpo_reward_scorer = scorer
    fn._kvpo_move_to_device = _move
    return fn


def aesthetic_score(device):
    from .scorers.image.aesthetic import AestheticScorer

    scorer = AestheticScorer(dtype=torch.float32, device=device)

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8)
        else:
            images = images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
            images = torch.tensor(images, dtype=torch.uint8)
        scores = scorer(images)
        return scores, {}

    return _attach_reward_device_hook(_fn, scorer, dtype=torch.float32)


def video_hpsv2_local(device):
    from .scorers.image.hpsv2 import HPSv2Scorer

    scorer = HPSv2Scorer(dtype=torch.float32, device=device)

    def _fn(videos, prompts, metadata=None):
        if not isinstance(videos, torch.Tensor):
            videos = torch.from_numpy(videos).permute(0, 1, 4, 2, 3)  # (B,F,H,W,C) -> (B,F,C,H,W)
        else:
            videos = _ensure_btchw(videos)

        if videos.dtype == torch.uint8:
            videos = videos.float() / 255.0

        batch_size, num_frames, c, h, w = videos.shape

        # Flatten frames and replicate each prompt once per frame
        flat_images = videos.reshape(-1, c, h, w)
        flat_prompts = [p for p in prompts for _ in range(num_frames)]

        # Score in mini-batches to avoid OOM
        reward_batch_size = 8
        all_flat_scores = []
        for i in range(0, len(flat_images), reward_batch_size):
            batch_scores = scorer(flat_images[i:i+reward_batch_size], flat_prompts[i:i+reward_batch_size])
            all_flat_scores.append(batch_scores)
            torch.cuda.empty_cache()

        flat_scores = torch.cat(all_flat_scores, dim=0)
        frame_scores = flat_scores.view(batch_size, num_frames)

        video_rewards = []
        for i in range(batch_size):
            scores = frame_scores[i].tolist()
            # Aggregate via top-30% mean to reduce noise from low-quality frames
            scores.sort(reverse=True)
            top_k = max(1, int(len(scores) * 0.3))
            video_rewards.append(sum(scores[:top_k]) / top_k)

        return np.array(video_rewards, dtype=np.float32), {}

    return _attach_reward_device_hook(_fn, scorer, dtype=torch.float32)


def video_hpsv3_local(device, **kwargs):
    del kwargs
    from .scorers.video.hpsv3 import HPSv3RewardInferencer

    # HPSv3 is a 7B VLM scorer; it returns (mu, sigma) per sample
    scorer = HPSv3RewardInferencer(
            config_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scorers/configs/HPSv3_7B.yaml'),
            checkpoint_path=HPSV3_CKPT_PATH,
            device=device
        )

    def _fn(videos, prompts, metadata=None):
        if not isinstance(videos, torch.Tensor):
            videos = torch.from_numpy(videos).permute(0, 1, 4, 2, 3)  # (B,F,H,W,C) -> (B,F,C,H,W)
        else:
            videos = _ensure_btchw(videos)

        if videos.dtype == torch.uint8:
            videos = videos.float() / 255.0

        batch_size, num_frames, c, h, w = videos.shape

        # HPSv3 expects a list of PIL images, so convert each frame
        flat_images_tensor = videos.reshape(-1, c, h, w)
        flat_images_np = (flat_images_tensor * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
        flat_images_np = flat_images_np.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        flat_images_pil = [Image.fromarray(img) for img in flat_images_np]

        flat_prompts = [p for p in prompts for _ in range(num_frames)]

        # Smaller batch size than HPSv2 due to the 7B model's higher VRAM usage
        reward_batch_size = 4
        all_flat_scores = []

        for i in range(0, len(flat_images_pil), reward_batch_size):
            batch_scores = scorer.reward(flat_images_pil[i:i+reward_batch_size], flat_prompts[i:i+reward_batch_size])
            # scorer.reward returns [batch, 2] as (mu, sigma); take mu
            if batch_scores.ndim == 2:
                batch_scores = batch_scores[:, 0]
            all_flat_scores.append(batch_scores.cpu())
            torch.cuda.empty_cache()

        flat_scores = torch.cat(all_flat_scores, dim=0)
        frame_scores = flat_scores.view(batch_size, num_frames)

        video_rewards_all = []
        video_rewards_top30 = []

        for i in range(batch_size):
            scores = frame_scores[i].tolist()
            avg_all = sum(scores) / len(scores)
            video_rewards_all.append(float(avg_all))
            # Keep the historical top-30% aggregate in metadata for comparison.
            scores_sorted = sorted(scores, reverse=True)
            top_k = max(1, int(len(scores) * 0.3))
            video_rewards_top30.append(sum(scores_sorted[:top_k]) / top_k)

        # Use full-frame average as the primary reward. The previous top-30%
        # aggregate is still returned in metadata for debugging/analysis.
        return np.array(video_rewards_all, dtype=np.float32), {
            "all_frame_avg": video_rewards_all,
            "top30_frame_avg": video_rewards_top30,
        }

    return _attach_reward_device_hook(_fn, scorer)


def video_hpsv3_score(device, **kwargs):
    return video_hpsv3_local(device=device, **kwargs)


def videoalign_score(device, reward_type="Overall", use_grayscale=False):
    from .scorers.video.videoalign import VideoAlignScorer

    # reward_type selects the scoring dimension: "Overall", "VQ" (visual quality),
    # "MQ" (motion quality, grayscale-sensitive), or "TA" (text alignment)
    scorer = VideoAlignScorer(device=device, dtype=torch.bfloat16, reward_type=reward_type, use_grayscale=use_grayscale)

    def _fn(videos, prompts, metadata=None):
        if not isinstance(videos, torch.Tensor):
            videos = torch.from_numpy(videos).permute(0, 1, 4, 2, 3)
        else:
            videos = _ensure_btchw(videos)

        if videos.dtype != torch.uint8:
            videos = (videos * 255).round().clamp(0, 255).to(torch.uint8)

        metadata = metadata or {}
        scores_tensor = scorer(
            list(videos),
            prompts,
            fps=metadata.get("fps"),
            num_frames=metadata.get("num_frames"),
            max_pixels=metadata.get("max_pixels"),
            source_fps=metadata.get("source_fps"),
            sample_type=metadata.get("sample_type"),
        )
        return scores_tensor.tolist(), {}

    return _attach_reward_device_hook(_fn, scorer, dtype=torch.bfloat16)


# Convenience wrappers for individual VideoAlign dimensions
def videoalign_vq_score(device):
    return videoalign_score(device, reward_type="VQ", use_grayscale=False)

def videoalign_mq_score(device):
    return videoalign_score(device, reward_type="MQ", use_grayscale=True)

def videoalign_mq_rgb_score(device):
    return videoalign_score(device, reward_type="MQ", use_grayscale=False)

def videoalign_ta_score(device):
    return videoalign_score(device, reward_type="TA", use_grayscale=False)


def motion_smoothness_score(device):
    from .scorers.video.flowscorer import OpticalFlowSmoothnessScorer
    scorer = OpticalFlowSmoothnessScorer(dtype=torch.float32, device=device)

    def _fn(videos, prompts, metadata=None):
        if not isinstance(videos, torch.Tensor):
            videos = torch.from_numpy(videos).permute(0, 4, 1, 2, 3)  # (B,F,H,W,C) -> (B,C,F,H,W)
        else:
            videos = _ensure_btchw(videos)
            # RAFT expects (B, C, T, H, W), permute from (B, T, C, H, W)
            videos = videos.permute(0, 2, 1, 3, 4)

        if videos.dtype == torch.uint8:
            videos = videos.float() / 255.0

        scores = scorer(videos, prompts)
        return scores.cpu().tolist(), {}

    return _attach_reward_device_hook(_fn, scorer, dtype=torch.float32)

def dynamic_degree_score(device, model_path=None, dtype=torch.float32, **kwargs):
    from .scorers.video.dynamic_degree import DynamicDegreeScorer
    scorer = DynamicDegreeScorer(device=device, model_path=model_path, dtype=dtype, **kwargs)

    def _fn(videos, prompts, metadata=None):
        del prompts, metadata
        if not isinstance(videos, torch.Tensor):
            videos = torch.from_numpy(videos).permute(0, 4, 1, 2, 3)  # (B,F,H,W,C) -> (B,C,F,H,W)
        else:
            videos = _ensure_btchw(videos)
            videos = videos.permute(0, 2, 1, 3, 4)  # (B,T,C,H,W) -> (B,C,T,H,W)

        if videos.dtype == torch.uint8:
            videos = videos.float() / 255.0
        elif not torch.is_floating_point(videos):
            videos = videos.float()

        videos = videos.to(device=scorer.device, dtype=scorer.dtype)

        scores = scorer(videos)
        return scores.cpu().tolist(), {}

    return _attach_reward_device_hook(_fn, scorer, dtype=dtype)


def sea_raft_score(device, **kwargs):
    return dynamic_degree_score(device=device, **kwargs)


def human_worst_window_score(device, window_size=4, score_thresh=0.05):
    from .scorers.video.human_vitdet import ViTDetHumanWorstWindowScorer

    scorer = ViTDetHumanWorstWindowScorer(
        device=device,
        window_size=window_size,
        score_thresh=score_thresh,
    )

    def _fn(videos, prompts, metadata=None):
        if metadata is None:
            metadata = {}

        scores = scorer(videos, prompts)
        return scores.cpu().tolist(), {
            "window_size": scorer.window_size,
            "score_thresh": scorer.score_thresh,
        }

    return _attach_reward_device_hook(_fn, scorer)


def face_consistency_score(device, window_size=4, min_face_ratio=0.3, use_anchor=True):
    from .scorers.video.face_consistency import FaceConsistencyScorer

    scorer = FaceConsistencyScorer(
        device=device,
        window_size=window_size,
        min_face_ratio=min_face_ratio,
        use_anchor=use_anchor,
    )

    def _fn(videos, prompts, metadata=None):
        if metadata is None:
            metadata = {}

        scores, extra = scorer(videos, prompts)
        return scores.cpu().tolist(), extra

    return _attach_reward_device_hook(_fn, scorer)


def unifiedreward_score_sglang(device):
    # Requires a running sglang server:
    #   python -m sglang.launch_server --model-path CodeGoat24/UnifiedReward-7b-v1.5
    #       --api-key flowgrpo --port 17140 --chat-template chatml-llava
    import asyncio
    from openai import AsyncOpenAI
    import base64
    from io import BytesIO
    import re

    def pil_image_to_base64(image):
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        encoded_image_text = base64.b64encode(buffered.getvalue()).decode("utf-8")
        base64_qwen = f"data:image;base64,{encoded_image_text}"
        return base64_qwen

    def _extract_scores(text_outputs):
        # Parse "Final Score: X" (1-5) from model free-text output; default to 0 on failure
        scores = []
        pattern = r"Final Score:\s*([1-5](?:\.\d+)?)"
        for text in text_outputs:
            match = re.search(pattern, text)
            if match:
                try:
                    scores.append(float(match.group(1)))
                except ValueError:
                    scores.append(0.0)
            else:
                scores.append(0.0)
        return scores

    client = AsyncOpenAI(base_url="http://127.0.0.1:17140/v1", api_key="flowgrpo")

    async def evaluate_image(prompt, image):
        question = f"<image>\nYou are given a text caption and a generated image based on that caption. Your task is to evaluate this image based on two key criteria:\n1. Alignment with the Caption: Assess how well this image aligns with the provided caption. Consider the accuracy of depicted objects, their relationships, and attributes as described in the caption.\n2. Overall Image Quality: Examine the visual quality of this image, including clarity, detail preservation, color accuracy, and overall aesthetic appeal.\nBased on the above criteria, assign a score from 1 to 5 after 'Final Score:'.\nYour task is provided as follows:\nText Caption: [{prompt}]"
        images_base64 = pil_image_to_base64(image)
        response = await client.chat.completions.create(
            model="UnifiedReward-7b-v1.5",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": images_base64},
                        },
                        {
                            "type": "text",
                            "text": question,
                        },
                    ],
                },
            ],
            temperature=0,
        )
        return response.choices[0].message.content

    async def evaluate_batch_image(images, prompts):
        tasks = [evaluate_image(prompt, img) for prompt, img in zip(prompts, images)]
        results = await asyncio.gather(*tasks)
        return results

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC

        # Resize to 512x512 to match the model's expected input resolution
        images = [Image.fromarray(image).resize((512, 512)) for image in images]

        text_outputs = asyncio.run(evaluate_batch_image(images, prompts))
        score = _extract_scores(text_outputs)
        score = [sc / 5.0 for sc in score]  # Normalize 1-5 scale to [0, 1]
        return score, {}

    return _fn


def vr_thinker_score(device, *args, **kwargs):
    del device, args, kwargs
    raise ModuleNotFoundError(
        "vr_thinker_score is not available in this checkout because rewards/scorers/video/vr_thinker.py is missing."
    )


def random_score(device=None, **kwargs):
    del device, kwargs

    def _fn(videos, prompts, metadata=None):
        del prompts, metadata
        return torch.randn(len(videos)).tolist(), {}

    return _fn


def build_named_reward_fn(name, device, **kwargs):
    normalized = str(name).strip().lower()
    removed_groups = {
        "reward1": "use reward_components: [video_hpsv3, videoalign_mq]",
        "hpsv3_motion": "use reward_components: [video_hpsv3, videoalign_mq]",
        "hpsv3_plus_motion": "use reward_components: [video_hpsv3, videoalign_mq]",
        "reward2": "use reward_components: [dynamic_degree, videoalign_ta]",
        "flow_text": "use reward_components: [dynamic_degree, videoalign_ta]",
        "flow_plus_text": "use reward_components: [dynamic_degree, videoalign_ta]",
    }
    if normalized in removed_groups:
        raise ValueError(
            f"Reward group '{name}' has been removed; {removed_groups[normalized]}."
        )

    canonical_name = {
        "random": "random",
        "aesthetic": "aesthetic",
        "video_hpsv2": "video_hpsv2",
        "video_hpsv2_local": "video_hpsv2",
        "video_hpsv3": "video_hpsv3",
        "video_hpsv3_local": "video_hpsv3",
        "hpsv3": "video_hpsv3",
        "videoalign": "videoalign_overall",
        "videoalign_score": "videoalign_overall",
        "videoalign_overall": "videoalign_overall",
        "videoalign_vq": "videoalign_vq",
        "videoalign_vq_score": "videoalign_vq",
        "videoalign_mq": "videoalign_mq",
        "videoalign_mq_score": "videoalign_mq",
        "videoalign_mq_rgb": "videoalign_mq_rgb",
        "videoalign_mq_rgb_score": "videoalign_mq_rgb",
        "videoalign_ta": "videoalign_ta",
        "videoalign_ta_score": "videoalign_ta",
        "dynamic_degree": "dynamic_degree",
        "dynamic_degree_score": "dynamic_degree",
        "sea_raft": "dynamic_degree",
        "sea_raft_score": "dynamic_degree",
        "motion_smoothness": "motion_smoothness",
        "motion_smoothness_score": "motion_smoothness",
        "human_worst_window": "human_worst_window",
        "human_worst_window_score": "human_worst_window",
        "face_consistency": "face_consistency",
        "face_consistency_score": "face_consistency",
        "unifiedreward": "unifiedreward",
        "vr_thinker": "vr_thinker",
    }.get(normalized)

    if canonical_name is None:
        raise ValueError(f"Unknown reward_fn: {name}")

    factories = {
        "random": random_score,
        "aesthetic": aesthetic_score,
        "video_hpsv2": video_hpsv2_local,
        "video_hpsv3": video_hpsv3_score,
        "videoalign_overall": lambda device, **inner_kwargs: videoalign_score(
            device=device,
            reward_type="Overall",
            use_grayscale=False,
            **inner_kwargs,
        ),
        "videoalign_vq": videoalign_vq_score,
        "videoalign_mq": videoalign_mq_score,
        "videoalign_mq_rgb": videoalign_mq_rgb_score,
        "videoalign_ta": videoalign_ta_score,
        "dynamic_degree": dynamic_degree_score,
        "motion_smoothness": motion_smoothness_score,
        "human_worst_window": human_worst_window_score,
        "face_consistency": face_consistency_score,
        "unifiedreward": unifiedreward_score_sglang,
        "vr_thinker": vr_thinker_score,
    }
    scorer = factories[canonical_name](device=device, **kwargs)
    return canonical_name, scorer


def multi_score(device, score_dict):
    score_functions = {
        "aesthetic": aesthetic_score,
        "unifiedreward": unifiedreward_score_sglang,
        "video_hpsv2": video_hpsv2_local,
        "video_hpsv2_local": video_hpsv2_local,
        "video_hpsv3_local": video_hpsv3_local,
        "motion_smoothness_score": motion_smoothness_score,
        "videoalign_score": lambda dev: videoalign_score(dev, "Overall"),
        "videoalign_vq_score": videoalign_vq_score,
        "videoalign_mq_score": videoalign_mq_score,
        "videoalign_ta_score": videoalign_ta_score,
        "dynamic_degree_score": dynamic_degree_score,
        "human_worst_window_score": human_worst_window_score,
        "face_consistency_score": face_consistency_score,
    }
    score_fns = {name: score_functions[name](device) for name in score_dict}

    # only_strict is only for geneval. During training, only the strict reward is needed, and non-strict rewards don't need to be computed, reducing reward calculation time.
    def _fn(images, prompts, metadata, only_strict=True):
        total_scores = []
        score_details = {}

        # Detect if input is video [B, F, C, H, W]
        is_video = isinstance(images, torch.Tensor) and images.ndim == 5

        for score_name, weight in score_dict.items():
            current_score_name = score_name
            if is_video and score_name == "hpsv2":
                current_score_name = "video_hpsv2"
                if "video_hpsv2" not in score_fns:
                    score_fns["video_hpsv2"] = video_hpsv2_local(device)

            if score_name == "geneval":
                scores, rewards, strict_rewards, group_rewards, group_strict_rewards = score_fns[score_name](
                    images, prompts, metadata, only_strict
                )
                score_details["accuracy"] = rewards
                score_details["strict_accuracy"] = strict_rewards
                for key, value in group_strict_rewards.items():
                    score_details[f"{key}_strict_accuracy"] = value
                for key, value in group_rewards.items():
                    score_details[f"{key}_accuracy"] = value
            else:
                scores, rewards = score_fns[score_name](images, prompts, metadata)
            score_details[score_name] = scores
            weighted_scores = [weight * score for score in scores]

            if not total_scores:
                total_scores = weighted_scores
            else:
                total_scores = [total + weighted for total, weighted in zip(total_scores, weighted_scores)]

        score_details["avg"] = total_scores
        return score_details, {}

    return _fn


if __name__ == "__main__":
    pass


__all__ = [
    "aesthetic_score",
    "build_named_reward_fn",
    "dynamic_degree_score",
    "face_consistency_score",
    "human_worst_window_score",
    "motion_smoothness_score",
    "multi_score",
    "random_score",
    "sea_raft_score",
    "unifiedreward_score_sglang",
    "video_hpsv2_local",
    "video_hpsv3_local",
    "video_hpsv3_score",
    "videoalign_mq_rgb_score",
    "videoalign_mq_score",
    "videoalign_score",
    "videoalign_ta_score",
    "videoalign_vq_score",
    "vr_thinker_score",
]
