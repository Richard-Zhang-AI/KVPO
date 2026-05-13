"""Video reward interface for long/short video scoring.

Provides raw reward scores only.  Advantage computation is handled by the
trainer (e.g. KVPO's anchor-based advantage).

This module is intentionally self-contained under ``rewards`` so other code can
import it without depending on the trainer package.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torchvision

from .rewards import build_named_reward_fn


REWARDS_ROOT = Path(__file__).resolve().parent
KVPO_ROOT = REWARDS_ROOT.parent


VideoInput = Union[str, Path]
PromptInput = Union[str, Sequence[str], Dict[str, Any], None]


@dataclass
class ClipSpec:
    name: str
    start_frame: int
    end_frame: int
    prompt: str
    group: str


class VideoRewardAdvantageInterface:
    """Reward-side interface for long/short video scoring (raw scores only)."""

    def __init__(
        self,
        *,
        device: str = "cuda",
        reward_components: Optional[Sequence[Union[str, Dict[str, Any]]]] = None,
        prompt_jsonl: Optional[Union[str, Path]] = None,
        switch_frame_indices: Union[str, Sequence[int]] = "57,93,129,165,201",
        vae_temporal_stride: int = 4,
        transition_clip_len: int = 48,
        normal_clip_len: int = 48,
        short_video_max_frames: int = 96,
        uniform_long_clip_count: int = 6,
    ):
        self.device = device
        self.prompt_jsonl = Path(prompt_jsonl) if prompt_jsonl is not None else (
            Path(KVPO_ROOT) / "prompts" / "interactive_example.jsonl"
        )
        self.switch_frame_indices = self._parse_switch_frame_indices(switch_frame_indices)
        self.vae_temporal_stride = int(vae_temporal_stride)
        self.transition_clip_len = int(transition_clip_len)
        self.normal_clip_len = int(normal_clip_len)
        self.short_video_max_frames = int(short_video_max_frames)
        self.uniform_long_clip_count = int(uniform_long_clip_count)

        if reward_components is None:
            reward_components = [
                {"name": "video_hpsv3", "weight": 0.5},
                {"name": "videoalign_mq", "weight": 1.0},
                {"name": "motion_smoothness", "weight": 0.5},
            ]

        self.reward_components = self._build_reward_components(reward_components)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_switch_frame_indices(raw_value: Union[str, Sequence[int]]) -> List[int]:
        if isinstance(raw_value, str):
            return [int(x) for x in raw_value.split(",") if x.strip()]
        return [int(x) for x in raw_value]

    def _build_reward_components(
        self,
        components: Sequence[Union[str, Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        built = []
        seen_keys = set()
        for idx, spec in enumerate(components):
            if isinstance(spec, str):
                name = spec
                weight = 1.0
                alias = None
                kwargs = {}
            elif isinstance(spec, dict):
                name = spec.get("name") or spec.get("reward_fn")
                if not name:
                    raise ValueError(f"reward_components[{idx}] must define 'name'")
                weight = float(spec.get("weight", 1.0))
                alias = spec.get("alias")
                kwargs = dict(spec.get("kwargs") or {})
            else:
                raise TypeError(
                    f"reward_components[{idx}] must be a string or dict, "
                    f"got {type(spec).__name__}"
                )

            canonical_name, scorer = build_named_reward_fn(
                name, device=self.device, **kwargs,
            )
            key = str(alias or canonical_name)
            if key in seen_keys:
                raise ValueError(f"Duplicate reward component key: {key}")
            seen_keys.add(key)
            built.append({
                "key": key,
                "name": canonical_name,
                "weight": float(weight),
                "fn": scorer,
            })
        return built

    @staticmethod
    def _normalize_prompts(prompts: PromptInput) -> Optional[List[str]]:
        if prompts is None:
            return None
        if isinstance(prompts, str):
            text = prompts.strip()
            return [text] if text else None
        if isinstance(prompts, dict):
            prompts = prompts.get("prompts")
        if not isinstance(prompts, (list, tuple)):
            raise TypeError(
                "prompts must be a string, a sequence of strings, "
                "or a dict with key 'prompts'"
            )
        normalized = [str(p).strip() for p in prompts if str(p).strip()]
        return normalized or None

    @staticmethod
    def _parse_sample_index_from_path(video_path: Path) -> Optional[int]:
        match = re.match(r"sample(\d+)_branch\d+\.mp4$", video_path.name)
        if match:
            return int(match.group(1))
        return None

    def _load_prompts_from_jsonl(self, sample_index: int) -> List[str]:
        with self.prompt_jsonl.open("r", encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        if sample_index >= len(rows):
            raise ValueError(
                f"Sample index {sample_index} out of range for {self.prompt_jsonl}"
            )
        prompts = rows[sample_index].get("prompts")
        if not prompts:
            raise ValueError(
                f"No prompts found for sample index {sample_index} "
                f"in {self.prompt_jsonl}"
            )
        return [str(p).strip() for p in prompts if str(p).strip()]

    @staticmethod
    def _make_window(
        center: int, total_frames: int, clip_len: int,
    ) -> Tuple[int, int]:
        start = max(0, center - clip_len // 2)
        end = start + clip_len
        if end > total_frames:
            end = total_frames
            start = max(0, end - clip_len)
        return int(start), int(end)

    def _latent_to_rgb_frame(self, latent_frame: int) -> int:
        stride = self.vae_temporal_stride
        return latent_frame * stride - (stride - 1)

    # ------------------------------------------------------------------
    # Clip spec builders
    # ------------------------------------------------------------------

    def _build_prompt_aware_specs(
        self,
        total_frames: int,
        prompts: Sequence[str],
        switch_frame_indices: Sequence[int],
    ) -> List[ClipSpec]:
        if len(prompts) <= 1:
            raise ValueError("Prompt-aware long-video mode requires multiple prompts")
        expected = len(prompts) - 1
        if len(switch_frame_indices) != expected:
            raise ValueError(
                f"Expected {expected} switch_frame_indices for "
                f"{len(prompts)} prompts, got {len(switch_frame_indices)}"
            )

        rgb_boundaries = [
            min(total_frames - 1, self._latent_to_rgb_frame(idx))
            for idx in switch_frame_indices
        ]
        specs: List[ClipSpec] = []

        prev_start = 0
        for i, boundary in enumerate(rgb_boundaries):
            center = (prev_start + boundary) // 2
            start, end = self._make_window(center, total_frames, self.normal_clip_len)
            specs.append(ClipSpec(
                name=f"normal_seg_{i + 1}",
                start_frame=start, end_frame=end,
                prompt=prompts[i], group="normal",
            ))
            prev_start = boundary

        center = (prev_start + total_frames) // 2
        start, end = self._make_window(center, total_frames, self.normal_clip_len)
        specs.append(ClipSpec(
            name=f"normal_seg_{len(prompts)}",
            start_frame=start, end_frame=end,
            prompt=prompts[-1], group="normal",
        ))

        for i, boundary in enumerate(rgb_boundaries):
            start, end = self._make_window(
                boundary, total_frames, self.transition_clip_len,
            )
            prompt = (
                f"Transition between two consecutive scenes. "
                f"First part: {prompts[i]} "
                f"Second part: {prompts[i + 1]} "
                f"Judge whether the transition is visually stable "
                f"and motion is natural across the boundary."
            )
            specs.append(ClipSpec(
                name=f"transition_{i + 1}_{i + 2}",
                start_frame=start, end_frame=end,
                prompt=prompt, group="transition",
            ))

        return specs

    def _build_uniform_long_specs(
        self, total_frames: int, prompt: str,
    ) -> List[ClipSpec]:
        clip_len = min(self.normal_clip_len, total_frames)
        if self.uniform_long_clip_count <= 1 or total_frames <= clip_len:
            center = total_frames // 2
            start, end = self._make_window(center, total_frames, clip_len)
            return [ClipSpec(
                name="key_seg_1",
                start_frame=start, end_frame=end,
                prompt=prompt, group="key",
            )]

        centers = torch.linspace(
            clip_len // 2,
            max(total_frames - clip_len // 2 - 1, clip_len // 2),
            steps=self.uniform_long_clip_count,
        ).round().to(torch.int64).tolist()
        specs = []
        for idx, center in enumerate(centers, start=1):
            start, end = self._make_window(int(center), total_frames, clip_len)
            specs.append(ClipSpec(
                name=f"key_seg_{idx}",
                start_frame=start, end_frame=end,
                prompt=prompt, group="key",
            ))
        return specs

    # ------------------------------------------------------------------
    # Single-video scoring
    # ------------------------------------------------------------------

    @staticmethod
    def _score_video_with_component(
        scorer,
        video_float_tchw: torch.Tensor,
        prompt: str,
    ) -> float:
        clip = video_float_tchw.unsqueeze(0)
        scores, _ = scorer(clip, [prompt], None)
        return float(scores[0])

    def _score_clips(
        self,
        video_float_tchw: torch.Tensor,
        specs: Sequence[ClipSpec],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Score all clips of a single video. Returns raw scores only."""
        clip_results = []
        group_indices: Dict[str, List[int]] = {}

        for spec in specs:
            group_indices.setdefault(spec.group, []).append(len(clip_results))
            clip_results.append({
                "name": spec.name,
                "group": spec.group,
                "start_frame": int(spec.start_frame),
                "end_frame": int(spec.end_frame),
                "prompt": spec.prompt,
                "component_scores": {},
            })

        for comp in self.reward_components:
            for idx, spec in enumerate(specs):
                clip = video_float_tchw[spec.start_frame:spec.end_frame]
                score = self._score_video_with_component(
                    comp["fn"], clip, spec.prompt,
                )
                clip_results[idx]["component_scores"][comp["key"]] = score

        for clip_result in clip_results:
            clip_result["combined_score"] = sum(
                comp["weight"] * clip_result["component_scores"][comp["key"]]
                for comp in self.reward_components
            )

        summary: Dict[str, Any] = {}
        for group_name, indices in group_indices.items():
            summary[group_name] = {
                "num_clips": len(indices),
                "mean_combined_score": sum(
                    clip_results[idx]["combined_score"] for idx in indices
                ) / len(indices),
                "components": {},
            }
            for comp in self.reward_components:
                group_scores = [
                    clip_results[idx]["component_scores"][comp["key"]]
                    for idx in indices
                ]
                summary[group_name]["components"][comp["key"]] = {
                    "weight": float(comp["weight"]),
                    "mean_score": sum(group_scores) / len(group_scores),
                    "score_range": (
                        max(group_scores) - min(group_scores)
                        if len(group_scores) > 1 else 0.0
                    ),
                }
        return clip_results, summary

    # ------------------------------------------------------------------
    # Video loading
    # ------------------------------------------------------------------

    @staticmethod
    def _to_video_float_tchw(
        video: Union[torch.Tensor, Path, str],
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if isinstance(video, (str, Path)):
            video_path = Path(video)
            video_frames, _, info = torchvision.io.read_video(
                str(video_path), pts_unit="sec", output_format="TCHW",
            )
            return video_frames.float() / 255.0, {
                "video_path": str(video_path),
                "fps": float(info.get("video_fps", 16.0)),
                "total_frames": int(video_frames.shape[0]),
            }

        if not isinstance(video, torch.Tensor):
            raise TypeError(f"Unsupported video input type: {type(video).__name__}")

        if video.ndim != 4:
            raise ValueError("Video tensor must have shape [T, C, H, W]")

        if video.dtype == torch.uint8:
            video = video.float() / 255.0
        elif not torch.is_floating_point(video):
            video = video.float()

        return video.cpu(), {
            "video_path": None,
            "fps": None,
            "total_frames": int(video.shape[0]),
        }

    def _prepare_video_batch(
        self,
        videos: Union[Sequence[Union[torch.Tensor, str, Path]], torch.Tensor],
    ) -> Tuple[List[torch.Tensor], List[Dict[str, Any]]]:
        if isinstance(videos, torch.Tensor):
            if videos.ndim != 5:
                raise ValueError(
                    "Batch video tensor must have shape [B, T, C, H, W]"
                )
            prepared = []
            meta = []
            for idx in range(videos.shape[0]):
                item, item_meta = self._to_video_float_tchw(videos[idx])
                item_meta["index"] = idx
                prepared.append(item)
                meta.append(item_meta)
            return prepared, meta

        if not isinstance(videos, (list, tuple)) or len(videos) == 0:
            raise ValueError(
                "videos must be a non-empty sequence or a [B, T, C, H, W] tensor"
            )

        prepared = []
        meta = []
        for idx, video in enumerate(videos):
            item, item_meta = self._to_video_float_tchw(video)
            item_meta["index"] = idx
            prepared.append(item)
            meta.append(item_meta)
        return prepared, meta

    def _resolve_shared_prompts(
        self,
        prompts: PromptInput,
        video_meta: Sequence[Dict[str, Any]],
        sample_index: Optional[int],
    ) -> List[str]:
        normalized_prompts = self._normalize_prompts(prompts)
        if normalized_prompts is not None:
            return normalized_prompts

        if sample_index is None:
            first_path = video_meta[0].get("video_path")
            if first_path is not None:
                sample_index = self._parse_sample_index_from_path(Path(first_path))

        if sample_index is not None:
            return self._load_prompts_from_jsonl(int(sample_index))

        raise ValueError(
            "prompts are required unless sample_index can be resolved "
            "against prompt_jsonl"
        )

    # ------------------------------------------------------------------
    # Batch scoring (raw scores, no advantage)
    # ------------------------------------------------------------------

    def _score_short_batch(
        self,
        videos_float_tchw: Sequence[torch.Tensor],
        prompt: str,
    ) -> Dict[str, Any]:
        """Score a batch of short videos. Returns raw per-component scores."""
        per_video = [
            {"index": idx, "component_scores": {}}
            for idx in range(len(videos_float_tchw))
        ]

        for comp in self.reward_components:
            scores = [
                self._score_video_with_component(comp["fn"], video, prompt)
                for video in videos_float_tchw
            ]
            for item, score in zip(per_video, scores):
                item["component_scores"][comp["key"]] = score

        for item in per_video:
            item["combined_score"] = sum(
                comp["weight"] * item["component_scores"][comp["key"]]
                for comp in self.reward_components
            )

        return {"mode": "short", "per_video": per_video}

    def _score_long_batch(
        self,
        videos_float_tchw: Sequence[torch.Tensor],
        specs: Sequence[ClipSpec],
    ) -> Dict[str, Any]:
        """Score a batch of long videos via clips. Returns raw per-component scores."""
        per_video = [
            {
                "index": idx,
                "clips": [
                    {
                        "name": spec.name, "group": spec.group,
                        "start_frame": int(spec.start_frame),
                        "end_frame": int(spec.end_frame),
                        "prompt": spec.prompt,
                        "component_scores": {},
                    }
                    for spec in specs
                ],
                "component_scores": {},
            }
            for idx in range(len(videos_float_tchw))
        ]

        for comp in self.reward_components:
            for clip_idx, spec in enumerate(specs):
                clip_scores = []
                for video in videos_float_tchw:
                    clip = video[spec.start_frame:spec.end_frame]
                    clip_scores.append(
                        self._score_video_with_component(
                            comp["fn"], clip, spec.prompt,
                        )
                    )
                for video_idx, score in enumerate(clip_scores):
                    per_video[video_idx]["clips"][clip_idx][
                        "component_scores"
                    ][comp["key"]] = score

            for item in per_video:
                comp_scores = [
                    clip["component_scores"][comp["key"]]
                    for clip in item["clips"]
                ]
                item["component_scores"][comp["key"]] = (
                    sum(comp_scores) / len(comp_scores)
                )

        for item in per_video:
            item["combined_score"] = sum(
                comp["weight"] * item["component_scores"][comp["key"]]
                for comp in self.reward_components
            )

        return {"mode": "long", "per_video": per_video}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate_video(
        self,
        video_path: VideoInput,
        *,
        prompts: PromptInput = None,
        sample_index: Optional[int] = None,
        mode: str = "auto",
        switch_frame_indices: Optional[Union[str, Sequence[int]]] = None,
    ) -> Dict[str, Any]:
        video_path = Path(video_path)
        if not video_path.is_file():
            raise FileNotFoundError(video_path)

        video_frames, _, info = torchvision.io.read_video(
            str(video_path), pts_unit="sec", output_format="TCHW",
        )
        fps = float(info.get("video_fps", 16.0))
        total_frames = int(video_frames.shape[0])
        video_float = video_frames.float() / 255.0

        normalized_prompts = self._normalize_prompts(prompts)
        if normalized_prompts is None:
            if sample_index is None:
                sample_index = self._parse_sample_index_from_path(video_path)
            if sample_index is not None:
                normalized_prompts = self._load_prompts_from_jsonl(int(sample_index))

        if normalized_prompts is None:
            raise ValueError(
                "prompts are required unless sample_index can be resolved "
                "against prompt_jsonl"
            )

        mode = str(mode).strip().lower()
        if mode not in {"auto", "long", "short"}:
            raise ValueError("mode must be one of: auto, long, short")

        if mode == "auto":
            if (
                total_frames <= self.short_video_max_frames
                and len(normalized_prompts) <= 1
            ):
                mode = "short"
            else:
                mode = "long"

        comp_meta = [
            {"key": c["key"], "name": c["name"], "weight": float(c["weight"])}
            for c in self.reward_components
        ]

        if mode == "short":
            component_scores = {}
            for comp in self.reward_components:
                score = self._score_video_with_component(
                    comp["fn"], video_float, normalized_prompts[0],
                )
                component_scores[comp["key"]] = score

            combined_score = sum(
                c["weight"] * component_scores[c["key"]]
                for c in self.reward_components
            )
            return {
                "video_path": str(video_path),
                "fps": fps,
                "total_frames": total_frames,
                "mode": "short",
                "reward_components": comp_meta,
                "component_scores": component_scores,
                "combined_score": combined_score,
            }

        resolved_switches = (
            self.switch_frame_indices if switch_frame_indices is None
            else self._parse_switch_frame_indices(switch_frame_indices)
        )

        if len(normalized_prompts) > 1:
            specs = self._build_prompt_aware_specs(
                total_frames=total_frames,
                prompts=normalized_prompts,
                switch_frame_indices=resolved_switches,
            )
            long_mode = "prompt_aware"
        else:
            specs = self._build_uniform_long_specs(
                total_frames=total_frames, prompt=normalized_prompts[0],
            )
            long_mode = "uniform"

        clip_results, summary = self._score_clips(video_float, specs)
        return {
            "video_path": str(video_path),
            "fps": fps,
            "total_frames": total_frames,
            "mode": "long",
            "long_mode": long_mode,
            "reward_components": comp_meta,
            "clips": clip_results,
            "summary": summary,
        }

    def evaluate_group(
        self,
        videos: Union[Sequence[Union[torch.Tensor, str, Path]], torch.Tensor],
        *,
        prompts: PromptInput = None,
        sample_index: Optional[int] = None,
        mode: str = "auto",
        switch_frame_indices: Optional[Union[str, Sequence[int]]] = None,
    ) -> Dict[str, Any]:
        videos_float_tchw, video_meta = self._prepare_video_batch(videos)
        shared_prompts = self._resolve_shared_prompts(
            prompts, video_meta, sample_index,
        )

        mode = str(mode).strip().lower()
        if mode not in {"auto", "long", "short"}:
            raise ValueError("mode must be one of: auto, long, short")

        first_total_frames = int(videos_float_tchw[0].shape[0])
        if mode == "auto":
            if (
                first_total_frames <= self.short_video_max_frames
                and len(shared_prompts) <= 1
            ):
                mode = "short"
            else:
                mode = "long"

        comp_meta = [
            {"key": c["key"], "name": c["name"], "weight": float(c["weight"])}
            for c in self.reward_components
        ]

        if mode == "short":
            if len(shared_prompts) != 1:
                raise ValueError("Short-video group mode expects a single prompt")
            result = self._score_short_batch(videos_float_tchw, shared_prompts[0])
            return {
                "mode": "short",
                "reward_components": comp_meta,
                "videos": [
                    {**video_meta[idx], **result["per_video"][idx]}
                    for idx in range(len(video_meta))
                ],
            }

        resolved_switches = (
            self.switch_frame_indices if switch_frame_indices is None
            else self._parse_switch_frame_indices(switch_frame_indices)
        )
        if len(shared_prompts) > 1:
            specs = self._build_prompt_aware_specs(
                total_frames=first_total_frames,
                prompts=shared_prompts,
                switch_frame_indices=resolved_switches,
            )
            long_mode = "prompt_aware"
        else:
            specs = self._build_uniform_long_specs(
                total_frames=first_total_frames, prompt=shared_prompts[0],
            )
            long_mode = "uniform"

        result = self._score_long_batch(videos_float_tchw, specs)
        return {
            "mode": "long",
            "long_mode": long_mode,
            "reward_components": comp_meta,
            "videos": [
                {**video_meta[idx], **result["per_video"][idx]}
                for idx in range(len(video_meta))
            ],
        }

    def score_group(
        self,
        videos: Union[Sequence[Union[torch.Tensor, str, Path]], torch.Tensor],
        *,
        segment_prompts: PromptInput = None,
        switch_frame_indices: Optional[Union[str, Sequence[int]]] = None,
        mode: str = "auto",
        sample_index: Optional[int] = None,
    ) -> Tuple[List[float], Dict[str, Any]]:
        group_result = self.evaluate_group(
            videos,
            prompts=segment_prompts,
            sample_index=sample_index,
            mode=mode,
            switch_frame_indices=switch_frame_indices,
        )
        final_scores = [
            video["combined_score"] for video in group_result["videos"]
        ]
        details = {
            "mode": group_result["mode"],
            "reward_components": group_result["reward_components"],
            "videos": group_result["videos"],
        }
        if "long_mode" in group_result:
            details["long_mode"] = group_result["long_mode"]
        return final_scores, details

    def __call__(
        self,
        videos: Union[Sequence[Union[torch.Tensor, str, Path]], torch.Tensor],
        *,
        segment_prompts: PromptInput = None,
        switch_frame_indices: Optional[Union[str, Sequence[int]]] = None,
        mode: str = "auto",
        sample_index: Optional[int] = None,
    ) -> Tuple[List[float], Dict[str, Any]]:
        return self.score_group(
            videos,
            segment_prompts=segment_prompts,
            switch_frame_indices=switch_frame_indices,
            mode=mode,
            sample_index=sample_index,
        )


__all__ = ["VideoRewardAdvantageInterface"]
