import os
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn

from ... import VITDET_CKPT_PATH, VITDET_CONFIG_PATH


class ViTDetHumanWorstWindowScorer(nn.Module):
    def __init__(
        self,
        device,
        config_path: str = VITDET_CONFIG_PATH,
        checkpoint_path: str = VITDET_CKPT_PATH,
        window_size: int = 4,
        score_thresh: float = 0.05,
    ):
        super().__init__()
        self.device = str(device)
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.window_size = window_size
        self.score_thresh = score_thresh
        self.predictor = None
        self.person_class_id = 0

    def _ensure_predictor(self):
        if self.predictor is not None:
            return

        if not os.path.exists(self.config_path):
            raise FileNotFoundError(
                f"ViTDet config not found: {self.config_path}. "
                "Set KVPO_VITDET_CONFIG to a valid detectron2 ViTDet config."
            )
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(
                f"ViTDet checkpoint not found: {self.checkpoint_path}. "
                "Set KVPO_VITDET_CKPT to a valid model checkpoint."
            )

        try:
            from detectron2.checkpoint import DetectionCheckpointer
            from detectron2.config import LazyConfig, instantiate
            import detectron2.data.transforms as T
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "detectron2 is not installed. Install detectron2 in the memflow env before "
                "using the ViTDet human scorer."
            ) from exc

        cfg = LazyConfig.load(self.config_path)
        self._assign_if_present(cfg, "train.init_checkpoint", self.checkpoint_path)
        self._assign_if_present(cfg, "model.roi_heads.box_predictor.test_score_thresh", self.score_thresh)
        self._assign_if_present(cfg, "model.roi_heads.box_predictors.0.test_score_thresh", self.score_thresh)
        self._assign_if_present(cfg, "model.roi_heads.box_predictors.1.test_score_thresh", self.score_thresh)
        self._assign_if_present(cfg, "model.roi_heads.box_predictors.2.test_score_thresh", self.score_thresh)

        model = instantiate(cfg.model)
        model.to(self.device)
        model.eval()
        DetectionCheckpointer(model).load(self.checkpoint_path)

        min_size = self._lookup(cfg, "dataloader.test.mapper.augmentations.0.short_edge_length", 1024)
        max_size = self._lookup(cfg, "dataloader.test.mapper.augmentations.0.max_size", 1024)
        if isinstance(min_size, Iterable) and not isinstance(min_size, (str, bytes)):
            min_size = list(min_size)[0]

        augment = T.ResizeShortestEdge(short_edge_length=min_size, max_size=max_size)
        self.predictor = _LazyViTDetPredictor(model=model, augment=augment, device=self.device)

    def _assign_if_present(self, root, dotted_key: str, value):
        parts = dotted_key.split(".")
        cur = root
        for key in parts[:-1]:
            if not self._has_attr_or_key(cur, key):
                return
            cur = self._get_attr_or_key(cur, key)
        leaf = parts[-1]
        if self._has_attr_or_key(cur, leaf):
            self._set_attr_or_key(cur, leaf, value)

    @staticmethod
    def _lookup(root, dotted_key: str, default=None):
        parts = dotted_key.split(".")
        cur = root
        for key in parts:
            if not ViTDetHumanWorstWindowScorer._has_attr_or_key(cur, key):
                return default
            cur = ViTDetHumanWorstWindowScorer._get_attr_or_key(cur, key)
        return cur

    @staticmethod
    def _has_attr_or_key(obj, key: str) -> bool:
        if isinstance(obj, (list, tuple)) and key.isdigit():
            return int(key) < len(obj)
        if isinstance(obj, dict):
            return key in obj
        return hasattr(obj, key)

    @staticmethod
    def _get_attr_or_key(obj, key: str):
        if isinstance(obj, (list, tuple)) and key.isdigit():
            return obj[int(key)]
        if isinstance(obj, dict):
            return obj[key]
        return getattr(obj, key)

    @staticmethod
    def _set_attr_or_key(obj, key: str, value):
        if isinstance(obj, dict):
            obj[key] = value
            return
        setattr(obj, key, value)

    def _normalize_videos(self, videos):
        if not isinstance(videos, torch.Tensor):
            videos = torch.from_numpy(videos)

        if videos.ndim == 4:
            videos = videos.unsqueeze(0)

        if videos.ndim != 5:
            raise ValueError(f"Expected 5D video tensor, got shape {tuple(videos.shape)}")

        if videos.shape[-1] == 3:
            videos = videos.permute(0, 1, 4, 2, 3)

        if videos.dtype != torch.uint8:
            if torch.is_floating_point(videos):
                videos = (videos * 255).round().clamp(0, 255).to(torch.uint8)
            else:
                videos = videos.to(torch.uint8)

        return videos.cpu()

    def _frame_person_score(self, frame_chw: torch.Tensor) -> float:
        frame_hwc = frame_chw.permute(1, 2, 0).numpy()
        outputs = self.predictor(frame_hwc)
        instances = outputs["instances"].to("cpu")

        if len(instances) == 0:
            return 0.0

        pred_classes = instances.pred_classes.numpy()
        scores = instances.scores.numpy()
        person_scores = scores[pred_classes == self.person_class_id]
        if person_scores.size == 0:
            return 0.0
        return float(person_scores.max())

    def _worst_window_score(self, frame_scores: list[float]) -> float:
        if not frame_scores:
            return 0.0

        if len(frame_scores) < self.window_size:
            return float(sum(frame_scores))

        windows = [
            sum(frame_scores[start : start + self.window_size])
            for start in range(len(frame_scores) - self.window_size + 1)
        ]
        return float(min(windows))

    @torch.no_grad()
    def __call__(self, videos, prompts=None):
        del prompts
        self._ensure_predictor()
        videos = self._normalize_videos(videos)

        results = []
        for video in videos:
            frame_scores = [self._frame_person_score(frame) for frame in video]
            results.append(self._worst_window_score(frame_scores))

        return torch.tensor(results, dtype=torch.float32)


class _LazyViTDetPredictor:
    def __init__(self, model, augment, device: str):
        self.model = model
        self.augment = augment
        self.device = device

    def __call__(self, original_image: np.ndarray):
        if original_image.ndim != 3 or original_image.shape[2] != 3:
            raise ValueError(f"Expected HWC RGB image, got shape {original_image.shape}")

        height, width = original_image.shape[:2]
        image = original_image[:, :, ::-1].copy()
        image = self.augment.get_transform(image).apply_image(image)
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1), device=self.device)

        inputs = {"image": image, "height": height, "width": width}
        return self.model([inputs])[0]
