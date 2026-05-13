import math

import numpy as np
import torch
import torch.nn as nn

from ... import INSIGHTFACE_ROOT


class FaceConsistencyScorer(nn.Module):
    def __init__(
        self,
        device,
        model_name: str = "buffalo_l",
        model_root: str = INSIGHTFACE_ROOT,
        window_size: int = 4,
        min_face_ratio: float = 0.3,
        det_thresh: float = 0.5,
        det_size: tuple[int, int] = (640, 640),
        max_num_faces: int = 0,
        neutral_score: float = 0.0,
        use_anchor: bool = True,
    ):
        super().__init__()
        self.device = str(device)
        self.model_name = model_name
        self.model_root = model_root
        self.window_size = window_size
        self.min_face_ratio = min_face_ratio
        self.det_thresh = det_thresh
        self.det_size = det_size
        self.max_num_faces = max_num_faces
        self.neutral_score = neutral_score
        self.use_anchor = use_anchor
        self.app = None

    def _ensure_app(self):
        if self.app is not None:
            return

        from insightface.app import FaceAnalysis

        providers = ["CPUExecutionProvider"]
        ctx_id = -1
        if self.device.startswith("cuda"):
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            ctx_id = 0

        app = FaceAnalysis(
            name=self.model_name,
            root=self.model_root,
            allowed_modules=["detection", "recognition"],
            providers=providers,
        )
        app.prepare(ctx_id=ctx_id, det_thresh=self.det_thresh, det_size=self.det_size)
        self.app = app

    @staticmethod
    def _normalize_videos(videos):
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

    @staticmethod
    def _bbox_area(face):
        bbox = np.asarray(face.bbox, dtype=np.float32)
        w = max(0.0, float(bbox[2] - bbox[0]))
        h = max(0.0, float(bbox[3] - bbox[1]))
        return w * h

    @staticmethod
    def _bbox_center(face):
        bbox = np.asarray(face.bbox, dtype=np.float32)
        return np.array([(bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5], dtype=np.float32)

    @staticmethod
    def _cosine(a, b) -> float:
        if a is None or b is None:
            return 0.0
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom <= 1e-12:
            return 0.0
        return float(np.clip(np.dot(a, b) / denom, -1.0, 1.0))

    def _pick_main_face(self, faces, prev_face):
        if not faces:
            return None

        best_face = None
        best_score = -float("inf")
        prev_center = None if prev_face is None else self._bbox_center(prev_face)
        prev_emb = None if prev_face is None else getattr(prev_face, "normed_embedding", None)

        for face in faces:
            det_score = float(getattr(face, "det_score", 0.0))
            area_score = math.log1p(self._bbox_area(face))
            track_bonus = 0.0

            if prev_center is not None:
                center = self._bbox_center(face)
                track_bonus -= 0.001 * float(np.linalg.norm(center - prev_center))

            emb = getattr(face, "normed_embedding", None)
            if prev_emb is not None and emb is not None:
                track_bonus += 0.5 * max(0.0, self._cosine(prev_emb, emb))

            score = det_score + 0.02 * area_score + track_bonus
            if score > best_score:
                best_score = score
                best_face = face

        return best_face

    def _frame_face(self, frame_chw, prev_face):
        frame_hwc = frame_chw.permute(1, 2, 0).numpy()
        faces = self.app.get(frame_hwc, max_num=self.max_num_faces)
        return self._pick_main_face(faces, prev_face)

    def _window_reduce(self, values: list[float]) -> float:
        if not values:
            return self.neutral_score
        if len(values) < self.window_size:
            return float(np.mean(values))
        windows = [
            float(np.mean(values[i : i + self.window_size]))
            for i in range(len(values) - self.window_size + 1)
        ]
        return float(min(windows))

    def _score_video(self, video):
        prev_face = None
        anchor_emb = None
        pair_scores = []
        anchor_scores = []
        visible = []

        for frame in video:
            face = self._frame_face(frame, prev_face)
            visible.append(1.0 if face is not None else 0.0)

            if face is None:
                pair_scores.append(0.0)
                anchor_scores.append(0.0)
                continue

            emb = getattr(face, "normed_embedding", None)
            det_score = float(getattr(face, "det_score", 0.0))
            if emb is not None and anchor_emb is None:
                anchor_emb = emb

            if prev_face is None:
                pair_scores.append(0.0)
            else:
                prev_emb = getattr(prev_face, "normed_embedding", None)
                prev_det = float(getattr(prev_face, "det_score", 0.0))
                pair_scores.append(min(det_score, prev_det) * max(0.0, self._cosine(emb, prev_emb)))

            if self.use_anchor and anchor_emb is not None and emb is not None:
                anchor_scores.append(det_score * max(0.0, self._cosine(emb, anchor_emb)))
            else:
                anchor_scores.append(0.0)

            prev_face = face

        face_ratio = float(np.mean(visible)) if visible else 0.0
        applicable = face_ratio >= self.min_face_ratio

        if not applicable:
            return {
                "score": float(self.neutral_score),
                "face_ratio": face_ratio,
                "applicable": False,
                "pair_score": 0.0,
                "anchor_score": 0.0,
            }

        pair_component = self._window_reduce(pair_scores)
        anchor_component = self._window_reduce(anchor_scores) if self.use_anchor else 0.0
        final_score = 0.7 * pair_component + 0.3 * anchor_component if self.use_anchor else pair_component

        return {
            "score": float(final_score),
            "face_ratio": face_ratio,
            "applicable": True,
            "pair_score": float(pair_component),
            "anchor_score": float(anchor_component),
        }

    @torch.no_grad()
    def __call__(self, videos, prompts=None):
        del prompts
        self._ensure_app()
        videos = self._normalize_videos(videos)

        scores = []
        metadata = {
            "face_ratio": [],
            "applicable": [],
            "pair_score": [],
            "anchor_score": [],
            "window_size": self.window_size,
            "min_face_ratio": self.min_face_ratio,
        }

        for video in videos:
            result = self._score_video(video)
            scores.append(result["score"])
            metadata["face_ratio"].append(result["face_ratio"])
            metadata["applicable"].append(result["applicable"])
            metadata["pair_score"].append(result["pair_score"])
            metadata["anchor_score"].append(result["anchor_score"])

        return torch.tensor(scores, dtype=torch.float32), metadata
