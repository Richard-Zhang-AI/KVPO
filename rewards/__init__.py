import os

REWARDS_ROOT = os.path.dirname(os.path.abspath(__file__))
KVPO_ROOT = os.path.dirname(REWARDS_ROOT)
CKPT_PATH = os.path.join(KVPO_ROOT, "checkpoints")
SEA_RAFT_CKPT_PATH = os.path.join(CKPT_PATH, "sea-raft", "model.safetensor")
HPSV3_ROOT = os.path.join(CKPT_PATH, "HPSv3")
HPSV3_CKPT_PATH = os.path.join(HPSV3_ROOT, "HPSv3.safetensors")
HPSV3_BASE_MODEL_PATH = os.environ.get(
    "KVPO_HPSV3_BASE_MODEL",
    os.path.join(HPSV3_ROOT, "Qwen2-VL-7B-Instruct"),
)
VIDEOALIGN_CKPT_PATH = os.path.join(CKPT_PATH, "VideoReward")
DETECTRON2_ROOT = os.path.join(REWARDS_ROOT, "reward_models", "detectron2")
VITDET_CONFIG_PATH = os.environ.get(
    "KVPO_VITDET_CONFIG",
    os.path.join(
        DETECTRON2_ROOT,
        "projects",
        "ViTDet",
        "configs",
        "COCO",
        "cascade_mask_rcnn_vitdet_b_100ep.py",
    ),
)
VITDET_CKPT_PATH = os.environ.get(
    "KVPO_VITDET_CKPT",
    os.path.join(CKPT_PATH, "vitdet", "model_final.pkl"),
)
INSIGHTFACE_ROOT = os.environ.get(
    "KVPO_INSIGHTFACE_ROOT",
    os.path.join(CKPT_PATH, "insightface"),
)

from .advantage_interface import VideoRewardAdvantageInterface

__all__ = [
    "CKPT_PATH",
    "DETECTRON2_ROOT",
    "HPSV3_BASE_MODEL_PATH",
    "HPSV3_CKPT_PATH",
    "HPSV3_ROOT",
    "INSIGHTFACE_ROOT",
    "KVPO_ROOT",
    "REWARDS_ROOT",
    "SEA_RAFT_CKPT_PATH",
    "VideoRewardAdvantageInterface",
    "VIDEOALIGN_CKPT_PATH",
    "VITDET_CONFIG_PATH",
    "VITDET_CKPT_PATH",
]
