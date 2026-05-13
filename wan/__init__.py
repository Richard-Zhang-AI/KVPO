"""Compatibility shim for the moved WAN package used by memflow."""

from pathlib import Path

__path__ = [
    str(Path(__file__).resolve().parent.parent / "models" / "memflow" / "wan")
]

from . import configs, distributed, modules
from .image2video import WanI2V
from .text2video import WanT2V

__all__ = [
    "configs",
    "distributed",
    "modules",
    "WanI2V",
    "WanT2V",
]

