"""Compatibility shim for the moved memflow trainer package."""

from pathlib import Path

__path__ = [
    str(Path(__file__).resolve().parent.parent / "models" / "memflow" / "trainer")
]

from .distillation import Trainer as ScoreDistillationTrainer

__all__ = [
    "ScoreDistillationTrainer",
]

