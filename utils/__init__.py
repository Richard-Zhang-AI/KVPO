"""Compatibility shim for the moved memflow utils package."""

from pathlib import Path

__path__ = [
    str(Path(__file__).resolve().parent.parent / "models" / "memflow" / "utils")
]
