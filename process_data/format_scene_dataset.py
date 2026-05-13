#!/usr/bin/env python3
"""Format rewritten video prompts into `n,[scene1],[scene2],...` lines."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from tqdm import tqdm


DEFAULT_INPUT_FILE = Path(
    "prompts/vidprom_filtered_extended_switch_2.0.txt"
)
DEFAULT_OUTPUT_FILE = Path(
    "prompts/vidprom_filtered_extended_switch_2.0_formatted.txt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Format prompt dataset into `n,[scene1],[scene2],...`."
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        default=DEFAULT_INPUT_FILE,
        help="Input txt file, one prompt per line.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
        help="Output txt file.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Only process the first N lines. 0 means all lines.",
    )
    return parser.parse_args()


def count_lines(path: Path, limit: int) -> int:
    total = 0
    with path.open("r", encoding="utf-8") as handle:
        for _ in handle:
            total += 1
            if limit and total >= limit:
                break
    return total


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def is_camera_tail(text: str) -> bool:
    words = text.lower().split()
    camera_keywords = {
        "slow",
        "fast",
        "static",
        "wide",
        "close-up",
        "close",
        "medium",
        "low-angle",
        "high-angle",
        "pan",
        "tilt",
        "zoom",
        "dolly",
        "pull",
        "push",
        "track",
        "tracking",
        "handheld",
        "crane",
        "shot",
        "rack",
        "focus",
        "hold",
        "aerial",
        "bird's-eye",
    }
    if not text or len(words) > 8:
        return False
    return any(word in camera_keywords for word in words)


def split_scenes(prompt: str) -> list[str]:
    parts = [part.strip() for part in prompt.split(".")]
    scenes = [normalize_whitespace(part) for part in parts if normalize_whitespace(part)]
    if len(scenes) >= 2 and is_camera_tail(scenes[-1]):
        scenes[-2] = normalize_whitespace(f"{scenes[-2]}, {scenes[-1]}")
        scenes.pop()
    return scenes


def format_prompt(prompt: str) -> str:
    scenes = split_scenes(prompt)
    if not scenes:
        return "0"
    scene_fields = [f"[{scene}]" for scene in scenes]
    return f"{len(scenes)},{','.join(scene_fields)}"


def main() -> int:
    args = parse_args()
    input_file = args.input_file.resolve()
    output_file = args.output_file.resolve()

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    total = count_lines(input_file, args.limit)

    with input_file.open("r", encoding="utf-8") as in_handle, output_file.open(
        "w", encoding="utf-8"
    ) as out_handle:
        for idx, raw_line in enumerate(
            tqdm(in_handle, total=total, desc="format", unit="prompt", dynamic_ncols=True)
        ):
            if args.limit and idx >= args.limit:
                break
            prompt = raw_line.rstrip("\n")
            out_handle.write(format_prompt(prompt) + "\n")

    print(f"Formatted dataset written to: {output_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
