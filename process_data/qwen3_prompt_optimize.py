#!/usr/bin/env python3
"""Use a local Qwen3.5 checkpoint to optimize prompts."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import torch
from transformers import AutoConfig, AutoProcessor, Qwen3_5ForConditionalGeneration


DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "qwen3"
DEFAULT_TEST_PROMPT = (
    "A quaint A-frame cottage made of wood and glass, nestled off the coast of Mexico. The cottage sits on a small peninsula surrounded by crystal-clear turquoise waters and sandy beaches. The exterior is crafted from weathered wooden planks and large glass windows, offering panoramic views of the ocean. A figure steps onto the porch, closing the screen door behind them with a soft click. Palm trees sway gently in the background, and colorful Mexican flowers adorn the front of the cottage. The sun sets in the distance, casting a warm golden glow over the scene. wide slow-pan"
    )
DEFAULT_OPTIMIZER_SYSTEM_PROMPT = """
You are a professional video script writer specializing in multi-scene text-to-video dataset optimization. Your task is to rewrite and expand the given multi-scene video descriptions according to the following strict rules:

[Creative Rewriting Freedom]
- You are NOT required to follow the original description literally. You may adjust, remove, reorder, or replace any part of the original content as long as the overall theme and setting are preserved.
- Your goal is to produce a cinematically coherent, richly detailed sequence — prioritize visual logic and narrative flow over fidelity to the source text.

[Scene Expansion]
- Each scene must be broken into 3–5 comma-separated clauses, each representing a distinct visual moment or camera beat within a single continuous, uninterrupted shot.
- Each scene description must be expanded to at least 3× its original length, with rich visual details covering: subject appearance, clothing texture, lighting conditions, color palette, spatial depth, ambient atmosphere, and sound cues.
- Each scene should read like a detailed cinematographic shot list entry — precise, immersive, and visual — not a narrative summary.
- Avoid vague adjectives like "beautiful", "nice", or "stunning". Replace with specific sensory observations (e.g., "the amber afternoon light casts long diagonal shadows across the cracked terracotta tiles").

[Camera Continuity — CRITICAL]
- The camera must never cut abruptly between scenes. All transitions must be motivated and physically continuous.
- Preferred transition methods: slow pan, tilt, dolly, crane drift, rack focus pull, or a character/object physically leading the camera into the next space.
- The camera may remain completely static while subjects move within the frame — this is equally valid as a transition method.
- Forbidden: any description implying a hard cut, a jump to a new location, or a sudden unexplained change in camera position or angle.
- Each scene must feel like a seamless continuation of the previous shot — the viewer should never feel the camera "teleported".

[Entity Introduction Protocol — MANDATORY AND STRICTLY ENFORCED]
- ⚠️ Any new person, animal, or object that appears in a scene MUST be explicitly and physically introduced. This is the most critical rule. Violating it will cause the video generation model to hallucinate entities appearing from nowhere.
- New person: you MUST describe the exact direction and physical manner of their entry (e.g., "a man in a grey coat walks into the frame from the left edge, pushing open a metal gate as he enters, his boots crunching on the gravel path").
- New animal: you MUST describe how it enters the visible space (e.g., "a stray cat drops silently from the top of a low wall at the right edge of the frame, landing softly on the cobblestones and pausing to sniff the air").
- New object: you MUST describe the physical action that brings it into frame (e.g., "she reaches into the worn canvas bag resting at her feet and slowly draws out a folded map, spreading it flat on the table surface with both hands").
- Objects that were "always there" but not yet mentioned must be introduced via character interaction or a deliberate camera reveal — they cannot simply appear in description without visual justification.
- If you cannot naturally introduce a new entity, do not include it. Omission is preferable to an unexplained appearance.

[Cross-Scene Story Consistency]
- Maintain strict continuity of: character clothing, hairstyle, accessories, and emotional state; object positions and conditions; time of day, weather, and ambient light direction; narrative logic and cause-effect relationships.
- Each scene must be a direct consequence of or continuation from the previous one.

[Camera Movement Rules]
- Each scene must end with exactly one camera movement instruction (e.g., slow dolly-in, static wide shot, handheld follow, overhead crane shot, rack focus pull).
- This final instruction must not contradict any camera motion already described within the scene body.

[Output Format]
- Separate scenes with a period (.).
- Within each scene, separate the 3–5 continuous action clauses with a comma (,).
- Do not add scene numbers, labels, headers, or any meta-commentary.
- Output only the rewritten descriptions, nothing else.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load the local Qwen3.5 model and optimize a prompt."
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="Local Hugging Face model directory. Default: process_data/qwen3",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=DEFAULT_TEST_PROMPT,
        help="Prompt to optimize. Defaults to an embedded test prompt.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=2048,
        help="Maximum number of generated tokens.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature. Set to 0 for greedy decoding.",
    )
    parser.add_argument(
        "--device-map",
        type=str,
        default="auto",
        help='Device map passed to from_pretrained, e.g. "auto" or "cuda:0".',
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=DEFAULT_OPTIMIZER_SYSTEM_PROMPT,
        help="System prompt that defines the prompt optimizer behavior.",
    )
    return parser.parse_args()


def ensure_model_supported(model_dir: Path) -> None:
    AutoConfig.from_pretrained(model_dir, local_files_only=True)


def build_optimizer_user_prompt(raw_prompt: str) -> str:
    return f"""Please polish the following raw text-to-video prompt.

Raw prompt:
\"\"\"
{raw_prompt}
\"\"\"

Return only the final polished T2V prompt."""


def strip_reasoning_content(text: str) -> str:
    text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL | re.IGNORECASE)

    for label in ("Final Prompt:", "Optimized Prompt:"):
        if label in text:
            text = text.split(label, 1)[1].strip()

    if "Thinking Process:" in text and "\n\n" in text:
        text = text.split("\n\n", 1)[1].strip()

    return text.strip()


def move_inputs_to_device(inputs: dict, device) -> dict:
    moved = {}
    for key, value in inputs.items():
        moved[key] = value.to(device) if hasattr(value, "to") else value
    return moved


def generate_optimized_prompt(args: argparse.Namespace) -> str:
    model_dir = args.model_dir.resolve()
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory does not exist: {model_dir}")

    ensure_model_supported(model_dir)

    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_dir,
        torch_dtype="auto",
        device_map=args.device_map,
        local_files_only=True,
    )

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": args.system_prompt}],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": build_optimizer_user_prompt(args.prompt)}],
        },
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=False,
    )

    target_device = next(model.parameters()).device

    inputs = move_inputs_to_device(inputs, target_device)

    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.temperature > 0,
    }
    if args.temperature > 0:
        generation_kwargs["temperature"] = args.temperature

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, **generation_kwargs)

    trimmed_ids = generated_ids[:, inputs["input_ids"].shape[1] :]
    output_text = processor.batch_decode(
        trimmed_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return strip_reasoning_content(output_text)


def main() -> int:
    args = parse_args()
    try:
        result = generate_optimized_prompt(args)
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    print("=== Test Prompt ===")
    print(args.prompt)
    print()
    print("=== Optimized Result ===")
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
