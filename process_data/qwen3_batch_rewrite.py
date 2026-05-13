#!/usr/bin/env python3
"""Batch rewrite T2V prompts with Qwen3.5 across multiple GPUs."""

from __future__ import annotations

import argparse
import multiprocessing as mp
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
from tqdm import tqdm

from qwen3_prompt_optimize import (
    DEFAULT_MODEL_DIR,
    DEFAULT_OPTIMIZER_SYSTEM_PROMPT,
    build_optimizer_user_prompt,
    ensure_model_supported,
    move_inputs_to_device,
    strip_reasoning_content,
)


DEFAULT_INPUT_FILE = Path(
    "prompts/vidprom_filtered_extended_switch.txt"
)
DEFAULT_OUTPUT_FILE = Path(
    "prompts/vidprom_filtered_extended_switch_qwen3_rewritten.txt"
)


@dataclass
class WorkerConfig:
    rank: int
    gpu_id: int
    num_workers: int
    model_dir: str
    input_file: str
    parts_dir: str
    system_prompt: str
    batch_size: int
    max_new_tokens: int
    temperature: float
    top_p: float
    top_k: int
    presence_penalty: float
    repetition_penalty: float
    limit: int
    log_every: int
    resume: bool
    dtype: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch rewrite prompts from a txt file with Qwen3.5 on multiple GPUs."
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        default=DEFAULT_INPUT_FILE,
        help="Source txt file. One prompt per line.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
        help="Merged rewritten output txt file.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="Local Hugging Face model directory.",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="0,1,2,3,4,5,6,7",
        help='Comma-separated GPU ids, for example "0,1,2,3,4,5,6,7".',
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Per-GPU micro-batch size.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=2048,
        help="Maximum new tokens per prompt.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.8,
        help="Nucleus sampling top-p.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Top-k sampling.",
    )
    parser.add_argument(
        "--presence-penalty",
        type=float,
        default=1.5,
        help="Presence penalty for reducing repetition.",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.0,
        help="Repetition penalty.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Only process the first N lines. 0 means all lines.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=50,
        help="Print one progress log every N generated batches per worker.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing shard files if present.",
    )
    parser.add_argument(
        "--merge-only",
        action="store_true",
        help="Skip generation and only merge existing shard files.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["auto", "bfloat16", "float16", "float32"],
        help="Torch dtype used to load the model.",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=DEFAULT_OPTIMIZER_SYSTEM_PROMPT,
        help="System prompt used for rewriting.",
    )
    return parser.parse_args()


def parse_gpu_ids(gpu_text: str) -> list[int]:
    ids = [int(part.strip()) for part in gpu_text.split(",") if part.strip()]
    if not ids:
        raise ValueError("`--gpus` must not be empty.")
    return ids


def resolve_dtype(dtype_name: str):
    if dtype_name == "auto":
        return "auto"
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def count_lines(path: Path, limit: int) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for _ in handle:
            count += 1
            if limit and count >= limit:
                break
    return count


def count_worker_lines(total_lines: int, rank: int, num_workers: int) -> int:
    if total_lines <= rank:
        return 0
    return ((total_lines - 1 - rank) // num_workers) + 1


def normalize_output_line(text: str) -> str:
    text = strip_reasoning_content(text)
    return re.sub(r"\s+", " ", text).strip()


def shard_path(parts_dir: Path, rank: int) -> Path:
    return parts_dir / f"part_{rank:02d}.tsv"


def completed_records(part_file: Path) -> int:
    if not part_file.exists():
        return 0
    with part_file.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def iter_worker_prompts(
    input_file: Path,
    rank: int,
    num_workers: int,
    limit: int,
    skip_completed: int,
) -> Iterable[tuple[int, str]]:
    seen_for_rank = 0
    with input_file.open("r", encoding="utf-8") as handle:
        for idx, raw_line in enumerate(handle):
            if limit and idx >= limit:
                break
            if idx % num_workers != rank:
                continue
            if seen_for_rank < skip_completed:
                seen_for_rank += 1
                continue
            yield idx, raw_line.rstrip("\n")


def build_conversations(prompts: list[str], system_prompt: str) -> list[list[dict]]:
    conversations = []
    for prompt in prompts:
        conversations.append(
            [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_prompt}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": build_optimizer_user_prompt(prompt)}
                    ],
                },
            ]
        )
    return conversations


def load_model_and_processor(
    model_dir: Path,
    gpu_id: int,
    dtype_name: str,
) -> tuple[AutoProcessor, Qwen3_5ForConditionalGeneration]:
    ensure_model_supported(model_dir)

    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
    tokenizer = processor.tokenizer
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = resolve_dtype(dtype_name)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_dir,
        torch_dtype=torch_dtype,
        device_map={"": gpu_id},
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    return processor, model


def generate_batch(
    processor: AutoProcessor,
    model: Qwen3_5ForConditionalGeneration,
    prompts: list[str],
    system_prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    presence_penalty: float,
    repetition_penalty: float,
) -> list[str]:
    tokenizer = processor.tokenizer
    conversations = build_conversations(prompts, system_prompt)
    inputs = tokenizer.apply_chat_template(
        conversations,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
        enable_thinking=False,
    )

    target_device = next(model.parameters()).device
    inputs = move_inputs_to_device(inputs, target_device)

    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "top_p": top_p,
        "top_k": top_k,
        "repetition_penalty": repetition_penalty,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if temperature > 0:
        generation_kwargs["temperature"] = temperature

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, **generation_kwargs)

    prompt_length = inputs["input_ids"].shape[1]
    decoded = tokenizer.batch_decode(
        generated_ids[:, prompt_length:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return [normalize_output_line(text) for text in decoded]


def flush_batch(
    part_handle,
    processor: AutoProcessor,
    model: Qwen3_5ForConditionalGeneration,
    batch: list[tuple[int, str]],
    config: WorkerConfig,
    progress: tqdm | None = None,
) -> None:
    prompts = [prompt for _, prompt in batch]
    batch_size = len(batch)
    started_at = time.time()

    if progress is not None:
        progress.set_postfix_str(f"stage=generate bsz={batch_size}")
        progress.refresh()

    outputs = generate_batch(
        processor=processor,
        model=model,
        prompts=prompts,
        system_prompt=config.system_prompt,
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        top_k=config.top_k,
        presence_penalty=config.presence_penalty,
        repetition_penalty=config.repetition_penalty,
    )

    if progress is not None:
        progress.set_postfix_str(
            f"stage=write bsz={batch_size} took={time.time() - started_at:.1f}s"
        )
        progress.refresh()

    for (idx, _), output in zip(batch, outputs):
        part_handle.write(f"{idx}\t{output}\n")
    part_handle.flush()

    if progress is not None:
        progress.set_postfix_str(
            f"stage=idle last_bsz={batch_size} last={time.time() - started_at:.1f}s"
        )
        progress.refresh()


def worker_main(config: WorkerConfig) -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.set_device(config.gpu_id)

    model_dir = Path(config.model_dir)
    input_file = Path(config.input_file)
    parts_dir = Path(config.parts_dir)
    part_file = shard_path(parts_dir, config.rank)

    already_done = completed_records(part_file) if config.resume else 0
    file_mode = "a" if config.resume else "w"
    total_lines = count_lines(input_file, config.limit)
    worker_total = count_worker_lines(total_lines, config.rank, config.num_workers)
    remaining = max(worker_total - already_done, 0)

    start_time = time.time()
    processor, model = load_model_and_processor(
        model_dir=model_dir,
        gpu_id=config.gpu_id,
        dtype_name=config.dtype,
    )

    batch: list[tuple[int, str]] = []
    processed = already_done
    generated_batches = 0

    progress = tqdm(
        total=remaining,
        initial=0,
        desc=f"gpu{config.gpu_id}",
        position=config.rank,
        dynamic_ncols=True,
        leave=True,
        unit="prompt",
    )

    try:
        with part_file.open(file_mode, encoding="utf-8") as handle:
            for idx, prompt in iter_worker_prompts(
                input_file=input_file,
                rank=config.rank,
                num_workers=config.num_workers,
                limit=config.limit,
                skip_completed=already_done,
            ):
                batch.append((idx, prompt))
                if len(batch) < config.batch_size:
                    progress.set_postfix_str(f"stage=collect batch={len(batch)}/{config.batch_size}")
                    continue

                flush_batch(handle, processor, model, batch, config, progress)
                processed += len(batch)
                generated_batches += 1
                progress.update(len(batch))
                batch.clear()

                if config.log_every and generated_batches % config.log_every == 0:
                    elapsed = time.time() - start_time
                    progress.set_postfix(
                        processed=processed,
                        batches=generated_batches,
                        elapsed=f"{elapsed:.1f}s",
                    )

            if batch:
                flush_batch(handle, processor, model, batch, config, progress)
                processed += len(batch)
                progress.update(len(batch))
    finally:
        progress.close()

    elapsed = time.time() - start_time
    print(
        f"[worker {config.rank} | gpu {config.gpu_id}] done processed={processed} elapsed={elapsed:.1f}s",
        flush=True,
    )


def merge_parts(parts_dir: Path, output_file: Path, num_workers: int, total_lines: int) -> None:
    part_handles = [
        shard_path(parts_dir, rank).open("r", encoding="utf-8")
        for rank in range(num_workers)
    ]
    try:
        with output_file.open("w", encoding="utf-8") as out_handle:
            with tqdm(
                total=total_lines,
                desc="merge",
                position=num_workers,
                dynamic_ncols=True,
                leave=True,
                unit="prompt",
            ) as progress:
                for idx in range(total_lines):
                    rank = idx % num_workers
                    line = part_handles[rank].readline()
                    if not line:
                        raise RuntimeError(
                            f"Shard file is incomplete: {shard_path(parts_dir, rank)} ended early at index {idx}."
                        )
                    current_idx_str, rewritten = line.rstrip("\n").split("\t", 1)
                    current_idx = int(current_idx_str)
                    if current_idx != idx:
                        raise RuntimeError(
                            f"Shard order mismatch: expected index {idx}, got {current_idx}."
                        )
                    out_handle.write(rewritten + "\n")
                    progress.update(1)

        for rank, handle in enumerate(part_handles):
            remainder = handle.readline()
            if remainder:
                raise RuntimeError(
                    f"Shard file contains extra records: {shard_path(parts_dir, rank)}"
                )
    finally:
        for handle in part_handles:
            handle.close()


def main() -> int:
    args = parse_args()

    input_file = args.input_file.resolve()
    output_file = args.output_file.resolve()
    model_dir = args.model_dir.resolve()
    parts_dir = output_file.parent / f"{output_file.stem}.parts"
    gpu_ids = parse_gpu_ids(args.gpus)

    if not input_file.exists():
        print(f"[ERROR] Input file does not exist: {input_file}", file=sys.stderr)
        return 1
    if not model_dir.exists():
        print(f"[ERROR] Model directory does not exist: {model_dir}", file=sys.stderr)
        return 1

    if args.presence_penalty != 0:
        print(
            "[WARN] Local Hugging Face generate() does not support presence_penalty; it will be ignored.",
            flush=True,
        )

    generation_load = args.batch_size * args.max_new_tokens
    if generation_load >= 8192:
        print(
            "[WARN] Current settings are likely very slow on the local fallback path: "
            f"batch_size={args.batch_size}, max_new_tokens={args.max_new_tokens}. "
            "If the first batch appears stuck, try --batch-size 2 or 4 and/or --max-new-tokens 512 or 1024.",
            flush=True,
        )

    total_lines = count_lines(input_file, args.limit)
    parts_dir.mkdir(parents=True, exist_ok=True)

    if not args.merge_only:
        worker_configs = [
            WorkerConfig(
                rank=rank,
                gpu_id=gpu_id,
                num_workers=len(gpu_ids),
                model_dir=str(model_dir),
                input_file=str(input_file),
                parts_dir=str(parts_dir),
                system_prompt=args.system_prompt,
                batch_size=args.batch_size,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                presence_penalty=args.presence_penalty,
                repetition_penalty=args.repetition_penalty,
                limit=args.limit,
                log_every=args.log_every,
                resume=args.resume,
                dtype=args.dtype,
            )
            for rank, gpu_id in enumerate(gpu_ids)
        ]

        torch.multiprocessing.spawn(
            _spawn_entry,
            args=(worker_configs,),
            nprocs=len(worker_configs),
            join=True,
        )

    merge_parts(
        parts_dir=parts_dir,
        output_file=output_file,
        num_workers=len(gpu_ids),
        total_lines=total_lines,
    )
    print(f"Merged output written to: {output_file}")
    return 0


def _spawn_entry(local_rank: int, worker_configs: list[WorkerConfig]) -> None:
    worker_main(worker_configs[local_rank])


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    raise SystemExit(main())

#   python process_data/qwen3_batch_rewrite.py \
#     --gpus 0,1,2,3,4,5,6,7 \
#     --batch-size 8 \
#     --resume
