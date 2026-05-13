"""Diversity sampling via KV local perturbation (interactive multi-prompt mode).

For each prompt sequence, this script generates a long video chunk-by-chunk,
switching prompts at specified frame indices.  At a designated chunk it branches
into K variants, each with a different random selection of older frames in the
local attention window, producing K diverse continuations from the same prefix.
All random selections are recorded so that the results are fully reproducible.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List

import torch
from einops import rearrange
from omegaconf import OmegaConf
from torchvision.io import write_video

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.diversity_sampling import DiversitySamplingPipeline
from models.memflow.utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper
from models.memflow.utils.misc import set_seed
from models.memflow.utils.memory import DynamicSwapInstaller
from models.memflow.utils.prompt_sequences import load_prompt_sequences, resolve_switch_frame_indices


def load_jsonl_prompts(path: str, max_n: int = -1):
    """Load multi-prompt sequences from JSONL or formatted text."""
    return load_prompt_sequences(path, max_n=max_n)


def build_pipeline(config, device, dtype):
    """Build the generator, text encoder, VAE and diversity pipeline."""
    model_kwargs = OmegaConf.to_container(config.model_kwargs, resolve=True)

    generator = WanDiffusionWrapper(**model_kwargs, is_causal=True)
    generator.model.requires_grad_(False)

    if config.generator_ckpt:
        ckpt = torch.load(config.generator_ckpt, map_location="cpu")
        if "generator" in ckpt or "generator_ema" in ckpt:
            raw = ckpt.get("generator_ema" if getattr(config, "use_ema", False) else "generator", ckpt.get("generator"))
        elif "model" in ckpt:
            raw = ckpt["model"]
        else:
            raise ValueError(f"Cannot find generator weights in {config.generator_ckpt}")
        generator.load_state_dict(raw)
        print(f"Loaded generator checkpoint from {config.generator_ckpt}")

    adapter_cfg = getattr(config, "adapter", None)
    if adapter_cfg:
        from models.memflow.utils.lora_utils import configure_lora_for_model
        import peft
        generator.model = configure_lora_for_model(
            generator.model, model_name="generator",
            lora_config=adapter_cfg, is_main_process=True)
        lora_ckpt = getattr(config, "lora_ckpt", None)
        if lora_ckpt:
            lc = torch.load(lora_ckpt, map_location="cpu")
            if isinstance(lc, dict) and "generator_lora" in lc:
                peft.set_peft_model_state_dict(generator.model, lc["generator_lora"])
            else:
                peft.set_peft_model_state_dict(generator.model, lc)
            print(f"Loaded LoRA weights from {lora_ckpt}")

    text_encoder = WanTextEncoder()
    text_encoder.requires_grad_(False)

    vae = WanVAEWrapper()
    vae.requires_grad_(False)

    scheduler = generator.get_scheduler()
    denoising_step_list = torch.tensor(config.denoising_step_list, dtype=torch.long)
    if getattr(config, "warp_denoising_step", False):
        timesteps = torch.cat((scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
        denoising_step_list = timesteps[1000 - denoising_step_list]

    local_attn_size = model_kwargs.get("local_attn_size", 12)
    bank_size = model_kwargs.get("bank_size", 3)
    record_interval = model_kwargs.get("record_interval", 3)
    sink_size = model_kwargs.get("sink_size", 3)
    m_nearest = getattr(config, "m_nearest_frames", 6)

    pipeline = DiversitySamplingPipeline(
        denoising_step_list=denoising_step_list,
        scheduler=scheduler,
        generator=generator,
        num_frame_per_block=getattr(config, "num_frame_per_block", 3),
        same_step_across_blocks=getattr(config, "same_step_across_blocks", True),
        last_step_only=getattr(config, "last_step_only", False),
        context_noise=getattr(config, "context_noise", 0),
        local_attn_size=local_attn_size,
        slice_last_frames=getattr(config, "slice_last_frames", 21),
        bank_size=bank_size,
        record_interval=record_interval,
        m_nearest_frames=m_nearest,
    )

    generator.to(dtype=dtype, device=device)
    vae.to(dtype=dtype, device=device)
    DynamicSwapInstaller.install_model(text_encoder, device=device)

    return pipeline, text_encoder, vae, scheduler, sink_size


def _get_cond_for_frame(cond_list, switch_frame_indices, current_frame):
    """Return the segment index and conditional_dict for *current_frame*."""
    seg = 0
    for idx in switch_frame_indices:
        if current_frame >= idx:
            seg += 1
        else:
            break
    return seg, cond_list[seg]


def _run_chunks(pipeline, chunks, start_ci, end_ci, current_length,
                cond_list, switch_frame_indices, global_sink,
                prev_seg_idx, collected_latents,
                rng_noise, device, dtype,
                perturb_ci=None, branch_plan=None,
                target_abs_frame=None, target_end_frame=None):
    """Generate chunks ``[start_ci, end_ci)`` and handle prompt switches.

    Returns (current_length, seg_idx, collected_latents_new_part).
    """
    seg_idx = prev_seg_idx
    new_parts = []
    for ci in range(start_ci, end_ci):
        chunk_info = chunks[ci]
        new_frames = chunk_info["new_frames"]
        if new_frames <= 0:
            continue

        new_seg, cond = _get_cond_for_frame(cond_list, switch_frame_indices, current_length)
        if new_seg != seg_idx:
            print(f"    [Switch] prompt segment {seg_idx} -> {new_seg} at frame {current_length}")
            all_so_far = collected_latents + new_parts
            pipeline.recache_after_switch(all_so_far, current_length, cond, global_sink)
            seg_idx = new_seg

        noise = torch.randn(
            [1, new_frames, 16, 60, 104],
            generator=rng_noise, device=device, dtype=dtype,
        )

        is_perturb = (ci == perturb_ci) and (branch_plan is not None)
        if is_perturb:
            pipeline.activate_perturbation(
                branch_plan,
                target_abs_frame=target_abs_frame,
                target_end_frame=target_end_frame,
            )
            print(f"  Chunk {ci} [PERTURB frames {target_abs_frame}..{target_end_frame}]: "
                  f"frames {current_length}..{current_length + new_frames}")
        else:
            print(f"  Chunk {ci}: frames {current_length}..{current_length + new_frames}")

        output = pipeline.generate_chunk_sampling(
            noise=noise,
            conditional_dict=cond,
            current_start_frame=current_length,
        )

        if is_perturb:
            pipeline.deactivate_perturbation()

        new_parts.append(output.cpu())
        current_length += new_frames

    return current_length, seg_idx, new_parts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument(
        "--sample_index_start",
        type=int,
        default=None,
        help="Inclusive start index into the dataset file (0-based). Overrides config sample_index_start.",
    )
    parser.add_argument(
        "--sample_index_end",
        type=int,
        default=None,
        help="Inclusive end index into the dataset file (0-based). Overrides config sample_index_end.",
    )
    args = parser.parse_args()

    config = OmegaConf.load(args.config_path)
    default_config = OmegaConf.load("configs/default_config.yaml")
    config = OmegaConf.merge(default_config, config)

    device = torch.device("cuda")
    dtype = torch.bfloat16
    base_seed = getattr(config, "seed", 42)
    set_seed(base_seed)
    torch.set_grad_enabled(False)

    N = getattr(config, "N", 1)
    K = getattr(config, "K", 4)
    sample_idx_offset = getattr(config, "sample_idx_offset", 0)
    idx_start = args.sample_index_start
    if idx_start is None:
        idx_start = OmegaConf.select(config, "sample_index_start")
    idx_end = args.sample_index_end
    if idx_end is None:
        idx_end = OmegaConf.select(config, "sample_index_end")
    use_index_range = idx_start is not None or idx_end is not None
    chunk_size = getattr(config, "streaming_chunk_size", 21)
    max_length = getattr(config, "streaming_max_length", 240)
    min_new_frame = getattr(config, "streaming_min_new_frame", 18)
    perturb_x_chunks = getattr(config, "perturb_within_first_x_chunks", 3)
    perturb_P = getattr(config, "perturb_num_blocks", 3)
    output_folder = getattr(config, "output_folder", "videos/diversity_samples")
    global_sink = getattr(config, "global_sink", True)
    os.makedirs(output_folder, exist_ok=True)

    switch_frame_indices_raw = getattr(config, "switch_frame_indices", "40,80,120,160,200")

    print("=" * 60)
    print("Diversity Sampling Configuration (interactive multi-prompt)")
    print(f"  N (num samples)           = {N}")
    if use_index_range:
        print(f"  sample_index range        = [{idx_start}, {idx_end}] (inclusive, into data file)")
    print(f"  K (branches/sample)       = {K}")
    print(f"  m (nearest frames)        = {getattr(config, 'm_nearest_frames', 4)}")
    print(f"  local_attn_size           = {config.model_kwargs.local_attn_size}")
    print(f"  chunk_size                = {chunk_size}")
    print(f"  max_length                = {max_length}")
    print(f"  perturb_within_first_x    = {perturb_x_chunks} chunks")
    print(f"  perturb_num_blocks        = {perturb_P} ({perturb_P * 3} frames)")
    print(f"  switch_frame_indices(raw) = {switch_frame_indices_raw}")
    print(f"  seed                      = {base_seed}")
    print(f"  output_folder             = {output_folder}")
    print("=" * 60)

    pipeline, text_encoder, vae, scheduler, sink_size = build_pipeline(config, device, dtype)

    if use_index_range:
        data_items = load_prompt_sequences(
            config.data_path,
            max_n=-1,
            index_start=idx_start,
            index_end=idx_end,
            min_segments=getattr(config, "prompt_min_segments", None),
            max_segments=getattr(config, "prompt_max_segments", None),
        )
        N = len(data_items)
        range_base = 0 if idx_start is None else int(idx_start)
    else:
        data_items = load_prompt_sequences(
            config.data_path,
            max_n=N,
            min_segments=getattr(config, "prompt_min_segments", None),
            max_segments=getattr(config, "prompt_max_segments", None),
        )
        if len(data_items) < N:
            print(f"Warning: only {len(data_items)} items in data, requested N={N}")
            N = len(data_items)
        range_base = 0

    all_plans = []

    for sample_idx in range(N):
        prompts_list: List[str] = data_items[sample_idx]["prompts"]
        num_segments = len(prompts_list)
        switch_frame_indices = resolve_switch_frame_indices(
            switch_frame_indices_raw,
            num_segments,
        )
        global_sample_idx = sample_idx_offset + range_base + sample_idx
        print(f"\n{'='*60}")
        print(f"Sample global={global_sample_idx} local={sample_idx}/{N}: {num_segments} segments")
        for si, p in enumerate(prompts_list):
            print(f"  Seg {si}: {p[:80]}...")
        print(f"{'='*60}")

        plan = pipeline.plan_perturbation(
            total_frames=max_length,
            chunk_size=chunk_size,
            min_new_frame=min_new_frame,
            K=K,
            base_seed=base_seed,
            sample_idx=global_sample_idx,
            seed_context=0,
            perturb_within_first_x_chunks=perturb_x_chunks,
            perturb_num_blocks=perturb_P,
            sink_size=sink_size,
        )
        all_plans.append(plan)

        print(f"Plan: perturb {plan['perturb_num_blocks']} blocks "
              f"@ frames {plan['perturb_block_abs_frame']}..{plan['perturb_block_end_frame']} "
              f"(chunk #{plan['perturb_chunk_idx']}, "
              f"start block #{plan['perturb_block_local_idx']}), "
              f"m={plan['m_nearest_frames']}, n={plan['n_random_frames']}, "
              f"pool={plan['num_older_available']}, "
              f"eligible={plan['eligible_blocks_count']}")
        for bp in plan["branch_plans"]:
            print(f"  Branch {bp['branch_k']}: seed={bp['seed']}, "
                  f"selected_older={bp['selected_older_frame_indices']}")

        with torch.no_grad():
            cond_list = []
            for p in prompts_list:
                cd = text_encoder(text_prompts=[p])
                cd = {k: v.to(device=device, dtype=dtype) if isinstance(v, torch.Tensor) else v
                      for k, v in cd.items()}
                cond_list.append(cd)

        noise_seed = base_seed + global_sample_idx * 10000
        perturb_idx = plan["perturb_chunk_idx"]
        chunks = plan["chunks"]

        # --- Phase 1: Generate prefix (chunks before perturbation) ---
        print(f"\n--- Phase 1: Generating prefix (chunks 0..{perturb_idx - 1}) ---")
        batch_size = 1
        pipeline._initialize_kv_cache(batch_size, dtype, device)
        pipeline._initialize_crossattn_cache(batch_size, dtype, device)
        pipeline._initialize_kv_bank(batch_size, dtype, device)
        pipeline.clear_kv_cache()
        pipeline.generator.model.local_attn_size = int(pipeline.local_attn_size)
        pipeline._set_all_modules_max_attention_size(int(pipeline.local_attn_size))

        rng_noise = torch.Generator(device=device)
        rng_noise.manual_seed(noise_seed)

        prefix_length, prefix_seg_idx, prefix_latents = _run_chunks(
            pipeline, chunks, 0, perturb_idx, 0,
            cond_list, switch_frame_indices, global_sink,
            prev_seg_idx=0, collected_latents=[],
            rng_noise=rng_noise, device=device, dtype=dtype,
        )

        pre_perturb_rng_state = rng_noise.get_state()
        pipeline.save_state()
        saved_cpu_rng = torch.random.get_rng_state()
        saved_cuda_rng = torch.cuda.get_rng_state(device)
        print(f"Prefix generated: {prefix_length} frames, state saved")

        # --- Phase 2: For each branch, generate from perturbation onward ---
        for k in range(K):
            print(f"\n--- Branch {k}/{K} ---")
            pipeline.restore_state()
            torch.random.set_rng_state(saved_cpu_rng)
            torch.cuda.set_rng_state(saved_cuda_rng, device)

            branch_rng = torch.Generator(device=device)
            branch_rng.manual_seed(noise_seed)
            branch_rng.set_state(pre_perturb_rng_state)

            current_length, _, branch_new = _run_chunks(
                pipeline, chunks, perturb_idx, len(chunks), prefix_length,
                cond_list, switch_frame_indices, global_sink,
                prev_seg_idx=prefix_seg_idx,
                collected_latents=list(prefix_latents),
                rng_noise=branch_rng, device=device, dtype=dtype,
                perturb_ci=perturb_idx,
                branch_plan=plan["branch_plans"][k],
                target_abs_frame=plan["perturb_block_abs_frame"],
                target_end_frame=plan["perturb_block_end_frame"],
            )

            all_lat = torch.cat(prefix_latents + branch_new, dim=1).to(device=device, dtype=dtype)
            print(f"  Decoding {all_lat.shape[1]} latent frames...")
            video = vae.decode_to_pixel(all_lat)
            video = (video * 0.5 + 0.5).clamp(0, 1)
            video = rearrange(video, 'b t c h w -> b t h w c').cpu()
            video_uint8 = (video[0] * 255.0).to(torch.uint8)

            out_path = os.path.join(output_folder, f"sample{global_sample_idx}_branch{k}.mp4")
            write_video(out_path, video_uint8, fps=16)
            print(f"  Saved: {out_path}")

            del all_lat, video, video_uint8
            vae.model.clear_cache()
            torch.cuda.empty_cache()

    plan_path = os.path.join(output_folder, "diversity_plans.json")
    serializable_plans = []
    for p in all_plans:
        sp = {k: v for k, v in p.items()}
        sp["chunks"] = [dict(c) for c in p["chunks"]]
        sp["branch_plans"] = [dict(bp) for bp in p["branch_plans"]]
        serializable_plans.append(sp)
    with open(plan_path, "w") as f:
        json.dump(serializable_plans, f, indent=2)
    print(f"\nDiversity plans saved to {plan_path}")
    print("Done!")


if __name__ == "__main__":
    main()
