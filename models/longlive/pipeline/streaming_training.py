# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0).
# You may not use this file except in compliance with the License.
# To view a copy of this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en
#
# No warranties are given. The work is provided "AS IS", without warranty of any kind, express or implied.
#
# SPDX-License-Identifier: CC-BY-NC-SA-4.0
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist

from models.longlive.utils.debug_option import DEBUG, LOG_GPU_MEMORY
from models.longlive.utils.memory import log_gpu_memory
from models.longlive.utils.scheduler import SchedulerInterface
from models.longlive.utils.wan_wrapper import WanDiffusionWrapper


class StreamingTrainingPipeline:
    def __init__(
        self,
        denoising_step_list: List[int],
        scheduler: SchedulerInterface,
        generator: WanDiffusionWrapper,
        num_frame_per_block=3,
        same_step_across_blocks: bool = False,
        last_step_only: bool = False,
        context_noise: int = 0,
        **kwargs,
    ):
        super().__init__()
        self.scheduler = scheduler
        self.generator = generator
        self.denoising_step_list = denoising_step_list
        if self.denoising_step_list[-1] == 0:
            self.denoising_step_list = self.denoising_step_list[:-1]

        self.num_transformer_blocks = 30
        self.frame_seq_length = 1560
        self.num_frame_per_block = num_frame_per_block
        self.context_noise = context_noise

        self.kv_cache1 = None
        self.crossattn_cache = None
        self.same_step_across_blocks = same_step_across_blocks
        self.last_step_only = last_step_only

        self.local_attn_size = kwargs.get("local_attn_size", -1)

        slice_last_frames = int(kwargs.get("slice_last_frames", 21))
        self.kv_cache_size = (self.local_attn_size + slice_last_frames) * self.frame_seq_length
        if DEBUG:
            print(
                f"[KV policy] local_attn_size={self.local_attn_size} "
                f"slice_last_frames={slice_last_frames} -> kv_frames={self.kv_cache_size}"
            )

    def generate_and_sync_list(self, num_blocks, num_denoising_steps, device):
        rank = dist.get_rank() if dist.is_initialized() else 0

        if rank == 0:
            indices = torch.randint(
                low=0,
                high=num_denoising_steps,
                size=(num_blocks,),
                device=device,
            )
            if self.last_step_only:
                indices = torch.ones_like(indices) * (num_denoising_steps - 1)
        else:
            indices = torch.empty(num_blocks, dtype=torch.long, device=device)
        if dist.is_initialized():
            dist.broadcast(indices, src=0)
        return indices.tolist()

    def generate_chunk_with_cache(
        self,
        noise: torch.Tensor,
        conditional_dict: dict,
        *,
        current_start_frame: int = 0,
        requires_grad: bool = True,
        return_sim_step: bool = False,
        return_trajectory: bool = False,
        randn_generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, Optional[int], Optional[int]]:
        batch_size, chunk_frames, _num_channels, _height, _width = noise.shape
        assert chunk_frames % self.num_frame_per_block == 0
        num_blocks = chunk_frames // self.num_frame_per_block
        all_num_frames = [self.num_frame_per_block] * num_blocks

        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[SeqTrain-Pipeline] generate_chunk_with_cache: batch_size={batch_size}, chunk_frames={chunk_frames}")
            print(f"[SeqTrain-Pipeline] current_start_frame={current_start_frame}, requires_grad={requires_grad}")

        if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY:
            log_gpu_memory(
                "SeqTrain-Pipeline: Before chunk generation",
                device=noise.device,
                rank=dist.get_rank() if dist.is_initialized() else 0,
            )

        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[SeqTrain-Pipeline] Block config: num_blocks={num_blocks}, all_num_frames={all_num_frames}")

        output = torch.zeros_like(noise)
        num_denoising_steps = len(self.denoising_step_list)
        exit_flags = self.generate_and_sync_list(
            len(all_num_frames), num_denoising_steps, device=noise.device
        )

        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[SeqTrain-Pipeline] Denoising steps: {num_denoising_steps}, exit_flags: {exit_flags}")

        start_gradient_frame_index = 0 if requires_grad else chunk_frames

        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[SeqTrain-Pipeline] start_gradient_frame_index={start_gradient_frame_index}")

        if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY:
            log_gpu_memory(
                "SeqTrain-Pipeline: Before block generation loop",
                device=noise.device,
                rank=dist.get_rank() if dist.is_initialized() else 0,
            )

        local_start_frame = 0
        self.generator.model.local_attn_size = int(self.local_attn_size)
        self._set_all_modules_max_attention_size(int(self.local_attn_size))

        trajectory_noisy_inputs = [] if return_trajectory else None
        trajectory_x0_preds = [] if return_trajectory else None

        for block_index, current_num_frames in enumerate(all_num_frames):
            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                print(
                    f"[SeqTrain-Pipeline] Processing block {block_index}: "
                    f"frames {local_start_frame}-{local_start_frame + current_num_frames}"
                )

            if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY and block_index == 0:
                log_gpu_memory(
                    "SeqTrain-Pipeline: Before first block generation",
                    device=noise.device,
                    rank=dist.get_rank() if dist.is_initialized() else 0,
                )

            noisy_input = noise[:, local_start_frame:local_start_frame + current_num_frames]
            block_noisy_inputs = [] if return_trajectory else None
            block_x0_preds = [] if return_trajectory else None

            for step_idx, current_timestep in enumerate(self.denoising_step_list):
                exit_flag = (
                    step_idx == exit_flags[0]
                    if self.same_step_across_blocks
                    else step_idx == exit_flags[block_index]
                )

                timestep = torch.ones(
                    [batch_size, current_num_frames],
                    device=noise.device,
                    dtype=torch.int64,
                ) * current_timestep

                if not exit_flag:
                    if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                        print(f"[SeqTrain-Pipeline] Block {block_index} intermediate steps (no grad)")

                    if return_trajectory:
                        block_noisy_inputs.append(noisy_input.detach().cpu())

                    with torch.no_grad():
                        _, denoised_pred = self.generator(
                            noisy_image_or_video=noisy_input,
                            conditional_dict=conditional_dict,
                            timestep=timestep,
                            kv_cache=self.kv_cache1,
                            crossattn_cache=self.crossattn_cache,
                            current_start=(current_start_frame + local_start_frame) * self.frame_seq_length,
                        )

                        if return_trajectory:
                            block_x0_preds.append(denoised_pred.detach().cpu())

                        if step_idx < len(self.denoising_step_list) - 1:
                            next_timestep = self.denoising_step_list[step_idx + 1]
                            step_noise = torch.randn(
                                denoised_pred.flatten(0, 1).shape,
                                device=noise.device,
                                dtype=denoised_pred.dtype,
                                generator=randn_generator,
                            )
                            noisy_input = self.scheduler.add_noise(
                                denoised_pred.flatten(0, 1),
                                step_noise,
                                next_timestep * torch.ones(
                                    [batch_size * current_num_frames],
                                    device=noise.device,
                                    dtype=torch.long,
                                ),
                            ).unflatten(0, denoised_pred.shape[:2])
                else:
                    enable_grad = local_start_frame >= start_gradient_frame_index

                    if return_trajectory:
                        block_noisy_inputs.append(noisy_input.detach().cpu())

                    if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                        print(f"[SeqTrain-Pipeline] Block {block_index} final step: enable_grad={enable_grad}")

                    context_manager = torch.enable_grad() if enable_grad else torch.no_grad()
                    with context_manager:
                        _, denoised_pred = self.generator(
                            noisy_image_or_video=noisy_input,
                            conditional_dict=conditional_dict,
                            timestep=timestep,
                            kv_cache=self.kv_cache1,
                            crossattn_cache=self.crossattn_cache,
                            current_start=(current_start_frame + local_start_frame) * self.frame_seq_length,
                        )

                    if return_trajectory:
                        block_x0_preds.append(denoised_pred.detach().cpu())
                    break

            if return_trajectory:
                trajectory_noisy_inputs.append(block_noisy_inputs)
                trajectory_x0_preds.append(block_x0_preds)

            output[:, local_start_frame:local_start_frame + current_num_frames] = denoised_pred

            context_timestep = torch.ones_like(timestep) * self.context_noise
            context_noise = torch.randn(
                denoised_pred.flatten(0, 1).shape,
                device=noise.device,
                dtype=denoised_pred.dtype,
                generator=randn_generator,
            )
            context_noisy = self.scheduler.add_noise(
                denoised_pred.flatten(0, 1),
                context_noise,
                context_timestep.flatten(0, 1),
            ).unflatten(0, denoised_pred.shape[:2])

            if DEBUG and block_index == 0 and (not dist.is_initialized() or dist.get_rank() == 0):
                print(f"[SeqTrain-Pipeline] Updating cache with context_noise={self.context_noise}")

            with torch.no_grad():
                self.generator(
                    noisy_image_or_video=context_noisy,
                    conditional_dict=conditional_dict,
                    timestep=context_timestep,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=(current_start_frame + local_start_frame) * self.frame_seq_length,
                )

            local_start_frame += current_num_frames

        if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY:
            log_gpu_memory(
                "SeqTrain-Pipeline: After all blocks generated",
                device=noise.device,
                rank=dist.get_rank() if dist.is_initialized() else 0,
            )

        if not self.same_step_across_blocks:
            denoised_timestep_from, denoised_timestep_to = None, None
        elif exit_flags[0] == len(self.denoising_step_list) - 1:
            denoised_timestep_to = 0
            denoised_timestep_from = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0]].cuda()).abs(),
                dim=0,
            ).item()
        else:
            denoised_timestep_to = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0] + 1].cuda()).abs(),
                dim=0,
            ).item()
            denoised_timestep_from = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0]].cuda()).abs(),
                dim=0,
            ).item()

        if return_sim_step:
            return output, denoised_timestep_from, denoised_timestep_to, exit_flags[0] + 1

        if return_trajectory:
            return output, denoised_timestep_from, denoised_timestep_to, {
                "noisy_inputs": trajectory_noisy_inputs,
                "x0_preds": trajectory_x0_preds,
            }

        return output, denoised_timestep_from, denoised_timestep_to

    def _initialize_kv_cache(self, batch_size, dtype, device):
        kv_cache1 = []
        if DEBUG:
            rank = dist.get_rank() if dist.is_initialized() else 0
            print(
                f"rank {rank} initialize kv cache with batch_size: "
                f"{batch_size}, kv_cache_size: {self.kv_cache_size}"
            )
        for _ in range(self.num_transformer_blocks):
            kv_cache1.append(
                {
                    "k": torch.zeros([batch_size, self.kv_cache_size, 12, 128], dtype=dtype, device=device),
                    "v": torch.zeros([batch_size, self.kv_cache_size, 12, 128], dtype=dtype, device=device),
                    "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                    "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
                }
            )

        self.kv_cache1 = kv_cache1

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        crossattn_cache = []

        for _ in range(self.num_transformer_blocks):
            crossattn_cache.append(
                {
                    "k": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                    "v": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                    "is_init": False,
                }
            )
        self.crossattn_cache = crossattn_cache

    def clear_kv_cache(self):
        if getattr(self, "kv_cache1", None) is not None:
            for blk in self.kv_cache1:
                blk["k"].zero_()
                blk["v"].zero_()
                if "global_end_index" in blk:
                    blk["global_end_index"].zero_()
                if "local_end_index" in blk:
                    blk["local_end_index"].zero_()

        if getattr(self, "crossattn_cache", None) is not None:
            for blk in self.crossattn_cache:
                blk["k"].zero_()
                blk["v"].zero_()
                blk["is_init"] = False

    def _set_all_modules_max_attention_size(self, local_attn_size_value: int):
        if isinstance(local_attn_size_value, (list, tuple)):
            raise ValueError("_set_all_modules_max_attention_size expects an int, got list/tuple.")

        if int(local_attn_size_value) == -1:
            target_size = 32760
        else:
            target_size = int(local_attn_size_value) * self.frame_seq_length

        if hasattr(self.generator.model, "max_attention_size"):
            try:
                _ = getattr(self.generator.model, "max_attention_size")
            except Exception:
                pass
            setattr(self.generator.model, "max_attention_size", target_size)

        for _name, module in self.generator.model.named_modules():
            if hasattr(module, "max_attention_size"):
                try:
                    setattr(module, "max_attention_size", target_size)
                except Exception:
                    pass
