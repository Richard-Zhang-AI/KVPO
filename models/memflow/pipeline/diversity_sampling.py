import torch
from typing import List, Dict, Any

from models.memflow.pipeline.streaming_training import StreamingTrainingPipeline


class DiversitySamplingPipeline(StreamingTrainingPipeline):
    """Pipeline for diversity sampling via KV local perturbation.

    Extends the streaming training pipeline with:
    - Pre-planned KV perturbation masks for a designated chunk
    - KV cache state save / restore for branching into K diverse samples
    - Deterministic, reproducible random selection via isolated torch.Generator
    """

    def __init__(self, *args, **kwargs):
        self.m_nearest_frames = kwargs.pop("m_nearest_frames", 6)
        super().__init__(*args, **kwargs)
        self._saved_state = None

    def generate_and_sync_list(self, num_blocks, num_denoising_steps, device):
        """Local-only version that skips dist.broadcast.

        Each GPU runs its rollout independently, so cross-rank
        synchronisation of denoising exit flags would deadlock.
        """
        if self.last_step_only:
            return [num_denoising_steps - 1] * num_blocks
        return torch.randint(0, num_denoising_steps, (num_blocks,),
                             device=device).tolist()

    # ------------------------------------------------------------------
    # Planning: decide *which single block* to perturb and *how*
    # ------------------------------------------------------------------

    def plan_perturbation(
        self,
        total_frames: int,
        chunk_size: int,
        min_new_frame: int,
        K: int,
        base_seed: int,
        sample_idx: int,
        seed_context: int = 0,
        perturb_within_first_x_chunks: int = 3,
        perturb_num_blocks: int = 3,
        sink_size: int = 3,
    ) -> Dict[str, Any]:
        """Plan multi-block perturbation before any generation starts.

        We select a *starting* block from the first
        ``perturb_within_first_x_chunks`` chunks, then perturb P =
        ``perturb_num_blocks`` consecutive blocks beginning there.

        Starting-block eligibility:
          1. abs_frame >= local_attn_size  (enough history)
          2. pool >= n_random              (enough older frames)
          3. the P consecutive blocks fit inside the same chunk

        Each of the K branches perturbs the same P-block range but with
        a different random selection of older frames in the KV window.
        """
        chunks = self._compute_chunk_layout(total_frames, chunk_size, min_new_frame)

        P = perturb_num_blocks
        num_frame_per_block = self.num_frame_per_block
        kv_cache_total_capacity = self.kv_cache_size // self.frame_seq_length
        n_random = self.local_attn_size - self.m_nearest_frames

        eligible_blocks = []
        search_limit = min(perturb_within_first_x_chunks, len(chunks))
        for ci in range(search_limit):
            c = chunks[ci]
            num_blocks_in_chunk = c["new_frames"] // num_frame_per_block
            for bi in range(num_blocks_in_chunk):
                abs_frame = c["start_frame"] + bi * num_frame_per_block
                if abs_frame < self.local_attn_size:
                    continue
                if bi + P > num_blocks_in_chunk:
                    continue
                frames_in_cache = min(abs_frame, kv_cache_total_capacity)
                pool = max(frames_in_cache - self.m_nearest_frames, 0)
                if pool < n_random:
                    continue
                eligible_blocks.append({
                    "chunk_idx": ci,
                    "block_local_idx": bi,
                    "abs_frame": abs_frame,
                    "pool_size": pool,
                })

        assert len(eligible_blocks) > 0, (
            f"No eligible start block for P={P} in first {search_limit} chunks "
            f"(need abs_frame >= {self.local_attn_size}, pool >= {n_random}, "
            f"and {P} consecutive blocks within one chunk)"
        )

        block_rng = torch.Generator()
        block_rng.manual_seed(base_seed + sample_idx * 97 + seed_context)
        pick_idx = torch.randint(len(eligible_blocks), (1,), generator=block_rng).item()
        target = eligible_blocks[pick_idx]

        target_abs_frame = target["abs_frame"]
        target_end_frame = target_abs_frame + P * num_frame_per_block
        perturb_chunk_idx = target["chunk_idx"]
        num_older_available = target["pool_size"]
        n_to_select = min(n_random, num_older_available)

        branch_plans = self._build_dispersed_branch_plans(
            K,
            n_to_select,
            num_older_available,
            base_seed,
            sample_idx,
            seed_context=seed_context,
        )

        plan = {
            "sample_idx": sample_idx,
            "seed_context": seed_context,
            "base_seed": base_seed,
            "K": K,
            "perturb_chunk_idx": perturb_chunk_idx,
            "perturb_block_abs_frame": target_abs_frame,
            "perturb_block_end_frame": target_end_frame,
            "perturb_num_blocks": P,
            "perturb_block_local_idx": target["block_local_idx"],
            "m_nearest_frames": self.m_nearest_frames,
            "n_random_frames": n_to_select,
            "num_older_available": num_older_available,
            "local_attn_size": self.local_attn_size,
            "eligible_blocks_count": len(eligible_blocks),
            "chunks": chunks,
            "branch_plans": branch_plans,
        }
        return plan

    # ------------------------------------------------------------------
    # State save / restore for branching
    # ------------------------------------------------------------------

    def save_state(self):
        """Deep-copy current KV caches to CPU for later restore."""
        state = {
            "kv_cache1": self._clone_cache_list(self.kv_cache1),
            "crossattn_cache": self._clone_crossattn_cache(self.crossattn_cache),
            "kv_bank1": self._clone_cache_list(self.kv_bank1),
        }
        self._saved_state = state

    def restore_state(self):
        """Restore KV caches from the previously saved snapshot."""
        if self._saved_state is None:
            raise RuntimeError("No saved state to restore")
        self._restore_cache_list(self.kv_cache1, self._saved_state["kv_cache1"])
        self._restore_crossattn_cache(self.crossattn_cache, self._saved_state["crossattn_cache"])
        self._restore_cache_list(self.kv_bank1, self._saved_state["kv_bank1"])

    # ------------------------------------------------------------------
    # Perturbation activation
    # ------------------------------------------------------------------

    def _get_causal_model(self):
        """Get the underlying CausalWanModel, unwrapping LoRA/peft if needed."""
        model = self.generator.model
        if hasattr(model, 'base_model') and hasattr(model.base_model, 'model'):
            return model.base_model.model
        return model

    def activate_perturbation(self, branch_plan: dict,
                              target_abs_frame: int, target_end_frame: int):
        """Set the perturbation config on the generator model.

        The perturbation fires for every block whose ``current_start``
        falls in ``[target_abs_frame, target_end_frame)`` (in token units).
        """
        config = {
            "m_nearest_frames": self.m_nearest_frames,
            "selected_older_frame_indices": branch_plan["selected_older_frame_indices"],
            "target_start_token": target_abs_frame * self.frame_seq_length,
            "target_end_token": target_end_frame * self.frame_seq_length,
        }
        self._get_causal_model().set_kv_perturbation(config)

    def deactivate_perturbation(self):
        """Clear the perturbation config."""
        self._get_causal_model().clear_kv_perturbation()

    # ------------------------------------------------------------------
    # Prompt switch support (recache)
    # ------------------------------------------------------------------

    def recache_after_switch(self, generated_latents: List[torch.Tensor],
                             current_start_frame: int,
                             new_conditional_dict: dict,
                             global_sink: bool = True):
        """Refresh KV / cross-attn caches when the prompt switches.

        Mirrors ``InteractiveCausalInferencePipeline._recache_after_switch``.
        """
        if not global_sink:
            for blk in self.kv_cache1:
                blk["k"].zero_()
                blk["v"].zero_()

        for blk in self.crossattn_cache:
            blk["k"].zero_()
            blk["v"].zero_()
            blk["is_init"] = False

        if current_start_frame == 0:
            return

        all_lat = torch.cat(generated_latents, dim=1)
        local_attn = self.local_attn_size if self.local_attn_size != -1 else current_start_frame
        num_recache = min(local_attn, current_start_frame, all_lat.shape[1])
        frames_to_recache = all_lat[:, -num_recache:].to(
            device=self.kv_cache1[0]["k"].device,
            dtype=self.kv_cache1[0]["k"].dtype,
        )
        batch_size = frames_to_recache.shape[0]

        device = frames_to_recache.device
        causal_model = self._get_causal_model()
        block_mask = causal_model._prepare_blockwise_causal_attn_mask(
            device=device,
            num_frames=num_recache,
            frame_seqlen=self.frame_seq_length,
            num_frame_per_block=self.num_frame_per_block,
            local_attn_size=self.local_attn_size,
        )
        context_ts = torch.ones(
            [batch_size, num_recache], device=device, dtype=torch.int64,
        ) * self.context_noise
        causal_model.block_mask = block_mask

        with torch.no_grad():
            self.generator(
                noisy_image_or_video=frames_to_recache,
                conditional_dict=new_conditional_dict,
                timestep=context_ts,
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                current_start=(current_start_frame - num_recache) * self.frame_seq_length,
                kv_bank=self.kv_bank1,
                update_bank=False,
                update_cache=True,
                q_bank=True,
                is_recache=True,
            )

        for blk in self.crossattn_cache:
            blk["k"].zero_()
            blk["v"].zero_()
            blk["is_init"] = False

    # ------------------------------------------------------------------
    # Chunk-level generation (no-grad, sampling only)
    # ------------------------------------------------------------------

    def generate_chunk_sampling(
        self,
        noise: torch.Tensor,
        conditional_dict: dict,
        current_start_frame: int = 0,
    ) -> torch.Tensor:
        """Generate a chunk without gradients (pure sampling).

        Delegates to the parent generate_chunk_with_cache but always
        disables gradients and enables bank updates.
        """
        with torch.no_grad():
            output, _, _ = self.generate_chunk_with_cache(
                noise=noise,
                conditional_dict=conditional_dict,
                current_start_frame=current_start_frame,
                requires_grad=False,
                update_bank=True,
            )
        return output

    def generate_chunk_sampling_with_trajectory(
        self,
        noise: torch.Tensor,
        conditional_dict: dict,
        current_start_frame: int = 0,
    ):
        """Generate a chunk and return per-step denoising trajectory.

        Returns:
            output: generated chunk [B, T, C, H, W]
            trajectory: dict with 'noisy_inputs' and 'x0_preds', each a list
                of lists: trajectory[key][block_idx][step_idx] -> [B, F, C, H, W] on CPU
        """
        with torch.no_grad():
            output, _, _, trajectory = self.generate_chunk_with_cache(
                noise=noise,
                conditional_dict=conditional_dict,
                current_start_frame=current_start_frame,
                requires_grad=False,
                update_bank=True,
                return_trajectory=True,
            )
        return output, trajectory

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_dispersed_branch_plans(
        K: int, n_to_select: int, num_older_available: int,
        base_seed: int, sample_idx: int,
        seed_context: int = 0,
    ) -> List[dict]:
        """Create K branch plans with maximally dispersed frame selections.

        Instead of K independent random draws (which overlap heavily),
        we generate one master permutation of the pool and give each
        branch a shifted window so that the overlap between any two
        branches is minimised.

        With pool P and selection size n:
          shift = P // K
          branch k gets master_perm[ k*shift : k*shift + n ]  (wrapping)
        Adjacent branches overlap by max(0, n - shift) frames.
        """
        branch_plans = []
        if n_to_select <= 0 or num_older_available <= 0:
            for k in range(K):
                branch_plans.append({
                    "branch_k": k,
                    "seed": base_seed + sample_idx * K + k + seed_context,
                    "selected_older_frame_indices": [],
                })
            return branch_plans

        master_rng = torch.Generator()
        master_rng.manual_seed(base_seed + sample_idx * 97 + 1 + seed_context)
        master_perm = torch.randperm(num_older_available, generator=master_rng).tolist()

        shift = max(1, num_older_available // K)

        for k in range(K):
            start = (k * shift) % num_older_available
            indices = []
            for i in range(n_to_select):
                indices.append(master_perm[(start + i) % num_older_available])
            selected = sorted(indices)

            branch_plans.append({
                "branch_k": k,
                "seed": base_seed + sample_idx * K + k + seed_context,
                "selected_older_frame_indices": selected,
            })

        return branch_plans

    @staticmethod
    def _compute_chunk_layout(total_frames: int, chunk_size: int, min_new_frame: int) -> List[dict]:
        """Compute the deterministic chunk layout for the full video."""
        chunks = []
        current = 0
        idx = 0
        while current < total_frames:
            if idx == 0:
                new_frames = chunk_size
            else:
                new_frames = min_new_frame
            end = min(current + new_frames, total_frames)
            actual_new = end - current
            chunks.append({
                "chunk_idx": idx,
                "start_frame": current,
                "new_frames": actual_new,
                "chunk_size": chunk_size if idx > 0 else actual_new,
            })
            current = end
            idx += 1
        return chunks

    @staticmethod
    def _clone_cache_list(cache_list):
        if cache_list is None:
            return None
        cloned = []
        for blk in cache_list:
            cloned.append({
                "k": blk["k"].clone().cpu(),
                "v": blk["v"].clone().cpu(),
                "global_end_index": blk["global_end_index"].clone().cpu(),
                "local_end_index": blk["local_end_index"].clone().cpu(),
                **({"k_new": blk["k_new"].clone().cpu()} if "k_new" in blk else {}),
                **({"v_new": blk["v_new"].clone().cpu()} if "v_new" in blk else {}),
            })
        return cloned

    @staticmethod
    def _clone_crossattn_cache(cache_list):
        if cache_list is None:
            return None
        cloned = []
        for blk in cache_list:
            cloned.append({
                "k": blk["k"].clone().cpu(),
                "v": blk["v"].clone().cpu(),
                "is_init": blk["is_init"],
            })
        return cloned

    @staticmethod
    def _restore_cache_list(dst_list, src_list):
        if src_list is None or dst_list is None:
            return
        for dst, src in zip(dst_list, src_list):
            dst["k"].copy_(src["k"].to(dst["k"].device))
            dst["v"].copy_(src["v"].to(dst["v"].device))
            dst["global_end_index"].copy_(src["global_end_index"].to(dst["global_end_index"].device))
            dst["local_end_index"].copy_(src["local_end_index"].to(dst["local_end_index"].device))
            if "k_new" in src and "k_new" in dst:
                dst["k_new"].copy_(src["k_new"].to(dst["k_new"].device))
            if "v_new" in src and "v_new" in dst:
                dst["v_new"].copy_(src["v_new"].to(dst["v_new"].device))

    @staticmethod
    def _restore_crossattn_cache(dst_list, src_list):
        if src_list is None or dst_list is None:
            return
        for dst, src in zip(dst_list, src_list):
            dst["k"].copy_(src["k"].to(dst["k"].device))
            dst["v"].copy_(src["v"].to(dst["v"].device))
            dst["is_init"] = src["is_init"]
