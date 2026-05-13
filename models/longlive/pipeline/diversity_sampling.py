import torch
from typing import Any, Dict, List, Optional

from models.longlive.pipeline.streaming_training import StreamingTrainingPipeline


class DiversitySamplingPipeline(StreamingTrainingPipeline):
    """Pipeline for diversity sampling via KV local perturbation.

    Extends the streaming training pipeline with:
    - Pre-planned KV perturbation masks for a designated chunk
    - KV cache state save / restore for branching into K diverse samples
    - Deterministic, reproducible random selection via isolated torch.Generator
    """

    def __init__(self, *args, **kwargs):
        self.m_nearest_frames = kwargs.pop("m_nearest_frames", 6)
        self.recache_full_kv_cache_after_switch = bool(
            kwargs.pop("recache_full_kv_cache_after_switch", False)
        )
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
        forced_perturb_chunk_idx: Optional[int] = None,
        forced_perturb_block_local_idx: Optional[int] = None,
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
        a different random selection of absolute older video frames that
        stay resident for the full P-block perturbation window.

        ``kv_cache_size`` (hence ``kv_cache_size // frame_seq_length``) comes from
        ``StreamingTrainingPipeline``: ``(local_attn_size + slice_last_frames)``
        latent frames × ``frame_seq_length`` tokens per frame — e.g. 12+21=33
        frames when those kwargs match ``train_kvpo_longlive.yaml``.

        """
        chunks = self._compute_chunk_layout(total_frames, chunk_size, min_new_frame)

        P = perturb_num_blocks
        num_frame_per_block = self.num_frame_per_block
        if self.local_attn_size == -1:
            raise ValueError("KV perturbation requires finite local_attn_size in longlive")
        local_budget = max(self.local_attn_size - sink_size, 0)
        n_random = max(local_budget - self.m_nearest_frames, 0)
        kv_cache_total_capacity = self.kv_cache_size // self.frame_seq_length
        non_sink_cache_capacity = max(kv_cache_total_capacity - sink_size, 0)

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
                perturb_last_start = abs_frame + (P - 1) * num_frame_per_block
                stable_pool_start = max(
                    sink_size,
                    perturb_last_start - non_sink_cache_capacity,
                )
                stable_pool_end = abs_frame - self.m_nearest_frames
                pool = max(stable_pool_end - stable_pool_start, 0)
                if pool < n_random:
                    continue
                eligible_blocks.append({
                    "chunk_idx": ci,
                    "block_local_idx": bi,
                    "abs_frame": abs_frame,
                    "pool_size": pool,
                    "stable_pool_start_frame": stable_pool_start,
                    "stable_pool_end_frame": stable_pool_end,
                })

        assert len(eligible_blocks) > 0, (
            f"No eligible start block for P={P} in first {search_limit} chunks "
            f"(need abs_frame >= {self.local_attn_size}, pool >= {n_random}, "
            f"and {P} consecutive blocks within one chunk)"
        )

        if forced_perturb_chunk_idx is not None:
            eligible_blocks = [
                blk for blk in eligible_blocks
                if blk["chunk_idx"] == forced_perturb_chunk_idx
            ]
        if forced_perturb_block_local_idx is not None:
            eligible_blocks = [
                blk for blk in eligible_blocks
                if blk["block_local_idx"] == forced_perturb_block_local_idx
            ]

        assert len(eligible_blocks) > 0, (
            "No eligible start block remains after forced perturb filtering "
            f"(chunk_idx={forced_perturb_chunk_idx}, "
            f"block_local_idx={forced_perturb_block_local_idx})"
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
        older_frame_candidates = list(
            range(target["stable_pool_start_frame"], target["stable_pool_end_frame"])
        )

        branch_plans = self._build_dispersed_branch_plans(
            K,
            n_to_select,
            older_frame_candidates,
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
            "forced_perturb_chunk_idx": forced_perturb_chunk_idx,
            "forced_perturb_block_local_idx": forced_perturb_block_local_idx,
            "m_nearest_frames": self.m_nearest_frames,
            "n_random_frames": n_to_select,
            "num_older_available": num_older_available,
            "stable_older_frame_start": target["stable_pool_start_frame"],
            "stable_older_frame_end": target["stable_pool_end_frame"],
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
        }
        self._saved_state = state

    def restore_state(self):
        """Restore KV caches from the previously saved snapshot."""
        if self._saved_state is None:
            raise RuntimeError("No saved state to restore")
        self._restore_cache_list(self.kv_cache1, self._saved_state["kv_cache1"])
        self._restore_crossattn_cache(self.crossattn_cache, self._saved_state["crossattn_cache"])

    # ------------------------------------------------------------------
    # Perturbation activation
    # ------------------------------------------------------------------

    def _get_causal_model(self):
        """Get the underlying CausalWanModel, unwrapping LoRA/peft if needed."""
        model = self.generator.model
        if hasattr(model, 'base_model') and hasattr(model.base_model, 'model'):
            return model.base_model.model
        return model

    def activate_perturbation(
        self,
        branch_plan: dict,
        target_abs_frame: int,
        target_end_frame: int,
    ):
        """Set the perturbation config on the generator model.

        The perturbation fires for every block whose ``current_start``
        falls in ``[target_abs_frame, target_end_frame)`` (in token units).
        """
        config = {
            "m_nearest_frames": self.m_nearest_frames,
            "selected_older_frame_indices": branch_plan["selected_older_frame_indices"],
            "target_start_token": target_abs_frame * self.frame_seq_length,
            "target_end_token": target_end_frame * self.frame_seq_length,
            "sink_size": getattr(self._get_causal_model(), "sink_size", 0),
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
        if self.recache_full_kv_cache_after_switch:
            recache_budget = self.kv_cache_size // self.frame_seq_length
        else:
            recache_budget = (
                self.local_attn_size if self.local_attn_size != -1 else current_start_frame
            )
        num_recache = min(recache_budget, current_start_frame, all_lat.shape[1])
        print(
            f"[KVPO recache_after_switch] mode="
            f"{'kv_cache' if self.recache_full_kv_cache_after_switch else 'local_attn'} "
            f"num_recache={num_recache} current_start_frame={current_start_frame}"
        )
        frames_to_recache = all_lat[:, -num_recache:].to(
            device=self.kv_cache1[0]["k"].device,
            dtype=self.kv_cache1[0]["k"].dtype,
        )
        batch_size = frames_to_recache.shape[0]
        device = frames_to_recache.device
        causal_model = self._get_causal_model()
        max_recache_pass_frames = max(1, int(self.generator.seq_len // self.frame_seq_length))
        recache_start_frame = current_start_frame - num_recache

        for offset in range(0, num_recache, max_recache_pass_frames):
            recache_chunk = frames_to_recache[:, offset:offset + max_recache_pass_frames]
            chunk_frames = recache_chunk.shape[1]
            context_ts = torch.ones(
                [batch_size, chunk_frames], device=device, dtype=torch.int64,
            ) * self.context_noise
            causal_model.block_mask = None

            with torch.no_grad():
                self.generator(
                    noisy_image_or_video=recache_chunk,
                    conditional_dict=new_conditional_dict,
                    timestep=context_ts,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=(recache_start_frame + offset) * self.frame_seq_length,
                    sink_recache_after_switch=not global_sink,
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
        randn_generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Generate a chunk without gradients (pure sampling).

        Delegates to the parent generate_chunk_with_cache but always
        disables gradients.
        """
        with torch.no_grad():
            output, _, _ = self.generate_chunk_with_cache(
                noise=noise,
                conditional_dict=conditional_dict,
                current_start_frame=current_start_frame,
                requires_grad=False,
                randn_generator=randn_generator,
            )
        return output

    def generate_chunk_sampling_with_trajectory(
        self,
        noise: torch.Tensor,
        conditional_dict: dict,
        current_start_frame: int = 0,
        randn_generator: Optional[torch.Generator] = None,
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
                return_trajectory=True,
                randn_generator=randn_generator,
            )
        return output, trajectory

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_dispersed_branch_plans(
        K: int,
        n_to_select: int,
        older_frame_candidates: List[int],
        base_seed: int,
        sample_idx: int,
        seed_context: int = 0,
    ) -> List[dict]:
        """Create K branch plans with maximally dispersed frame selections.

        Instead of K independent random draws (which overlap heavily),
        we generate one master permutation of the stable older-frame pool
        and give each branch a shifted window so that the overlap between
        any two branches is minimised.

        ``selected_older_frame_indices`` are absolute video-frame indices,
        not relative positions inside the resident cache.
        """
        branch_plans = []
        num_older_available = len(older_frame_candidates)
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
            selected = sorted(older_frame_candidates[idx] for idx in indices)

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

    @staticmethod
    def _restore_crossattn_cache(dst_list, src_list):
        if src_list is None or dst_list is None:
            return
        for dst, src in zip(dst_list, src_list):
            dst["k"].copy_(src["k"].to(dst["k"].device))
            dst["v"].copy_(src["v"].to(dst["v"].device))
            dst["is_init"] = src["is_init"]
