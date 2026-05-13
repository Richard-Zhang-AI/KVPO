"""Diversity via extra Gaussian noise on the diffusion latent at selected denoise steps.

Injections occur *before* each denoising forward at step indices listed in
``noise_inject_at_steps`` (0 = initial noisy latent for the block, 1 = latent
before the second scheduler step, etc.).  First version defaults to applying at
every temporal block in the chunk; controlled by ``inject_noise_every_block``.
"""

from __future__ import annotations

from typing import List, Optional, Set

import torch

from models.memflow.pipeline.diversity_sampling import DiversitySamplingPipeline


class NoiseDiversitySamplingPipeline(DiversitySamplingPipeline):
    """Interactive streaming pipeline with latent noise injection for diversity."""

    def __init__(self, *args, **kwargs) -> None:
        self.noise_injection_scale = float(kwargs.pop("noise_injection_scale", 0.05))
        self.noise_injection_relative = bool(kwargs.pop("noise_injection_relative", False))
        raw_steps = kwargs.pop("noise_inject_at_steps", None)
        if raw_steps is None:
            self.noise_inject_at_steps: List[int] = [0, 1, 2, 3]
        else:
            self.noise_inject_at_steps = [int(x) for x in list(raw_steps)]
        self.inject_noise_every_block = bool(kwargs.pop("inject_noise_every_block", True))
        super().__init__(*args, **kwargs)
        self._injection_generator: Optional[torch.Generator] = None
        self._inject_block_indices: Optional[Set[int]] = None

    def set_injection_generator(self, gen: Optional[torch.Generator]) -> None:
        """RNG for injection eps (per-branch); None uses torch.randn_like."""
        self._injection_generator = gen

    def set_inject_block_indices(self, indices: Optional[Set[int]]) -> None:
        """If set and inject_noise_every_block is False, only these block indices get injection."""
        self._inject_block_indices = indices

    def _maybe_latent_noise_inject(
        self,
        noisy_input: torch.Tensor,
        block_index: int,
        step_idx: int,
    ) -> torch.Tensor:
        if self.inject_noise_every_block:
            use_block = True
        else:
            use_block = (
                self._inject_block_indices is not None
                and block_index in self._inject_block_indices
            )
        if not use_block:
            return noisy_input
        if step_idx not in set(self.noise_inject_at_steps):
            return noisy_input

        gen = self._injection_generator
        if gen is not None:
            eps = torch.randn(
                noisy_input.shape,
                generator=gen,
                device=noisy_input.device,
                dtype=noisy_input.dtype,
            )
        else:
            eps = torch.randn_like(noisy_input)

        if self.noise_injection_relative:
            s = noisy_input.detach().float().std().clamp(min=1e-8)
            scale = (self.noise_injection_scale * s).to(dtype=noisy_input.dtype)
        else:
            scale = torch.tensor(
                self.noise_injection_scale,
                dtype=noisy_input.dtype,
                device=noisy_input.device,
            )

        return noisy_input + scale * eps
