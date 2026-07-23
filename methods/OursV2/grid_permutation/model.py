"""K/V-only fixed-grid permutation for position-aware Ours V2.

Unlike the historical Ours V1 control, this intervention does not permute the
alignment or fusion MSE targets.  It changes only which teacher content is
associated with each 2D key/value coordinate inside cross-attention.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from methods.OursV2.ours import GridAwareCrossAttention, Ours


class GridPermutedCrossAttention(GridAwareCrossAttention):
    """Jointly permute teacher K/V content while keeping 2D bias coordinates."""

    def __init__(
        self,
        *args: object,
        stage: int,
        permutation_seed: int = 1,
        **kwargs: object,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.stage = int(stage)
        self.permutation_seed = int(permutation_seed)
        self.register_buffer(
            "grid_permutation_seed",
            torch.tensor(self.permutation_seed, dtype=torch.int64),
            persistent=True,
        )
        self._permutation_cache: dict[tuple[int, int], torch.Tensor] = {}

    def fixed_permutation(
        self,
        height: int,
        width: int,
        device: torch.device,
    ) -> torch.Tensor:
        shape = (int(height), int(width))
        if shape not in self._permutation_cache:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(self.permutation_seed + self.stage)
            self._permutation_cache[shape] = torch.randperm(
                height * width,
                generator=generator,
                device="cpu",
            )
        return self._permutation_cache[shape].to(device=device)

    def permute_teacher(self, teacher_feature: torch.Tensor) -> torch.Tensor:
        if teacher_feature.ndim != 4:
            raise ValueError(
                "Teacher feature must be BCHW, got "
                f"{tuple(teacher_feature.shape)}"
            )
        batch, channels, height, width = teacher_feature.shape
        permutation = self.fixed_permutation(
            height,
            width,
            teacher_feature.device,
        )
        return teacher_feature.flatten(2).index_select(2, permutation).reshape(
            batch,
            channels,
            height,
            width,
        )

    def forward(
        self,
        student_feature: torch.Tensor,
        teacher_feature: torch.Tensor,
    ) -> torch.Tensor:
        return super().forward(
            student_feature,
            self.permute_teacher(teacher_feature),
        )


class GridPermutedOurs(Ours):
    """Ours V2 with only fusion K/V grid content permuted."""

    def __init__(
        self,
        *args: object,
        student_channels: int = 192,
        teacher_channels: Sequence[int] = (16, 32, 64),
        num_student_blocks: int = 12,
        num_heads: int = 4,
        spatial_kernel_size: int = 5,
        grid_resize_mode: str = "larger",
        max_grid_size: int = 32,
        permutation_seed: int = 1,
        **kwargs: object,
    ) -> None:
        def fusion_factory(channels: int, stage: int) -> nn.Module:
            return GridPermutedCrossAttention(
                channels,
                num_heads=num_heads,
                spatial_kernel_size=spatial_kernel_size,
                qkv_kernel_size=1,
                max_grid_size=max_grid_size,
                stage=stage,
                permutation_seed=permutation_seed,
            )

        super().__init__(
            *args,
            student_channels=student_channels,
            teacher_channels=teacher_channels,
            num_student_blocks=num_student_blocks,
            num_heads=num_heads,
            spatial_kernel_size=spatial_kernel_size,
            grid_resize_mode=grid_resize_mode,
            max_grid_size=max_grid_size,
            fusion_factory=fusion_factory,
            **kwargs,
        )
        self.register_buffer(
            "ours_v2_grid_permutation_enabled",
            torch.tensor(True, dtype=torch.bool),
            persistent=True,
        )
        print(
            "[OURS_V2_GRID_PERMUTATION] enabled=True "
            f"seed={permutation_seed} scope=fusion_teacher_KV_only "
            "relative_position_bias=kept mse_targets=unpermuted",
            flush=True,
        )
