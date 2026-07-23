"""Ours control that shuffles teacher positions only inside local 2x2 windows."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F

from methods.Ours.ours import Ours


def log(message: str) -> None:
    print(message, flush=True)


class LocalPatchPermutedOurs(Ours):
    """Apply one fixed spatial shuffle inside every non-overlapping window.

    The same locally permuted teacher tensor is passed to the parent Ours
    module, so it remains the shared source for K, V, L_align, and L_fuse.
    """

    def __init__(
        self,
        *args: object,
        permutation_seed: int = 1,
        local_patch_size: int = 2,
        **kwargs: object,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.permutation_seed = int(permutation_seed)
        self.local_patch_size = int(local_patch_size)
        if self.local_patch_size < 2:
            raise ValueError("local_patch_size must be at least 2")
        self.register_buffer(
            "table4_local_permutation_seed",
            torch.tensor(self.permutation_seed, dtype=torch.int64),
            persistent=True,
        )
        self.register_buffer(
            "table4_local_patch_size",
            torch.tensor(self.local_patch_size, dtype=torch.int64),
            persistent=True,
        )
        self._permutation_shapes: dict[int, tuple[int, int]] = {}
        log(
            "[TABLE4_LOCAL_PATCH_PERMUTATION] enabled=True "
            f"run_seed={self.permutation_seed} "
            f"window={self.local_patch_size}x{self.local_patch_size} "
            "scope=within_each_non_overlapping_window "
            "fixed_across_samples_and_epochs=True "
            "application=after_grid_resize targets=K,V,L_align,L_fuse"
        )

    def _fixed_local_permutation(
        self,
        stage: int,
        height: int,
        width: int,
        device: torch.device,
    ) -> torch.Tensor:
        buffer_name = f"table4_local_grid_permutation_stage_{stage}"
        expected_shape = (height, width)
        if hasattr(self, buffer_name):
            if self._permutation_shapes[stage] != expected_shape:
                raise RuntimeError(
                    "A fixed local permutation cannot change shape within a run: "
                    f"stage={stage} first={self._permutation_shapes[stage]} "
                    f"current={expected_shape}"
                )
            return getattr(self, buffer_name).to(device=device)

        patch = self.local_patch_size
        if height % patch or width % patch:
            raise ValueError(
                "Grid dimensions must be divisible by local_patch_size: "
                f"grid={height}x{width} patch={patch}"
            )

        generator = torch.Generator(device="cpu")
        stage_seed = self.permutation_seed + stage
        generator.manual_seed(stage_seed)
        grid = torch.arange(height * width, dtype=torch.int64).reshape(
            height,
            width,
        )
        permutation_grid = torch.empty_like(grid)
        identity = torch.arange(patch * patch, dtype=torch.int64)
        window_count = 0

        for top in range(0, height, patch):
            for left in range(0, width, patch):
                sources = grid[
                    top : top + patch,
                    left : left + patch,
                ].reshape(-1)
                # Use a derangement so the small local intervention does not
                # leave an expected 25% of positions unchanged in a 2x2 tile.
                while True:
                    order = torch.randperm(
                        patch * patch,
                        generator=generator,
                        device="cpu",
                    )
                    if torch.all(order != identity):
                        break
                permutation_grid[
                    top : top + patch,
                    left : left + patch,
                ] = sources.index_select(0, order).reshape(patch, patch)
                window_count += 1

        permutation = permutation_grid.reshape(-1).to(device=device)
        destination = torch.arange(
            height * width,
            dtype=torch.int64,
            device=device,
        )
        source_rows = torch.div(permutation, width, rounding_mode="floor")
        source_cols = permutation.remainder(width)
        destination_rows = torch.div(destination, width, rounding_mode="floor")
        destination_cols = destination.remainder(width)
        if not torch.equal(
            torch.div(source_rows, patch, rounding_mode="floor"),
            torch.div(destination_rows, patch, rounding_mode="floor"),
        ) or not torch.equal(
            torch.div(source_cols, patch, rounding_mode="floor"),
            torch.div(destination_cols, patch, rounding_mode="floor"),
        ):
            raise RuntimeError("Local permutation moved a value across windows")

        self.register_buffer(buffer_name, permutation, persistent=True)
        self._permutation_shapes[stage] = expected_shape
        changed_positions = int((permutation != destination).sum().item())
        log(
            "[TABLE4_LOCAL_PATCH_PERMUTATION] "
            f"stage={stage + 1} grid={height}x{width} "
            f"window={patch}x{patch} windows={window_count} "
            "derangement=True "
            f"changed_positions={changed_positions}/{height * width} "
            f"stage_seed={stage_seed}"
        )
        return permutation

    def _resize_and_permute_teacher(
        self,
        student_features: Sequence[torch.Tensor],
        teacher_features: Sequence[torch.Tensor],
    ) -> list[torch.Tensor]:
        if not student_features:
            raise ValueError("At least one student feature is required")
        if len(teacher_features) != len(self.teacher_channels):
            raise ValueError(
                f"Expected {len(self.teacher_channels)} teacher stages, "
                f"got {len(teacher_features)}"
            )

        student_height, student_width = student_features[0].shape[-2:]
        permuted_features: list[torch.Tensor] = []
        for stage, teacher_feature in enumerate(teacher_features):
            if teacher_feature.ndim != 4:
                raise ValueError(
                    "Teacher features must be BCHW tensors: "
                    f"stage={stage} shape={tuple(teacher_feature.shape)}"
                )
            teacher_height, teacher_width = teacher_feature.shape[-2:]
            if self.grid_resize_mode == "larger":
                target_size = (
                    max(student_height, teacher_height),
                    max(student_width, teacher_width),
                )
            else:
                target_size = (teacher_height, teacher_width)
            if teacher_feature.shape[-2:] != target_size:
                teacher_feature = F.interpolate(
                    teacher_feature,
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                )

            height, width = teacher_feature.shape[-2:]
            permutation = self._fixed_local_permutation(
                stage,
                height,
                width,
                teacher_feature.device,
            )
            batch, channels = teacher_feature.shape[:2]
            teacher_feature = teacher_feature.flatten(2).index_select(
                2,
                permutation,
            )
            teacher_feature = teacher_feature.reshape(
                batch,
                channels,
                height,
                width,
            )
            permuted_features.append(teacher_feature)
        return permuted_features

    def forward(
        self,
        student_features: Sequence[torch.Tensor],
        teacher_features: Sequence[torch.Tensor],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        list[torch.Tensor],
        list[torch.Tensor],
        list[torch.Tensor],
    ]:
        permuted_teacher = self._resize_and_permute_teacher(
            student_features,
            teacher_features,
        )
        # Parent Ours receives one tensor per stage, so K and V are identical.
        return super().forward(student_features, permuted_teacher)
