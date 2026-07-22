"""Grid-permuted Ours variant for the Table 4 attribution control.

This module deliberately changes only the spatial correspondence of the
teacher targets.  The parent Ours implementation still performs aggregation,
projection, deformable enhancement, convolutional cross-attention, and both
feature losses.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F

from methods.Ours.ours import Ours


def log(message: str) -> None:
    print(message, flush=True)


class GridPermutedOurs(Ours):
    """Apply one deterministic, fixed spatial permutation per teacher stage.

    A single run-level seed deterministically produces one permutation for
    each stage because the active stage grids contain different numbers of
    positions (32x32, 16x16, and 14x14).  Each permutation is shared by every
    sample and remains unchanged for every epoch in the run.
    """

    def __init__(
        self,
        *args: object,
        permutation_seed: int = 1,
        **kwargs: object,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.permutation_seed = int(permutation_seed)
        self.register_buffer(
            "table4_permutation_seed",
            torch.tensor(self.permutation_seed, dtype=torch.int64),
            persistent=True,
        )
        self._permutation_shapes: dict[int, tuple[int, int]] = {}
        log(
            "[TABLE4_GRID_PERMUTATION] enabled=True "
            f"run_seed={self.permutation_seed} fixed_across_samples_and_epochs=True "
            "application=after_grid_resize targets=K,V,L_align,L_fuse"
        )

    def _fixed_permutation(
        self,
        stage: int,
        height: int,
        width: int,
        device: torch.device,
    ) -> torch.Tensor:
        buffer_name = f"table4_grid_permutation_stage_{stage}"
        expected_shape = (height, width)
        if hasattr(self, buffer_name):
            if self._permutation_shapes[stage] != expected_shape:
                raise RuntimeError(
                    "A fixed Table 4 permutation cannot change shape within a run: "
                    f"stage={stage} first={self._permutation_shapes[stage]} "
                    f"current={expected_shape}"
                )
            return getattr(self, buffer_name).to(device=device)

        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.permutation_seed + stage)
        permutation = torch.randperm(
            height * width,
            generator=generator,
            device="cpu",
        ).to(device=device)
        self.register_buffer(buffer_name, permutation, persistent=True)
        self._permutation_shapes[stage] = expected_shape
        log(
            "[TABLE4_GRID_PERMUTATION] "
            f"stage={stage + 1} grid={height}x{width} "
            f"positions={height * width} stage_seed={self.permutation_seed + stage}"
        )
        return permutation

    def _resize_and_permute_teacher(
        self,
        student_features: Sequence[torch.Tensor],
        teacher_features: Sequence[torch.Tensor],
    ) -> list[torch.Tensor]:
        if not student_features:
            raise ValueError("At least one student feature is required")
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
            permutation = self._fixed_permutation(
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
            teacher_feature = teacher_feature.reshape(batch, channels, height, width)
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
        # The exact same permuted tensors become K/V inside the parent fusion
        # blocks and the positionwise targets of both alignment and fusion MSE.
        return super().forward(student_features, permuted_teacher)
