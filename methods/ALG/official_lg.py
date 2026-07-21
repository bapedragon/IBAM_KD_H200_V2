"""LG feature matching ported from the authors' public tiny-transformers code.

The public implementation is available at https://github.com/lkhl/tiny-transformers
and was audited at commit d2165f74049c906b0afc9f957491960fb3c0cc8b.
ALG keeps this feature objective and changes only when it is applied.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


OFFICIAL_LG_COMMIT = "d2165f74049c906b0afc9f957491960fb3c0cc8b"
TEACHER_CHANNELS = (16, 32, 64)
STUDENT_CHANNELS = 192
STUDENT_BLOCK_INDICES = (0, 6, 11)


class LocalityGuidance(nn.Module):
    """Match three DeiT feature grids to three ResNet56 feature grids.

    This follows ``pycls/models/distill.py`` in the public LG repository:

    * student blocks ``[0, 6, 11]`` and teacher stages ``[0, 1, 2]``;
    * learned 1x1 linear channel projection;
    * both tensors resized bilinearly to the larger spatial grid;
    * the three element-wise mean-squared errors are summed.
    """

    def __init__(
        self,
        student_channels: int = STUDENT_CHANNELS,
        teacher_channels: Sequence[int] = TEACHER_CHANNELS,
    ) -> None:
        super().__init__()
        self.teacher_channels = tuple(int(value) for value in teacher_channels)
        self.projections = nn.ModuleList(
            nn.Conv2d(student_channels, channels, kernel_size=1)
            for channels in self.teacher_channels
        )

    def forward(
        self,
        student_features: Sequence[torch.Tensor],
        teacher_features: Sequence[torch.Tensor],
    ) -> tuple[
        torch.Tensor,
        list[torch.Tensor],
        list[torch.Tensor],
    ]:
        if len(student_features) != len(self.projections):
            raise ValueError(
                f"Expected {len(self.projections)} student features, "
                f"got {len(student_features)}"
            )
        if len(teacher_features) != len(self.projections):
            raise ValueError(
                f"Expected {len(self.projections)} teacher features, "
                f"got {len(teacher_features)}"
            )

        total_loss: torch.Tensor | None = None
        aligned_students: list[torch.Tensor] = []
        aligned_teachers: list[torch.Tensor] = []
        for projection, student_feature, teacher_feature in zip(
            self.projections,
            student_features,
            teacher_features,
            strict=True,
        ):
            student_feature = projection(student_feature)
            target_size = (
                max(student_feature.shape[-2], teacher_feature.shape[-2]),
                max(student_feature.shape[-1], teacher_feature.shape[-1]),
            )
            student_feature = F.interpolate(
                student_feature,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )
            teacher_feature = F.interpolate(
                teacher_feature,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )
            stage_loss = F.mse_loss(student_feature, teacher_feature)
            total_loss = stage_loss if total_loss is None else total_loss + stage_loss
            aligned_students.append(student_feature)
            aligned_teachers.append(teacher_feature)

        if total_loss is None:
            raise RuntimeError("Locality guidance requires at least one feature stage")
        return total_loss, aligned_students, aligned_teachers
