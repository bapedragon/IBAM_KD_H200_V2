"""Standalone port of the authors' official MGD classification loss.

Modified from ``cls/mmcls/distillation/losses/mgd.py`` at the pinned commit
recorded in ``methods/MGD/README.md``. The MMClassification registry dependency
was removed, type checks were added, and the loss behavior was left unchanged:
channel-wise random masking, optional 1x1 alignment, a two-layer convolutional
generator, summed MSE divided by batch size, and the official alpha weight.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MGDLoss(nn.Module):
    """Official classification MGD loss without the MMClassification registry."""

    def __init__(
        self,
        student_channels: int,
        teacher_channels: int,
        *,
        alpha_mgd: float = 0.00007,
        lambda_mgd: float = 0.15,
    ) -> None:
        super().__init__()
        self.alpha_mgd = alpha_mgd
        self.lambda_mgd = lambda_mgd
        self.align = (
            nn.Conv2d(student_channels, teacher_channels, kernel_size=1)
            if student_channels != teacher_channels
            else nn.Identity()
        )
        self.generation = nn.Sequential(
            nn.Conv2d(teacher_channels, teacher_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(teacher_channels, teacher_channels, kernel_size=3, padding=1),
        )

    def align_feature(self, student_feature: torch.Tensor) -> torch.Tensor:
        return self.align(student_feature)

    def forward(
        self,
        student_feature: torch.Tensor,
        teacher_feature: torch.Tensor,
    ) -> torch.Tensor:
        if student_feature.shape[-2:] != teacher_feature.shape[-2:]:
            raise RuntimeError(
                "MGD spatial mismatch: "
                f"student={tuple(student_feature.shape)} "
                f"teacher={tuple(teacher_feature.shape)}"
            )
        student_feature = self.align_feature(student_feature)
        if student_feature.shape != teacher_feature.shape:
            raise RuntimeError(
                "MGD aligned feature mismatch: "
                f"student={tuple(student_feature.shape)} "
                f"teacher={tuple(teacher_feature.shape)}"
            )
        return self.get_dis_loss(student_feature, teacher_feature) * self.alpha_mgd

    def get_dis_loss(
        self,
        student_feature: torch.Tensor,
        teacher_feature: torch.Tensor,
    ) -> torch.Tensor:
        batch, channels, _, _ = teacher_feature.shape
        random_mask = torch.rand(
            (batch, channels, 1, 1),
            device=student_feature.device,
            dtype=student_feature.dtype,
        )
        mask = torch.where(random_mask < self.lambda_mgd, 0.0, 1.0)
        generated_feature = self.generation(student_feature * mask)
        return F.mse_loss(
            generated_feature,
            teacher_feature,
            reduction="sum",
        ) / batch
