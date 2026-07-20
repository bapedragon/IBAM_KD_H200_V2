"""ReviewKD ABF and hierarchical context loss.

The implementation follows the behavior of the authors' official ReviewKD
repository at the pinned commit recorded in ``methods/ReviewKD/README.md``.
The module is kept independent of any particular CNN or ViT backbone.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionBasedFusion(nn.Module):
    """Transform one student feature and optionally fuse a deeper review."""

    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        *,
        fuse: bool,
    ) -> None:
        super().__init__()
        self.fuse = fuse
        self.input_projection = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
        )
        self.output_projection = nn.Sequential(
            nn.Conv2d(
                mid_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        )
        self.attention = (
            nn.Sequential(
                nn.Conv2d(mid_channels * 2, 2, kernel_size=1),
                nn.Sigmoid(),
            )
            if fuse
            else None
        )
        nn.init.kaiming_uniform_(self.input_projection[0].weight, a=1)
        nn.init.kaiming_uniform_(self.output_projection[0].weight, a=1)

    def forward(
        self,
        feature: torch.Tensor,
        deeper_review: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        transformed = self.input_projection(feature)
        if self.fuse:
            if deeper_review is None or self.attention is None:
                raise RuntimeError("A deeper review feature is required for ABF fusion")
            if deeper_review.shape[-2:] != transformed.shape[-2:]:
                deeper_review = F.interpolate(
                    deeper_review,
                    size=transformed.shape[-2:],
                    mode="nearest",
                )
            gates = self.attention(torch.cat((transformed, deeper_review), dim=1))
            transformed = (
                transformed * gates[:, 0:1]
                + deeper_review * gates[:, 1:2]
            )
        output = self.output_projection(transformed)
        return output, transformed


class ReviewKDAdapter(nn.Module):
    """Apply official attention-based review from deep to shallow features."""

    def __init__(
        self,
        in_channels: Sequence[int],
        out_channels: Sequence[int],
        mid_channels: int,
    ) -> None:
        super().__init__()
        if len(in_channels) != len(out_channels):
            raise ValueError("ReviewKD input/output feature counts must match")
        if len(in_channels) < 2:
            raise ValueError("ReviewKD requires at least two feature levels")

        shallow_to_deep = [
            AttentionBasedFusion(
                input_channels,
                mid_channels,
                output_channels,
                fuse=index < len(in_channels) - 1,
            )
            for index, (input_channels, output_channels) in enumerate(
                zip(in_channels, out_channels)
            )
        ]
        self.deep_to_shallow = nn.ModuleList(reversed(shallow_to_deep))

    def forward(
        self,
        student_features: Sequence[torch.Tensor],
    ) -> list[torch.Tensor]:
        if len(student_features) != len(self.deep_to_shallow):
            raise RuntimeError(
                "ReviewKD feature count mismatch: "
                f"features={len(student_features)} "
                f"abfs={len(self.deep_to_shallow)}"
            )

        reviewed_deep_to_shallow: list[torch.Tensor] = []
        residual: torch.Tensor | None = None
        for feature, abf in zip(
            reversed(student_features),
            self.deep_to_shallow,
        ):
            output, residual = abf(feature, residual)
            reviewed_deep_to_shallow.append(output)
        return list(reversed(reviewed_deep_to_shallow))


def hierarchical_context_loss(
    student_features: Sequence[torch.Tensor],
    teacher_features: Sequence[torch.Tensor],
) -> torch.Tensor:
    """Official HCL: full-resolution MSE plus pooled 4, 2, and 1 grids."""

    if len(student_features) != len(teacher_features):
        raise RuntimeError(
            "HCL feature count mismatch: "
            f"student={len(student_features)} teacher={len(teacher_features)}"
        )
    total_loss: torch.Tensor | None = None
    for student_feature, teacher_feature in zip(
        student_features,
        teacher_features,
    ):
        if student_feature.shape != teacher_feature.shape:
            raise RuntimeError(
                "HCL feature shape mismatch: "
                f"student={tuple(student_feature.shape)} "
                f"teacher={tuple(teacher_feature.shape)}"
            )

        level_loss = F.mse_loss(student_feature, teacher_feature)
        weight = 1.0
        normalizer = 1.0
        height = student_feature.shape[-2]
        for grid in (4, 2, 1):
            if grid >= height:
                continue
            weight *= 0.5
            level_loss = level_loss + weight * F.mse_loss(
                F.adaptive_avg_pool2d(student_feature, (grid, grid)),
                F.adaptive_avg_pool2d(teacher_feature, (grid, grid)),
            )
            normalizer += weight
        level_loss = level_loss / normalizer
        total_loss = level_loss if total_loss is None else total_loss + level_loss

    if total_loss is None:
        raise RuntimeError("HCL received no features")
    return total_loss
