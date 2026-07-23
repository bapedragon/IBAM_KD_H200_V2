"""Ours V2 with explicit learnable 2D relative-position attention.

The original :mod:`methods.Ours` package remains unchanged.  V2 keeps its
aggregation, projection, deformable enhancement, loss, and resize behavior,
but makes the spatial relation used by cross-attention explicit:

    score(p, q) = Q_p K_q^T / sqrt(d) + b_h(dx(p, q), dy(p, q))

The bias is zero-initialized, so V2 starts from the exact content-attention
behavior of Ours V1 and can learn a position-dependent correction.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from methods.Ours.ours import DeformableCBAM, TransformerAggregationPooling


FusionFactory = Callable[[int, int], nn.Module]


class RelativePositionBias2D(nn.Module):
    """Head-specific learned bias indexed by a 2D relative displacement."""

    def __init__(self, num_heads: int, max_grid_size: int = 32) -> None:
        super().__init__()
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if max_grid_size <= 0:
            raise ValueError("max_grid_size must be positive")
        self.num_heads = int(num_heads)
        self.max_grid_size = int(max_grid_size)
        relative_extent = 2 * self.max_grid_size - 1
        self.relative_extent = relative_extent
        self.bias_table = nn.Parameter(
            torch.zeros(
                self.num_heads,
                relative_extent * relative_extent,
            )
        )

    def position_index(
        self,
        height: int,
        width: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Return the flattened 2D displacement index for every position pair."""

        if height <= 0 or width <= 0:
            raise ValueError("height and width must be positive")
        if height > self.max_grid_size or width > self.max_grid_size:
            raise ValueError(
                "Grid exceeds the configured relative-position table: "
                f"grid={height}x{width} max={self.max_grid_size}"
            )

        rows = torch.arange(height, device=device)
        columns = torch.arange(width, device=device)
        row_grid, column_grid = torch.meshgrid(rows, columns, indexing="ij")
        coordinates = torch.stack(
            (row_grid.reshape(-1), column_grid.reshape(-1)),
            dim=0,
        )
        displacement = coordinates[:, :, None] - coordinates[:, None, :]
        displacement = displacement + self.max_grid_size - 1
        return (
            displacement[0] * self.relative_extent + displacement[1]
        ).to(dtype=torch.long)

    def forward(
        self,
        height: int,
        width: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        index = self.position_index(height, width, device)
        positions = height * width
        bias = self.bias_table[:, index.reshape(-1)].reshape(
            self.num_heads,
            positions,
            positions,
        )
        return bias.unsqueeze(0).to(dtype=dtype)


class GridAwareCrossAttention(nn.Module):
    """Global cross-attention with an explicit head-specific 2D relation."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        dropout: float = 0.0,
        reduction_ratio: int = 16,
        spatial_kernel_size: int = 5,
        qkv_kernel_size: int = 1,
        max_grid_size: int = 32,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim**-0.5
        self.cbam = DeformableCBAM(
            self.embed_dim,
            reduction_ratio=reduction_ratio,
            spatial_kernel_size=spatial_kernel_size,
        )
        padding = qkv_kernel_size // 2
        self.q_conv = nn.Conv2d(
            self.embed_dim,
            self.embed_dim,
            qkv_kernel_size,
            padding=padding,
            bias=True,
        )
        self.k_conv = nn.Conv2d(
            self.embed_dim,
            self.embed_dim,
            qkv_kernel_size,
            padding=padding,
            bias=True,
        )
        self.v_conv = nn.Conv2d(
            self.embed_dim,
            self.embed_dim,
            qkv_kernel_size,
            padding=padding,
            bias=True,
        )
        self.out_conv = nn.Conv2d(
            self.embed_dim,
            self.embed_dim,
            kernel_size=1,
            bias=True,
        )
        self.relative_position_bias = RelativePositionBias2D(
            self.num_heads,
            max_grid_size=max_grid_size,
        )
        self.dropout = nn.Dropout(dropout)

    def split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, positions, channels = tensor.shape
        if channels != self.embed_dim:
            raise ValueError(
                f"Expected token dimension {self.embed_dim}, got {channels}"
            )
        return tensor.reshape(
            batch,
            positions,
            self.num_heads,
            self.head_dim,
        ).permute(0, 2, 1, 3)

    def forward(
        self,
        student_feature: torch.Tensor,
        teacher_feature: torch.Tensor,
    ) -> torch.Tensor:
        if teacher_feature.shape != student_feature.shape:
            raise ValueError(
                "Teacher and aligned student features must have the same shape: "
                f"teacher={tuple(teacher_feature.shape)} "
                f"student={tuple(student_feature.shape)}"
            )

        batch, channels, height, width = student_feature.shape
        enhanced_student = self.cbam(student_feature)
        query = self.split_heads(
            self.q_conv(enhanced_student).flatten(2).transpose(1, 2)
        )
        key = self.split_heads(
            self.k_conv(teacher_feature).flatten(2).transpose(1, 2)
        )
        value = self.split_heads(
            self.v_conv(teacher_feature).flatten(2).transpose(1, 2)
        )

        scores = (query @ key.transpose(-2, -1)) * self.scale
        scores = scores + self.relative_position_bias(
            height,
            width,
            device=scores.device,
            dtype=scores.dtype,
        )
        attention = self.dropout(torch.softmax(scores, dim=-1))
        outputs = (attention @ value).transpose(1, 2).reshape(
            batch,
            height * width,
            channels,
        )
        outputs = outputs.transpose(1, 2).reshape(
            batch,
            channels,
            height,
            width,
        )
        return self.out_conv(outputs)


class Ours(nn.Module):
    """Ours V2 with stage-specific aggregation and position-aware fusion."""

    def __init__(
        self,
        student_channels: int = 192,
        teacher_channels: Sequence[int] = (16, 32, 64),
        num_student_blocks: int = 12,
        num_heads: int = 4,
        spatial_kernel_size: int = 5,
        grid_resize_mode: str = "larger",
        max_grid_size: int = 32,
        fusion_factory: FusionFactory | None = None,
    ) -> None:
        super().__init__()
        if grid_resize_mode not in {"teacher", "larger"}:
            raise ValueError(
                "grid_resize_mode must be either 'teacher' or 'larger'"
            )
        self.teacher_channels = tuple(int(value) for value in teacher_channels)
        self.grid_resize_mode = grid_resize_mode
        self.aggregation = TransformerAggregationPooling(
            num_transformer=num_student_blocks,
            num_cnn=len(self.teacher_channels),
        )
        self.projections = nn.ModuleList(
            nn.Conv2d(student_channels, channels, kernel_size=1)
            for channels in self.teacher_channels
        )
        if fusion_factory is None:
            fusion_factory = lambda channels, _stage: GridAwareCrossAttention(
                channels,
                num_heads=num_heads,
                spatial_kernel_size=spatial_kernel_size,
                qkv_kernel_size=1,
                max_grid_size=max_grid_size,
            )
        self.fusion_blocks = nn.ModuleList(
            fusion_factory(channels, stage)
            for stage, channels in enumerate(self.teacher_channels)
        )
        self.register_buffer(
            "ours_v2_relative_position_enabled",
            torch.tensor(True, dtype=torch.bool),
            persistent=True,
        )

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
        if len(teacher_features) != len(self.teacher_channels):
            raise ValueError(
                f"Expected {len(self.teacher_channels)} teacher stages, "
                f"got {len(teacher_features)}"
            )
        aggregated = self.aggregation(student_features)
        aligned_features: list[torch.Tensor] = []
        fused_features: list[torch.Tensor] = []
        target_features: list[torch.Tensor] = []
        alignment_loss = aggregated.new_zeros(())
        fusion_loss = aggregated.new_zeros(())

        for stage, (teacher_feature, projection, fusion) in enumerate(
            zip(teacher_features, self.projections, self.fusion_blocks, strict=True)
        ):
            aligned = projection(aggregated[:, stage])
            if self.grid_resize_mode == "larger":
                target_size = (
                    max(aligned.shape[-2], teacher_feature.shape[-2]),
                    max(aligned.shape[-1], teacher_feature.shape[-1]),
                )
            else:
                target_size = teacher_feature.shape[-2:]
            if aligned.shape[-2:] != target_size:
                aligned = F.interpolate(
                    aligned,
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                )
            if teacher_feature.shape[-2:] != target_size:
                teacher_feature = F.interpolate(
                    teacher_feature,
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                )
            if aligned.shape[1] != teacher_feature.shape[1]:
                raise RuntimeError(
                    f"Stage {stage} channel mismatch after projection: "
                    f"aligned={aligned.shape[1]} teacher={teacher_feature.shape[1]}"
                )

            fused = fusion(aligned, teacher_feature)
            alignment_loss = alignment_loss + F.mse_loss(
                aligned.float(),
                teacher_feature.float(),
            )
            fusion_loss = fusion_loss + F.mse_loss(
                fused.float(),
                teacher_feature.float(),
            )
            aligned_features.append(aligned)
            fused_features.append(fused)
            target_features.append(teacher_feature)

        return (
            alignment_loss,
            fusion_loss,
            aligned_features,
            fused_features,
            target_features,
        )
