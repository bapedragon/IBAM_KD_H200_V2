"""Convert teacher CNN maps into the student transformer representation.

This isolated Ours V1 variant reverses the original feature-alignment
direction. The original projects each 192-channel student transformer map
into the channel/grid geometry of one teacher CNN stage. Here every teacher
stage is resized to the student's single 14x14 patch grid, projected from
16/32/64 channels to the transformer dimension 192, and flattened to a BND
sequence. There is no spatial permutation and no explicit 2D position term.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from methods.Ours.ours import (
    DeformableCBAM,
    TransformerAggregationPooling,
)


class TransformerRepresentationCrossAttention(nn.Module):
    """Ours V1 content cross-attention in the common transformer dimension."""

    def __init__(
        self,
        embed_dim: int = 192,
        num_heads: int = 4,
        dropout: float = 0.0,
        reduction_ratio: int = 16,
        spatial_kernel_size: int = 5,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim**-0.5

        # This is the original Ours V1 student-side enhancement. It runs
        # before the student map is flattened; the teacher is not processed
        # by a 2D attention module after its transformer projection.
        self.cbam = DeformableCBAM(
            self.embed_dim,
            reduction_ratio=reduction_ratio,
            spatial_kernel_size=spatial_kernel_size,
        )
        self.q_linear = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.k_linear = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.v_linear = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.out_linear = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
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
        student_map: torch.Tensor,
        teacher_tokens: torch.Tensor,
    ) -> torch.Tensor:
        if student_map.ndim != 4:
            raise ValueError(
                f"Student feature must be BCHW, got {tuple(student_map.shape)}"
            )
        batch, channels, height, width = student_map.shape
        expected_teacher_shape = (batch, height * width, channels)
        if tuple(teacher_tokens.shape) != expected_teacher_shape:
            raise ValueError(
                "Projected teacher tokens must match the transformer sequence: "
                f"expected={expected_teacher_shape} "
                f"actual={tuple(teacher_tokens.shape)}"
            )

        student_tokens = self.cbam(student_map).flatten(2).transpose(1, 2)
        query = self.split_heads(self.q_linear(student_tokens))
        key = self.split_heads(self.k_linear(teacher_tokens))
        value = self.split_heads(self.v_linear(teacher_tokens))
        attention = self.dropout(
            torch.softmax(
                (query @ key.transpose(-2, -1)) * self.scale,
                dim=-1,
            )
        )
        output_tokens = (attention @ value).transpose(1, 2).reshape(
            batch,
            height * width,
            channels,
        )
        return self.out_linear(output_tokens)


class CNNToTransformerSpatialCollapseOurs(nn.Module):
    """Ours V1 with the teacher projected into transformer token space."""

    def __init__(
        self,
        student_channels: int = 192,
        teacher_channels: Sequence[int] = (16, 32, 64),
        num_student_blocks: int = 12,
        num_heads: int = 4,
        spatial_kernel_size: int = 5,
        grid_resize_mode: str = "larger",
    ) -> None:
        super().__init__()
        if grid_resize_mode not in {"teacher", "larger"}:
            raise ValueError(
                "grid_resize_mode must be either 'teacher' or 'larger'"
            )
        self.student_channels = int(student_channels)
        self.teacher_channels = tuple(int(value) for value in teacher_channels)
        # Kept only for compatibility with the locked Ours V1 trainer. This
        # variant always uses the student transformer patch grid.
        self.base_grid_resize_mode = str(grid_resize_mode)
        self.aggregation = TransformerAggregationPooling(
            num_transformer=num_student_blocks,
            num_cnn=len(self.teacher_channels),
        )

        # 핵심 1: 기존 Student 192→Teacher C 투영을 Teacher C→192로 반전한다.
        self.teacher_to_transformer_projections = nn.ModuleList(
            nn.Conv2d(channels, self.student_channels, kernel_size=1)
            for channels in self.teacher_channels
        )
        # 변환된 세 stage 모두 Transformer dimension 192에서 V1 attention을 한다.
        self.fusion_blocks = nn.ModuleList(
            TransformerRepresentationCrossAttention(
                self.student_channels,
                num_heads=num_heads,
                spatial_kernel_size=spatial_kernel_size,
            )
            for _ in self.teacher_channels
        )
        self.register_buffer(
            "cnn_to_transformer_spatial_collapse_enabled",
            torch.tensor(True, dtype=torch.bool),
            persistent=True,
        )
        print(
            "[CNN_TO_TRANSFORMER_SPATIAL_COLLAPSE] enabled=True "
            "projection_direction=teacher_C_to_student_D "
            f"student_dimension={self.student_channels} "
            "target_grid=student_patch_grid "
            "representation=BND no_permutation=True "
            "explicit_2d_position=False",
            flush=True,
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
        if not student_features:
            raise ValueError("At least one student feature is required")

        aggregated = self.aggregation(student_features)
        student_token_features: list[torch.Tensor] = []
        fused_token_features: list[torch.Tensor] = []
        teacher_token_targets: list[torch.Tensor] = []
        alignment_loss = aggregated.new_zeros(())
        fusion_loss = aggregated.new_zeros(())

        for stage, (teacher_feature, projection, fusion) in enumerate(
            zip(
                teacher_features,
                self.teacher_to_transformer_projections,
                self.fusion_blocks,
                strict=True,
            )
        ):
            student_map = aggregated[:, stage]
            if student_map.shape[1] != self.student_channels:
                raise RuntimeError(
                    f"Stage {stage} student dimension mismatch: "
                    f"expected={self.student_channels} "
                    f"actual={student_map.shape[1]}"
                )
            transformer_grid = student_map.shape[-2:]
            # 핵심 2: 모든 Teacher stage를 Student patch grid(14×14)에 맞춘다.
            if teacher_feature.shape[-2:] != transformer_grid:
                teacher_feature = F.interpolate(
                    teacher_feature,
                    size=transformer_grid,
                    mode="bilinear",
                    align_corners=False,
                )
            teacher_map = projection(teacher_feature)
            # 핵심 3: shuffle/permutation 없이 BCHW를 BND로 reshape만 한다.
            student_tokens = student_map.flatten(2).transpose(1, 2)
            teacher_tokens = teacher_map.flatten(2).transpose(1, 2)
            if student_tokens.shape != teacher_tokens.shape:
                raise RuntimeError(
                    f"Stage {stage} token mismatch after reverse projection: "
                    f"student={tuple(student_tokens.shape)} "
                    f"teacher={tuple(teacher_tokens.shape)}"
                )

            fused_tokens = fusion(student_map, teacher_tokens)
            # 핵심 4: L_align과 L_fuse를 같은 Transformer 표현에서 계산한다.
            alignment_loss = alignment_loss + F.mse_loss(
                student_tokens.float(),
                teacher_tokens.float(),
            )
            fusion_loss = fusion_loss + F.mse_loss(
                fused_tokens.float(),
                teacher_tokens.float(),
            )
            student_token_features.append(student_tokens)
            fused_token_features.append(fused_tokens)
            teacher_token_targets.append(teacher_tokens)

        return (
            alignment_loss,
            fusion_loss,
            student_token_features,
            fused_token_features,
            teacher_token_targets,
        )
