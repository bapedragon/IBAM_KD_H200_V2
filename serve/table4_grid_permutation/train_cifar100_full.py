#!/usr/bin/env python3
"""Consolidated, executable snapshot of the completed Table 4 control.

Unlike the modular source in ``methods/Ours/table4_grid_permutation``, this
file expands the supplied Ours model and fixed teacher-grid permutation in one
place. Shared data/model construction and the training loop remain in
``methods.Ours.core`` so this snapshot executes the same pipeline.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


def log(message: str) -> None:
    print(message, flush=True)


class DeformableConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        stride: int = 1,
        padding: int = 2,
        dilation: int = 1,
        bias: bool = False,
    ) -> None:
        super().__init__()
        kernel = (kernel_size, kernel_size)
        self.stride = (stride, stride)
        self.padding = (padding, padding)
        self.dilation = (dilation, dilation)
        sampling_points = kernel[0] * kernel[1]
        self.offset_conv = nn.Conv2d(
            in_channels,
            2 * sampling_points,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=True,
        )
        self.modulator_conv = nn.Conv2d(
            in_channels,
            sampling_points,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=True,
        )
        self.regular_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        nn.init.zeros_(self.offset_conv.weight)
        nn.init.zeros_(self.offset_conv.bias)
        nn.init.zeros_(self.modulator_conv.weight)
        nn.init.zeros_(self.modulator_conv.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        offset = self.offset_conv(inputs)
        modulation = 2.0 * torch.sigmoid(self.modulator_conv(inputs))
        return torchvision.ops.deform_conv2d(
            input=inputs,
            offset=offset,
            weight=self.regular_conv.weight,
            bias=self.regular_conv.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            mask=modulation,
        )


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction_ratio: int = 16) -> None:
        super().__init__()
        hidden_channels = max(1, channels // reduction_ratio)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(
            self.fc(self.avg_pool(inputs)) + self.fc(self.max_pool(inputs))
        )


class DeformableSpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 5) -> None:
        super().__init__()
        self.spatial_conv = DeformableConv2d(
            in_channels=2,
            out_channels=1,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=False,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        average = torch.mean(inputs, dim=1, keepdim=True)
        maximum = torch.max(inputs, dim=1, keepdim=True).values
        return torch.sigmoid(self.spatial_conv(torch.cat((average, maximum), dim=1)))


class DeformableCBAM(nn.Module):
    def __init__(
        self,
        channels: int,
        reduction_ratio: int = 16,
        spatial_kernel_size: int = 5,
    ) -> None:
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction_ratio)
        self.spatial_attention = DeformableSpatialAttention(spatial_kernel_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = inputs * self.channel_attention(inputs)
        return outputs * self.spatial_attention(outputs)


class CBAMConvCrossAttention(nn.Module):
    """Unpermuted student query with teacher key and value."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        dropout: float = 0.0,
        reduction_ratio: int = 16,
        spatial_kernel_size: int = 5,
        qkv_kernel_size: int = 1,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.cbam = DeformableCBAM(
            embed_dim,
            reduction_ratio=reduction_ratio,
            spatial_kernel_size=spatial_kernel_size,
        )
        padding = qkv_kernel_size // 2
        self.q_conv = nn.Conv2d(
            embed_dim, embed_dim, qkv_kernel_size, padding=padding, bias=True
        )
        self.k_conv = nn.Conv2d(
            embed_dim, embed_dim, qkv_kernel_size, padding=padding, bias=True
        )
        self.v_conv = nn.Conv2d(
            embed_dim, embed_dim, qkv_kernel_size, padding=padding, bias=True
        )
        self.out_conv = nn.Conv2d(embed_dim, embed_dim, kernel_size=1, bias=True)
        self.dropout = nn.Dropout(dropout)

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

        # Q: unpermuted student feature after deformable CBAM.
        enhanced_student = self.cbam(student_feature)
        query = self.q_conv(enhanced_student).flatten(2).transpose(1, 2)

        # K and V: the same teacher feature after the fixed grid permutation.
        key = self.k_conv(teacher_feature).flatten(2).transpose(1, 2)
        value = self.v_conv(teacher_feature).flatten(2).transpose(1, 2)

        def split_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.reshape(
                batch, height * width, self.num_heads, self.head_dim
            ).permute(0, 2, 1, 3)

        query = split_heads(query)
        key = split_heads(key)
        value = split_heads(value)
        attention = torch.softmax(
            (query @ key.transpose(-2, -1)) * self.scale,
            dim=-1,
        )
        attention = self.dropout(attention)
        outputs = (attention @ value).transpose(1, 2).reshape(
            batch, height * width, channels
        )
        outputs = outputs.transpose(1, 2).reshape(batch, channels, height, width)
        return self.out_conv(outputs)


class TransformerAggregationPooling(nn.Module):
    """Learn one convex mixture of all student blocks for each CNN stage."""

    def __init__(self, num_transformer: int = 12, num_cnn: int = 3) -> None:
        super().__init__()
        self.weights = nn.Parameter(torch.zeros(num_cnn, num_transformer))

    def forward(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(features) != self.weights.shape[1]:
            raise ValueError(
                f"Expected {self.weights.shape[1]} student features, got {len(features)}"
            )
        stacked = torch.stack(tuple(features), dim=1)
        normalized_weights = torch.softmax(self.weights, dim=-1)
        return torch.einsum("gl,bldhw->bgdhw", normalized_weights, stacked)

    def normalized_weights(self) -> torch.Tensor:
        return torch.softmax(self.weights.detach(), dim=-1)


class GridPermutedOurs(nn.Module):
    """The complete Ours forward path with the Table 4 intervention inline."""

    def __init__(
        self,
        student_channels: int = 192,
        teacher_channels: Sequence[int] = (16, 32, 64),
        num_student_blocks: int = 12,
        num_heads: int = 4,
        spatial_kernel_size: int = 5,
        grid_resize_mode: str = "larger",
        permutation_seed: int = 1,
    ) -> None:
        super().__init__()
        if grid_resize_mode not in {"teacher", "larger"}:
            raise ValueError("grid_resize_mode must be either 'teacher' or 'larger'")
        self.teacher_channels = tuple(int(value) for value in teacher_channels)
        self.grid_resize_mode = grid_resize_mode
        self.permutation_seed = int(permutation_seed)
        self.aggregation = TransformerAggregationPooling(
            num_transformer=num_student_blocks,
            num_cnn=len(self.teacher_channels),
        )
        self.projections = nn.ModuleList(
            nn.Conv2d(student_channels, channels, kernel_size=1)
            for channels in self.teacher_channels
        )
        self.fusion_blocks = nn.ModuleList(
            CBAMConvCrossAttention(
                channels,
                num_heads=num_heads,
                spatial_kernel_size=spatial_kernel_size,
                qkv_kernel_size=1,
            )
            for channels in self.teacher_channels
        )
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
        """Create the exact stage-specific permutation used in the run."""
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
        """Resize each teacher stage, then move complete C-vectors spatially."""
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

            # BCHW -> BC(HW) -> select the same spatial permutation -> BCHW.
            # No channel, sample, label, or student-token permutation occurs.
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
        if len(teacher_features) != len(self.teacher_channels):
            raise ValueError(
                f"Expected {len(self.teacher_channels)} teacher stages, "
                f"got {len(teacher_features)}"
            )

        # Intervention: only teacher spatial correspondence changes.
        permuted_teacher = self._resize_and_permute_teacher(
            student_features,
            teacher_features,
        )

        # Original Ours execution continues below, now written explicitly.
        aggregated = self.aggregation(student_features)
        aligned_features: list[torch.Tensor] = []
        fused_features: list[torch.Tensor] = []
        target_features: list[torch.Tensor] = []
        alignment_loss = aggregated.new_zeros(())
        fusion_loss = aggregated.new_zeros(())

        for stage, (teacher_feature, projection, fusion) in enumerate(
            zip(
                permuted_teacher,
                self.projections,
                self.fusion_blocks,
                strict=True,
            )
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

            # Q is aligned student; K and V are the same permuted teacher.
            fused = fusion(aligned, teacher_feature)

            # Both position-wise targets are also the same permuted teacher.
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


PERMUTATION_SEED = 1
PROTOCOL_DEFAULTS = (
    ("--protocol-name", "table4_grid_permuted_cifar100_researcher_sync_v1"),
    ("--student-epochs", "300"),
    ("--batch-size", "64"),
    ("--eval-batch-size", "200"),
    ("--lr", "0.0005"),
    ("--min-lr", "0.000005"),
    ("--weight-decay", "0.05"),
    ("--warmup-epochs", "20"),
    ("--warmup-factor", "0.001"),
    ("--label-smoothing", "0.0"),
    ("--drop-path-rate", "0.1"),
    ("--seed", "1"),
    ("--base-protocol", "lg_official"),
    ("--teacher-image-size", "32"),
    ("--beta-schedule", "alg"),
    ("--beta-on", "2.5"),
    ("--alg-threshold", "-0.02"),
    ("--alg-smoothing-window", "50"),
    ("--alg-warmup-epochs", "20"),
    ("--grid-resize-mode", "larger"),
    ("--eval-resize-mode", "direct"),
)


def has_option(option: str) -> bool:
    return any(
        argument == option or argument.startswith(f"{option}=")
        for argument in sys.argv[1:]
    )


def build_grid_permuted_ours(
    *args: object,
    **kwargs: object,
) -> GridPermutedOurs:
    return GridPermutedOurs(
        *args,
        permutation_seed=PERMUTATION_SEED,
        **kwargs,
    )


def main() -> None:
    # Import the shared training infrastructure only for an actual run. This
    # keeps the consolidated model importable for review/regression checks
    # without triggering dependency installation.
    from methods.Ours import core

    if has_option("--dataset"):
        raise SystemExit("This wrapper fixes --dataset cifar100; remove --dataset.")
    sys.argv[1:1] = ["--dataset", "cifar100"]
    for option, value in reversed(PROTOCOL_DEFAULTS):
        if not has_option(option):
            sys.argv[1:1] = [option, value]

    print(
        "[TABLE4_CONTROL] reference_full_top1=82.90% "
        "only_change=fixed_teacher_grid_permutation permutation_seed=1",
        flush=True,
    )
    core.Ours = build_grid_permuted_ours  # type: ignore[assignment]
    core.cli_main()


if __name__ == "__main__":
    main()
