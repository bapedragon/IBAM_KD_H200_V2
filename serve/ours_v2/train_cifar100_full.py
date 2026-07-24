#!/usr/bin/env python3
"""Consolidated executable snapshot of base Ours V2 on CIFAR-100.

The complete model-side path is expanded in this file for review:

1. learnable aggregation of all student blocks;
2. stage-wise 1x1 channel projection and grid resize;
3. deformable channel/spatial enhancement;
4. convolutional Q/K/V cross-attention;
5. learned head-specific 2D relative-position bias;
6. alignment and fusion MSE losses;
7. the exact locked CIFAR-100 protocol and Ours V2 artifact metadata.

Dataset loading, teacher/DeiT construction, optimizer, scheduler, evaluation,
and atomic checkpoint I/O use the maintained ``methods.Ours.core`` trainer.
No model or loss implementation is imported from ``methods.OursV2``.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


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
        return torch.sigmoid(
            self.spatial_conv(torch.cat((average, maximum), dim=1))
        )


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


class RelativePositionBias2D(nn.Module):
    """Learn one full 2D displacement table for each attention head."""

    def __init__(self, num_heads: int, max_grid_size: int = 32) -> None:
        super().__init__()
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if max_grid_size <= 0:
            raise ValueError("max_grid_size must be positive")
        self.num_heads = int(num_heads)
        self.max_grid_size = int(max_grid_size)
        self.relative_extent = 2 * self.max_grid_size - 1
        self.bias_table = nn.Parameter(
            torch.zeros(
                self.num_heads,
                self.relative_extent * self.relative_extent,
            )
        )

    def position_index(
        self,
        height: int,
        width: int,
        device: torch.device,
    ) -> torch.Tensor:
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
    """Global convolutional cross-attention with an explicit 2D relation."""

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


class TransformerAggregationPooling(nn.Module):
    """Learn one convex mixture of all student blocks for each CNN stage."""

    def __init__(self, num_transformer: int = 12, num_cnn: int = 3) -> None:
        super().__init__()
        self.weights = nn.Parameter(torch.zeros(num_cnn, num_transformer))

    def forward(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(features) != self.weights.shape[1]:
            raise ValueError(
                f"Expected {self.weights.shape[1]} student features, "
                f"got {len(features)}"
            )
        stacked = torch.stack(tuple(features), dim=1)
        normalized_weights = torch.softmax(self.weights, dim=-1)
        return torch.einsum("gl,bldhw->bgdhw", normalized_weights, stacked)

    def normalized_weights(self) -> torch.Tensor:
        return torch.softmax(self.weights.detach(), dim=-1)


class Ours(nn.Module):
    """Complete base Ours V2 model and both feature losses."""

    def __init__(
        self,
        student_channels: int = 192,
        teacher_channels: Sequence[int] = (16, 32, 64),
        num_student_blocks: int = 12,
        num_heads: int = 4,
        spatial_kernel_size: int = 5,
        grid_resize_mode: str = "larger",
        max_grid_size: int = 32,
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
        self.fusion_blocks = nn.ModuleList(
            GridAwareCrossAttention(
                channels,
                num_heads=num_heads,
                spatial_kernel_size=spatial_kernel_size,
                qkv_kernel_size=1,
                max_grid_size=max_grid_size,
            )
            for channels in self.teacher_channels
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


PROTOCOL_NAME = "cifar100_deit_ti_ours_v2_relative_position_v1"
PROTOCOL_DEFAULTS = (
    ("--protocol-name", PROTOCOL_NAME),
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

OURS_V2_METADATA = {
    "package": "serve.ours_v2",
    "canonical_package": "methods.OursV2",
    "variant": "relative_position_v1",
    "base_training_protocol": "methods.Ours.core",
    "original_ours_model_untouched": True,
    "fusion": "conv1x1_global_cross_attention_plus_2d_relative_bias",
    "relative_position_bias": (
        "zero_initialized_learned_full_2d_displacement_table_per_head"
    ),
    "consolidated_review_snapshot": True,
}


def has_option(option: str) -> bool:
    return any(
        argument == option or argument.startswith(f"{option}=")
        for argument in sys.argv[1:]
    )


def install_protocol_defaults() -> None:
    if has_option("--dataset"):
        raise SystemExit("This entry point fixes --dataset cifar100.")
    sys.argv[1:1] = ["--dataset", "cifar100"]
    for option, value in reversed(PROTOCOL_DEFAULTS):
        if not has_option(option):
            sys.argv[1:1] = [option, value]


def install_ours_v2_artifact_metadata(core: Any) -> None:
    """Label shared-trainer artifacts as the consolidated Ours V2 variant."""

    original_checkpoint_payload = core.checkpoint_payload
    original_finalize_args = core.finalize_args
    original_write_summary = core.write_summary

    def checkpoint_payload(*args: Any, **kwargs: Any) -> dict[str, Any]:
        payload = original_checkpoint_payload(*args, **kwargs)
        payload["method"] = "OursV2"
        payload["ours_v2"] = dict(OURS_V2_METADATA)
        payload["base_ours_source_sha256"] = payload.pop(
            "source_snippet_sha256",
            None,
        )
        return payload

    def finalize_args(args: Any) -> None:
        automatic_run_name = args.run_name is None
        original_finalize_args(args)
        if automatic_run_name:
            args.run_name = args.run_name.replace(
                "ours_",
                "ours_v2_relative_position_v1_",
                1,
            )

    def write_summary(*args: Any, **kwargs: Any) -> None:
        original_write_summary(*args, **kwargs)
        path_value = args[0] if args else kwargs["path"]
        path = Path(path_value)
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["method"] = "OursV2"
        payload["ours_v2"] = dict(OURS_V2_METADATA)
        payload["base_ours_source_sha256"] = payload.pop(
            "source_snippet_sha256",
            None,
        )
        temporary = path.with_suffix(path.suffix + ".ours_v2.tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary.replace(path)

    core.checkpoint_payload = checkpoint_payload
    core.finalize_args = finalize_args
    core.write_summary = write_summary


def main() -> None:
    # The maintained trainer is loaded only for an actual executable run. The
    # full model above remains importable without triggering timm installation.
    from methods.Ours import core

    install_protocol_defaults()
    install_ours_v2_artifact_metadata(core)
    core.Ours = Ours
    print(
        "[OURS_V2_CONSOLIDATED] "
        "variant=relative_position_v1 "
        "model_and_losses_expanded_in=serve/ours_v2/train_cifar100_full.py "
        f"metadata={OURS_V2_METADATA}",
        flush=True,
    )
    core.cli_main()


if __name__ == "__main__":
    main()
