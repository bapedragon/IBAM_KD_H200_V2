#!/usr/bin/env python3
"""Full review snapshot of the Ours V1 CNN-to-transformer collapse experiment.

The model-side execution path is fully expanded here:

1. learn one stage-specific mixture of all 12 DeiT block features;
2. keep each aggregated student feature at D=192 and the 14x14 patch grid;
3. resize every teacher CNN stage to that one transformer grid;
4. reverse the original projection direction with Conv1x1 C_teacher->192;
5. flatten both representations from BCHW to BND (196 tokens, dimension 192);
6. run Ours V1 content-only cross-attention without permutation or 2D bias;
7. calculate both alignment and fusion MSE in the transformer representation.

Only generic data loading, frozen teacher/DeiT construction, optimization,
evaluation, and checkpoint I/O are reused from ``methods.Ours.core``. No model
or loss implementation is imported from the modular experiment.
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


class TransformerAggregationPooling(nn.Module):
    """Learn one convex mixture of all student blocks for each teacher stage."""

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


PROTOCOL_NAME = (
    "cifar100_deit_ti_ours_v1_cnn_to_transformer_spatial_collapse_v1"
)
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
VARIANT_METADATA = {
    "package": "serve.ours_v1_cnn_to_transformer_spatial_collapse",
    "canonical_package": (
        "methods.Ours.cnn_to_transformer_spatial_collapse"
    ),
    "variant": "cnn_to_transformer_spatial_collapse_v1",
    "base_architecture": "OursV1",
    "original_ours_v1_untouched": True,
    "projection_direction": "teacher_channels_16_32_64_to_student_dimension_192",
    "teacher_geometry": "each_stage_bilinear_to_student_patch_grid_14x14",
    "representation": "flattened_BND_196x192",
    "permutation": "none",
    "explicit_2d_position": "none",
    "loss_space": "transformer_token_representation",
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


def install_variant(core: Any) -> None:
    original_checkpoint_payload = core.checkpoint_payload
    original_finalize_args = core.finalize_args
    original_log = core.log
    original_write_summary = core.write_summary

    def checkpoint_payload(*args: Any, **kwargs: Any) -> dict[str, Any]:
        payload = original_checkpoint_payload(*args, **kwargs)
        payload["ours_variant"] = dict(VARIANT_METADATA)
        return payload

    def finalize_args(args: Any) -> None:
        automatic_run_name = args.run_name is None
        original_finalize_args(args)
        if automatic_run_name:
            args.run_name = args.run_name.replace(
                "ours_",
                "ours_v1_cnn_to_transformer_spatial_collapse_",
                1,
            )

    def write_summary(*args: Any, **kwargs: Any) -> None:
        original_write_summary(*args, **kwargs)
        path_value = args[0] if args else kwargs["path"]
        path = Path(path_value)
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["ours_variant"] = dict(VARIANT_METADATA)
        temporary = path.with_suffix(path.suffix + ".variant.tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary.replace(path)

    def variant_log(message: str = "") -> None:
        if message.startswith("[OURS] student_blocks=all_12"):
            original_log(
                "[OURS_VARIANT] student_blocks=all_12 "
                "aggregation=learnable_uniform_init teacher_stages=1/2/3 "
                "teacher_resize=all_to_student_14x14 "
                "projection=Conv1x1_16/32/64_to_192 "
                "flatten=BND_196x192 attention=content_only "
                "permutation=none explicit_2d_position=none"
            )
            return
        if message.startswith("[REPRO_STATUS] Paper-confirmed:"):
            original_log(
                "[REPRO_STATUS] Base Ours V1 optimizer, controller, losses, "
                "aggregation, deformable student enhancement, and frozen "
                "teacher are retained. Variant change: reverse teacher CNN "
                "features into the common transformer representation before "
                "fusion and feature losses."
            )
            return
        original_log(message)

    core.Ours = CNNToTransformerSpatialCollapseOurs
    core.checkpoint_payload = checkpoint_payload
    core.finalize_args = finalize_args
    core.log = variant_log
    core.write_summary = write_summary


def main() -> None:
    from methods.Ours import core

    install_protocol_defaults()
    install_variant(core)
    print(
        "[OURS_V1_CNN_TO_TRANSFORMER_SPATIAL_COLLAPSE_FULL] "
        f"protocol={PROTOCOL_NAME} metadata={VARIANT_METADATA}",
        flush=True,
    )
    core.cli_main()


if __name__ == "__main__":
    main()
