"""Token-space cross-attention substitution for the Table 4 control.

The full Ours pipeline is preserved. Only the fusion block changes from
1x1-convolutional Q/K/V projections on BCHW grids to fully connected Q/K/V
projections on flattened BNC token sequences, matching the working paper.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from methods.Ours.ours import DeformableCBAM, Ours


def log(message: str) -> None:
    print(message, flush=True)


class TokenSpaceCrossAttention(nn.Module):
    """Standard multi-head cross-attention over flattened spatial tokens."""

    def __init__(
        self,
        embed_dim: int,
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

        # Enhancement is not part of the intervention.
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

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
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
        student_tokens = self.cbam(student_feature).flatten(2).transpose(1, 2)
        teacher_tokens = teacher_feature.flatten(2).transpose(1, 2)
        query = self._split_heads(self.q_linear(student_tokens))
        key = self._split_heads(self.k_linear(teacher_tokens))
        value = self._split_heads(self.v_linear(teacher_tokens))

        attention = torch.softmax(
            (query @ key.transpose(-2, -1)) * self.scale,
            dim=-1,
        )
        attention = self.dropout(attention)
        output_tokens = (attention @ value).transpose(1, 2).reshape(
            batch,
            height * width,
            channels,
        )
        output_tokens = self.out_linear(output_tokens)
        return output_tokens.transpose(1, 2).reshape(
            batch,
            channels,
            height,
            width,
        )


class TokenSpaceOurs(Ours):
    """Ours with only fusion attention replaced by token-space attention."""

    def __init__(
        self,
        *args: object,
        student_channels: int = 192,
        teacher_channels: Sequence[int] = (16, 32, 64),
        num_student_blocks: int = 12,
        num_heads: int = 4,
        spatial_kernel_size: int = 5,
        grid_resize_mode: str = "larger",
        **kwargs: object,
    ) -> None:
        super().__init__(
            *args,
            student_channels=student_channels,
            teacher_channels=teacher_channels,
            num_student_blocks=num_student_blocks,
            num_heads=num_heads,
            spatial_kernel_size=spatial_kernel_size,
            grid_resize_mode=grid_resize_mode,
            **kwargs,
        )
        self.fusion_blocks = nn.ModuleList(
            TokenSpaceCrossAttention(
                int(channels),
                num_heads=num_heads,
                spatial_kernel_size=spatial_kernel_size,
            )
            for channels in teacher_channels
        )
        self.register_buffer(
            "table4_token_space_enabled",
            torch.tensor(True, dtype=torch.bool),
            persistent=True,
        )
        log(
            "[TABLE4_TOKEN_SPACE] enabled=True "
            "only_change=fusion_qkv_projection "
            "grid_qkv=Conv1x1 token_qkv=Linear_on_flattened_BNC "
            "positional_encoding=False"
        )
