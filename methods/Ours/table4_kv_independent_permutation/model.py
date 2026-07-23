"""Ours control with different fixed global permutations for key and value.

This experiment is intentionally derived from the completed Table 4 global
grid-permutation control. It preserves the key-side permutation and both MSE
targets from that run, but applies a second, different permutation to the
teacher tensor used as attention value.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F

from methods.Ours.ours import CBAMConvCrossAttention, Ours


def log(message: str) -> None:
    print(message, flush=True)


def independent_kv_attention(
    fusion: CBAMConvCrossAttention,
    student_feature: torch.Tensor,
    key_feature: torch.Tensor,
    value_feature: torch.Tensor,
) -> torch.Tensor:
    """Run an existing fusion block while supplying separate K/V tensors.

    Reusing the block created by :class:`Ours` preserves the exact parameter
    initialization and random-number consumption of the reference run.
    """

    if key_feature.shape != student_feature.shape:
        raise ValueError(
            "Key and aligned student features must have the same shape: "
            f"key={tuple(key_feature.shape)} "
            f"student={tuple(student_feature.shape)}"
        )
    if value_feature.shape != key_feature.shape:
        raise ValueError(
            "Key and value features must have the same shape: "
            f"key={tuple(key_feature.shape)} "
            f"value={tuple(value_feature.shape)}"
        )

    batch, channels, height, width = student_feature.shape
    enhanced_student = fusion.cbam(student_feature)
    query = fusion.q_conv(enhanced_student).flatten(2).transpose(1, 2)
    key = fusion.k_conv(key_feature).flatten(2).transpose(1, 2)
    value = fusion.v_conv(value_feature).flatten(2).transpose(1, 2)

    def split_heads(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.reshape(
            batch,
            height * width,
            fusion.num_heads,
            fusion.head_dim,
        ).permute(0, 2, 1, 3)

    query = split_heads(query)
    key = split_heads(key)
    value = split_heads(value)
    attention = torch.softmax(
        (query @ key.transpose(-2, -1)) * fusion.scale,
        dim=-1,
    )
    attention = fusion.dropout(attention)
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
    return fusion.out_conv(outputs)


class IndependentKVGridPermutedOurs(Ours):
    """Apply deterministic but different global spatial permutations to K/V."""

    def __init__(
        self,
        *args: object,
        permutation_seed: int = 1,
        value_seed_offset: int = 1000,
        **kwargs: object,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.permutation_seed = int(permutation_seed)
        self.value_seed_offset = int(value_seed_offset)
        if self.value_seed_offset == 0:
            raise ValueError("value_seed_offset must be non-zero")

        self.register_buffer(
            "table4_k_permutation_seed",
            torch.tensor(self.permutation_seed, dtype=torch.int64),
            persistent=True,
        )
        self.register_buffer(
            "table4_v_permutation_seed",
            torch.tensor(
                self.permutation_seed + self.value_seed_offset,
                dtype=torch.int64,
            ),
            persistent=True,
        )
        self._permutation_shapes: dict[tuple[str, int], tuple[int, int]] = {}
        log(
            "[TABLE4_KV_INDEPENDENT] enabled=True "
            f"k_seed={self.permutation_seed} "
            f"v_seed={self.permutation_seed + self.value_seed_offset} "
            "fixed_across_samples_and_epochs=True "
            "application=after_grid_resize "
            "targets=K,L_align,L_fuse value_only=V"
        )

    def _fixed_permutation(
        self,
        role: str,
        stage: int,
        height: int,
        width: int,
        device: torch.device,
    ) -> torch.Tensor:
        if role not in {"k", "v"}:
            raise ValueError(f"Unknown permutation role: {role}")
        shape_key = (role, stage)
        expected_shape = (height, width)
        buffer_name = f"table4_{role}_grid_permutation_stage_{stage}"
        if hasattr(self, buffer_name):
            if self._permutation_shapes[shape_key] != expected_shape:
                raise RuntimeError(
                    "A fixed Table 4 permutation cannot change shape within a run: "
                    f"role={role} stage={stage} "
                    f"first={self._permutation_shapes[shape_key]} "
                    f"current={expected_shape}"
                )
            return getattr(self, buffer_name).to(device=device)

        role_offset = 0 if role == "k" else self.value_seed_offset
        stage_seed = self.permutation_seed + role_offset + stage
        generator = torch.Generator(device="cpu")
        generator.manual_seed(stage_seed)
        permutation = torch.randperm(
            height * width,
            generator=generator,
            device="cpu",
        ).to(device=device)
        self.register_buffer(buffer_name, permutation, persistent=True)
        self._permutation_shapes[shape_key] = expected_shape
        log(
            "[TABLE4_KV_INDEPENDENT] "
            f"role={role.upper()} stage={stage + 1} "
            f"grid={height}x{width} positions={height * width} "
            f"stage_seed={stage_seed}"
        )
        return permutation

    def _resize_and_permute_teacher(
        self,
        student_features: Sequence[torch.Tensor],
        teacher_features: Sequence[torch.Tensor],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        if not student_features:
            raise ValueError("At least one student feature is required")
        if len(teacher_features) != len(self.teacher_channels):
            raise ValueError(
                f"Expected {len(self.teacher_channels)} teacher stages, "
                f"got {len(teacher_features)}"
            )

        student_height, student_width = student_features[0].shape[-2:]
        key_features: list[torch.Tensor] = []
        value_features: list[torch.Tensor] = []

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
            k_permutation = self._fixed_permutation(
                "k",
                stage,
                height,
                width,
                teacher_feature.device,
            )
            v_permutation = self._fixed_permutation(
                "v",
                stage,
                height,
                width,
                teacher_feature.device,
            )
            if torch.equal(k_permutation, v_permutation):
                raise RuntimeError(
                    f"K/V permutations unexpectedly match at stage {stage + 1}"
                )

            batch, channels = teacher_feature.shape[:2]
            flattened = teacher_feature.flatten(2)
            key_feature = flattened.index_select(2, k_permutation).reshape(
                batch,
                channels,
                height,
                width,
            )
            value_feature = flattened.index_select(2, v_permutation).reshape(
                batch,
                channels,
                height,
                width,
            )
            key_features.append(key_feature)
            value_features.append(value_feature)

        return key_features, value_features

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
        key_features, value_features = self._resize_and_permute_teacher(
            student_features,
            teacher_features,
        )
        aggregated = self.aggregation(student_features)
        aligned_features: list[torch.Tensor] = []
        fused_features: list[torch.Tensor] = []
        target_features: list[torch.Tensor] = []
        alignment_loss = aggregated.new_zeros(())
        fusion_loss = aggregated.new_zeros(())

        for stage, (
            key_feature,
            value_feature,
            projection,
            fusion,
        ) in enumerate(
            zip(
                key_features,
                value_features,
                self.projections,
                self.fusion_blocks,
                strict=True,
            )
        ):
            aligned = projection(aggregated[:, stage])
            if aligned.shape[-2:] != key_feature.shape[-2:]:
                aligned = F.interpolate(
                    aligned,
                    size=key_feature.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            if aligned.shape[1] != key_feature.shape[1]:
                raise RuntimeError(
                    f"Stage {stage} channel mismatch after projection: "
                    f"aligned={aligned.shape[1]} key={key_feature.shape[1]}"
                )

            fused = independent_kv_attention(
                fusion,
                aligned,
                key_feature,
                value_feature,
            )
            # Preserve the completed control's loss targets. Only V changes.
            alignment_loss = alignment_loss + F.mse_loss(
                aligned.float(),
                key_feature.float(),
            )
            fusion_loss = fusion_loss + F.mse_loss(
                fused.float(),
                key_feature.float(),
            )
            aligned_features.append(aligned)
            fused_features.append(fused)
            target_features.append(key_feature)

        return (
            alignment_loss,
            fusion_loss,
            aligned_features,
            fused_features,
            target_features,
        )
