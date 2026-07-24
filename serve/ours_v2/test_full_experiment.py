"""Regression checks for the consolidated Ours V2 review snapshot."""

from __future__ import annotations

import unittest

import torch

from methods.OursV2.cifar100.protocol import protocol_defaults
from methods.OursV2.ours import Ours as ModularOursV2
from serve.ours_v2.train_cifar100_full import (
    OURS_V2_METADATA,
    PROTOCOL_DEFAULTS,
    PROTOCOL_NAME,
    Ours as ConsolidatedOursV2,
)


class ConsolidatedOursV2Test(unittest.TestCase):
    def test_protocol_defaults_match_maintained_v2(self) -> None:
        self.assertEqual(
            PROTOCOL_DEFAULTS,
            protocol_defaults(PROTOCOL_NAME),
        )

    def test_consolidated_model_matches_maintained_v2(self) -> None:
        torch.manual_seed(29)
        modular = ModularOursV2(
            student_channels=12,
            teacher_channels=(4, 8, 12),
            num_student_blocks=3,
            num_heads=4,
            grid_resize_mode="larger",
            max_grid_size=8,
        ).eval()
        torch.manual_seed(29)
        consolidated = ConsolidatedOursV2(
            student_channels=12,
            teacher_channels=(4, 8, 12),
            num_student_blocks=3,
            num_heads=4,
            grid_resize_mode="larger",
            max_grid_size=8,
        ).eval()

        modular_state = modular.state_dict()
        consolidated_state = consolidated.state_dict()
        self.assertEqual(modular_state.keys(), consolidated_state.keys())
        for name in modular_state:
            self.assertTrue(
                torch.equal(modular_state[name], consolidated_state[name]),
                name,
            )

        student = [torch.randn(2, 12, 4, 4) for _ in range(3)]
        teacher = [
            torch.randn(2, 4, 8, 8),
            torch.randn(2, 8, 4, 4),
            torch.randn(2, 12, 2, 2),
        ]
        with torch.no_grad():
            modular_outputs = modular(student, teacher)
            consolidated_outputs = consolidated(student, teacher)

        torch.testing.assert_close(
            modular_outputs[0],
            consolidated_outputs[0],
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            modular_outputs[1],
            consolidated_outputs[1],
            rtol=0.0,
            atol=0.0,
        )
        for modular_group, consolidated_group in zip(
            modular_outputs[2:],
            consolidated_outputs[2:],
            strict=True,
        ):
            for modular_tensor, consolidated_tensor in zip(
                modular_group,
                consolidated_group,
                strict=True,
            ):
                torch.testing.assert_close(
                    modular_tensor,
                    consolidated_tensor,
                    rtol=0.0,
                    atol=0.0,
                )

    def test_snapshot_is_labeled_as_base_position_aware_v2(self) -> None:
        self.assertEqual(
            OURS_V2_METADATA["variant"],
            "relative_position_v1",
        )
        self.assertEqual(
            OURS_V2_METADATA["fusion"],
            "conv1x1_global_cross_attention_plus_2d_relative_bias",
        )
        model = ConsolidatedOursV2()
        trainable = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        self.assertEqual(trainable, 103_529)


if __name__ == "__main__":
    unittest.main()
