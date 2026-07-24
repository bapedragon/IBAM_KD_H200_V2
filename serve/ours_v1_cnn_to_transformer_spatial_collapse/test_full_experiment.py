"""Regression checks for the consolidated CNN-to-transformer experiment."""

from __future__ import annotations

import unittest

import torch

from methods.Ours.cnn_to_transformer_spatial_collapse.model import (
    CNNToTransformerSpatialCollapseOurs as ModularModel,
)
from methods.Ours.cnn_to_transformer_spatial_collapse.train_cifar100 import (
    PROTOCOL_DEFAULTS as MODULAR_PROTOCOL_DEFAULTS,
)
from methods.Ours.cifar100.train import (
    PROTOCOL_DEFAULTS as BASE_OURS_V1_PROTOCOL_DEFAULTS,
)
from serve.ours_v1_cnn_to_transformer_spatial_collapse.train_cifar100_full import (
    CNNToTransformerSpatialCollapseOurs as ConsolidatedModel,
)
from serve.ours_v1_cnn_to_transformer_spatial_collapse.train_cifar100_full import (
    PROTOCOL_DEFAULTS,
    VARIANT_METADATA,
)


class CNNToTransformerSpatialCollapseTest(unittest.TestCase):
    def build_inputs(
        self,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        student = [torch.randn(2, 12, 4, 4) for _ in range(3)]
        teacher = [
            torch.randn(2, 4, 8, 8),
            torch.randn(2, 8, 4, 4),
            torch.randn(2, 16, 2, 2),
        ]
        return student, teacher

    def test_projection_direction_and_token_shapes(self) -> None:
        model = ModularModel(
            student_channels=12,
            teacher_channels=(4, 8, 16),
            num_student_blocks=3,
            num_heads=4,
        )
        self.assertEqual(
            [tuple(layer.weight.shape) for layer in model.teacher_to_transformer_projections],
            [(12, 4, 1, 1), (12, 8, 1, 1), (12, 16, 1, 1)],
        )
        student, teacher = self.build_inputs()
        outputs = model(student, teacher)
        for group in outputs[2:]:
            self.assertEqual(
                [tuple(tensor.shape) for tensor in group],
                [(2, 16, 12)] * 3,
            )
        state_keys = tuple(model.state_dict())
        self.assertFalse(any("permutation" in key for key in state_keys))
        self.assertFalse(any("position" in key for key in state_keys))

    def test_consolidated_model_exactly_matches_modular_model(self) -> None:
        torch.manual_seed(41)
        modular = ModularModel(
            student_channels=12,
            teacher_channels=(4, 8, 16),
            num_student_blocks=3,
            num_heads=4,
        ).eval()
        torch.manual_seed(41)
        consolidated = ConsolidatedModel(
            student_channels=12,
            teacher_channels=(4, 8, 16),
            num_student_blocks=3,
            num_heads=4,
        ).eval()
        self.assertEqual(modular.state_dict().keys(), consolidated.state_dict().keys())
        for name, tensor in modular.state_dict().items():
            self.assertTrue(torch.equal(tensor, consolidated.state_dict()[name]), name)

        student, teacher = self.build_inputs()
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

    def test_backward_reaches_reversed_teacher_projections(self) -> None:
        model = ModularModel(
            student_channels=12,
            teacher_channels=(4, 8, 16),
            num_student_blocks=3,
            num_heads=4,
        )
        student, teacher = self.build_inputs()
        alignment, fusion, *_ = model(student, teacher)
        (alignment + fusion).backward()
        for projection in model.teacher_to_transformer_projections:
            self.assertIsNotNone(projection.weight.grad)
            assert projection.weight.grad is not None
            self.assertGreater(float(projection.weight.grad.abs().sum()), 0.0)

    def test_protocol_and_metadata_are_unambiguous(self) -> None:
        self.assertEqual(PROTOCOL_DEFAULTS, MODULAR_PROTOCOL_DEFAULTS)
        defaults = dict(PROTOCOL_DEFAULTS)
        self.assertEqual(defaults["--student-epochs"], "300")
        self.assertEqual(defaults["--batch-size"], "64")
        base_defaults = dict(BASE_OURS_V1_PROTOCOL_DEFAULTS)
        defaults.pop("--protocol-name")
        base_defaults.pop("--protocol-name")
        self.assertEqual(defaults, base_defaults)
        self.assertEqual(
            VARIANT_METADATA["projection_direction"],
            "teacher_channels_16_32_64_to_student_dimension_192",
        )
        self.assertEqual(VARIANT_METADATA["permutation"], "none")
        self.assertEqual(VARIANT_METADATA["explicit_2d_position"], "none")


if __name__ == "__main__":
    unittest.main()
