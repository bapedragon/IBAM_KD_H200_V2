"""Unit tests for the fixed local-patch permutation control."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

import torch

from methods.Ours.table4_grid_permutation.model import GridPermutedOurs
from methods.Ours.table4_local_patch_permutation.model import (
    LocalPatchPermutedOurs,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def protocol_defaults(path: Path) -> tuple[tuple[str, str], ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "PROTOCOL_DEFAULTS"
            for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"PROTOCOL_DEFAULTS not found in {path}")


def features() -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    student = [torch.randn(2, 192, 14, 14) for _ in range(12)]
    teacher = [
        torch.arange(2 * 16 * 32 * 32, dtype=torch.float32).reshape(
            2,
            16,
            32,
            32,
        ),
        torch.arange(2 * 32 * 16 * 16, dtype=torch.float32).reshape(
            2,
            32,
            16,
            16,
        ),
        torch.arange(2 * 64 * 8 * 8, dtype=torch.float32).reshape(
            2,
            64,
            8,
            8,
        ),
    ]
    return student, teacher


class LocalPatchPermutationTest(unittest.TestCase):
    def test_protocol_differs_only_by_experiment_name(self) -> None:
        global_control = dict(
            protocol_defaults(
                REPOSITORY_ROOT
                / "methods/Ours/table4_grid_permutation/train_cifar100.py"
            )
        )
        local_control = dict(
            protocol_defaults(
                REPOSITORY_ROOT
                / "methods/Ours/table4_local_patch_permutation/train_cifar100.py"
            )
        )
        self.assertNotEqual(
            global_control.pop("--protocol-name"),
            local_control.pop("--protocol-name"),
        )
        self.assertEqual(global_control, local_control)

    def test_parameter_initialization_matches_global_control(self) -> None:
        torch.manual_seed(123)
        global_control = GridPermutedOurs(permutation_seed=1)
        torch.manual_seed(123)
        local_control = LocalPatchPermutedOurs(
            permutation_seed=1,
            local_patch_size=2,
        )
        global_parameters = dict(global_control.named_parameters())
        local_parameters = dict(local_control.named_parameters())
        self.assertEqual(global_parameters.keys(), local_parameters.keys())
        for name, global_value in global_parameters.items():
            self.assertTrue(
                torch.equal(global_value, local_parameters[name]),
                msg=name,
            )

    def test_permutation_stays_inside_each_local_window(self) -> None:
        module = LocalPatchPermutedOurs(
            permutation_seed=1,
            local_patch_size=2,
        )
        for stage, (height, width) in enumerate(((32, 32), (16, 16), (14, 14))):
            permutation = module._fixed_local_permutation(
                stage,
                height,
                width,
                torch.device("cpu"),
            )
            destination = torch.arange(height * width)
            source_rows = torch.div(permutation, width, rounding_mode="floor")
            source_cols = permutation.remainder(width)
            destination_rows = torch.div(
                destination,
                width,
                rounding_mode="floor",
            )
            destination_cols = destination.remainder(width)
            self.assertTrue(
                torch.equal(source_rows // 2, destination_rows // 2)
            )
            self.assertTrue(
                torch.equal(source_cols // 2, destination_cols // 2)
            )
            self.assertFalse(torch.equal(permutation, destination))
            self.assertTrue(torch.all(permutation != destination))
            self.assertTrue(
                torch.equal(
                    permutation.sort().values,
                    destination,
                )
            )

    def test_permutation_is_fixed_and_parent_uses_same_target(self) -> None:
        student, teacher = features()
        module = LocalPatchPermutedOurs(
            permutation_seed=1,
            local_patch_size=2,
        )
        first = module._resize_and_permute_teacher(student, teacher)
        second = module._resize_and_permute_teacher(student, teacher)
        _, _, _, _, actual_targets = module(student, teacher)

        expected_shapes = [
            (2, 16, 32, 32),
            (2, 32, 16, 16),
            (2, 64, 14, 14),
        ]
        self.assertEqual([tuple(value.shape) for value in first], expected_shapes)
        for first_value, second_value, target in zip(
            first,
            second,
            actual_targets,
            strict=True,
        ):
            self.assertTrue(torch.equal(first_value, second_value))
            self.assertTrue(torch.equal(first_value, target))

    def test_invalid_patch_size_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least 2"):
            LocalPatchPermutedOurs(local_patch_size=1)


if __name__ == "__main__":
    unittest.main()
