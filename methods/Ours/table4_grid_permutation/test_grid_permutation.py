"""Unit tests for the isolated Table 4 grid-permutation control."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

import torch

from methods.Ours.table4_grid_permutation.model import GridPermutedOurs


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
        torch.arange(2 * 16 * 32 * 32, dtype=torch.float32).reshape(2, 16, 32, 32),
        torch.arange(2 * 32 * 16 * 16, dtype=torch.float32).reshape(2, 32, 16, 16),
        torch.arange(2 * 64 * 8 * 8, dtype=torch.float32).reshape(2, 64, 8, 8),
    ]
    return student, teacher


class GridPermutationTest(unittest.TestCase):
    def test_protocol_differs_only_by_experiment_name(self) -> None:
        full = dict(protocol_defaults(REPOSITORY_ROOT / "methods/Ours/cifar100/train.py"))
        permuted = dict(
            protocol_defaults(
                REPOSITORY_ROOT
                / "methods/Ours/table4_grid_permutation/train_cifar100.py"
            )
        )
        self.assertNotEqual(
            full.pop("--protocol-name"),
            permuted.pop("--protocol-name"),
        )
        self.assertEqual(full, permuted)

    def test_permutation_is_fixed_and_preserves_each_value_multiset(self) -> None:
        student, teacher = features()
        module = GridPermutedOurs(permutation_seed=1)

        first = module._resize_and_permute_teacher(student, teacher)
        second = module._resize_and_permute_teacher(student, teacher)

        self.assertEqual(int(module.table4_permutation_seed), 1)
        self.assertIn("table4_permutation_seed", module.state_dict())
        self.assertIn("table4_grid_permutation_stage_0", module.state_dict())
        expected_shapes = [
            (2, 16, 32, 32),
            (2, 32, 16, 16),
            (2, 64, 14, 14),
        ]
        self.assertEqual([tuple(value.shape) for value in first], expected_shapes)
        for first_value, second_value in zip(first, second, strict=True):
            self.assertTrue(torch.equal(first_value, second_value))

        resized_stage0 = teacher[0]
        self.assertFalse(torch.equal(first[0], resized_stage0))
        self.assertTrue(
            torch.equal(
                first[0].flatten(2).sort(dim=2).values,
                resized_stage0.flatten(2).sort(dim=2).values,
            )
        )

    def test_parent_uses_permuted_teacher_for_all_targets(self) -> None:
        student, teacher = features()
        module = GridPermutedOurs(permutation_seed=1)
        expected_targets = module._resize_and_permute_teacher(student, teacher)

        _, _, _, _, actual_targets = module(student, teacher)

        for expected, actual in zip(expected_targets, actual_targets, strict=True):
            self.assertTrue(torch.equal(expected, actual))

    def test_different_seed_changes_spatial_order_only(self) -> None:
        student, teacher = features()
        first = GridPermutedOurs(permutation_seed=1)._resize_and_permute_teacher(
            student,
            teacher,
        )
        second = GridPermutedOurs(permutation_seed=2)._resize_and_permute_teacher(
            student,
            teacher,
        )

        self.assertFalse(torch.equal(first[0], second[0]))
        self.assertTrue(
            torch.equal(
                first[0].flatten(2).sort(dim=2).values,
                second[0].flatten(2).sort(dim=2).values,
            )
        )


if __name__ == "__main__":
    unittest.main()
