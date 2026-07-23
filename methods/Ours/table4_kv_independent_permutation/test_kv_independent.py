"""Unit tests for the independent-K/V global permutation control."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

import torch

from methods.Ours.table4_grid_permutation.model import GridPermutedOurs
from methods.Ours.table4_kv_independent_permutation.model import (
    IndependentKVGridPermutedOurs,
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


class IndependentKVPermutationTest(unittest.TestCase):
    def test_parameter_initialization_matches_previous_control(self) -> None:
        torch.manual_seed(123)
        previous = GridPermutedOurs(permutation_seed=1)
        torch.manual_seed(123)
        independent = IndependentKVGridPermutedOurs(
            permutation_seed=1,
            value_seed_offset=1000,
        )

        previous_parameters = dict(previous.named_parameters())
        independent_parameters = dict(independent.named_parameters())
        self.assertEqual(previous_parameters.keys(), independent_parameters.keys())
        for name, previous_value in previous_parameters.items():
            self.assertTrue(
                torch.equal(previous_value, independent_parameters[name]),
                msg=name,
            )

    def test_protocol_differs_only_by_experiment_name(self) -> None:
        previous = dict(
            protocol_defaults(
                REPOSITORY_ROOT
                / "methods/Ours/table4_grid_permutation/train_cifar100.py"
            )
        )
        independent = dict(
            protocol_defaults(
                REPOSITORY_ROOT
                / "methods/Ours/table4_kv_independent_permutation/train_cifar100.py"
            )
        )
        self.assertNotEqual(
            previous.pop("--protocol-name"),
            independent.pop("--protocol-name"),
        )
        self.assertEqual(previous, independent)

    def test_k_and_v_use_fixed_different_permutations(self) -> None:
        student, teacher = features()
        module = IndependentKVGridPermutedOurs(
            permutation_seed=1,
            value_seed_offset=1000,
        )
        first_k, first_v = module._resize_and_permute_teacher(student, teacher)
        second_k, second_v = module._resize_and_permute_teacher(student, teacher)

        expected_shapes = [
            (2, 16, 32, 32),
            (2, 32, 16, 16),
            (2, 64, 14, 14),
        ]
        self.assertEqual([tuple(value.shape) for value in first_k], expected_shapes)
        self.assertEqual([tuple(value.shape) for value in first_v], expected_shapes)

        for key, value, key_again, value_again in zip(
            first_k,
            first_v,
            second_k,
            second_v,
            strict=True,
        ):
            self.assertFalse(torch.equal(key, value))
            self.assertTrue(torch.equal(key, key_again))
            self.assertTrue(torch.equal(value, value_again))
            self.assertTrue(
                torch.equal(
                    key.flatten(2).sort(dim=2).values,
                    value.flatten(2).sort(dim=2).values,
                )
            )

        state = module.state_dict()
        self.assertIn("table4_k_grid_permutation_stage_0", state)
        self.assertIn("table4_v_grid_permutation_stage_0", state)
        self.assertFalse(
            torch.equal(
                state["table4_k_grid_permutation_stage_0"],
                state["table4_v_grid_permutation_stage_0"],
            )
        )

    def test_key_permutation_remains_loss_target(self) -> None:
        student, teacher = features()
        module = IndependentKVGridPermutedOurs(
            permutation_seed=1,
            value_seed_offset=1000,
        )
        expected_key, _ = module._resize_and_permute_teacher(student, teacher)
        _, _, _, _, actual_targets = module(student, teacher)

        for expected, actual in zip(expected_key, actual_targets, strict=True):
            self.assertTrue(torch.equal(expected, actual))

    def test_zero_value_seed_offset_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be non-zero"):
            IndependentKVGridPermutedOurs(value_seed_offset=0)


if __name__ == "__main__":
    unittest.main()
