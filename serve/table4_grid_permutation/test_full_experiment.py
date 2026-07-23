"""Regression checks for the consolidated Table 4 snapshot."""

from __future__ import annotations

import torch

from methods.Ours.table4_grid_permutation.model import (
    GridPermutedOurs as ModularGridPermutedOurs,
)
from serve.table4_grid_permutation.train_cifar100_full import (
    GridPermutedOurs as ConsolidatedGridPermutedOurs,
)
from serve.table4_grid_permutation.train_cifar100_full import (
    PROTOCOL_DEFAULTS as CONSOLIDATED_PROTOCOL_DEFAULTS,
)

EXPECTED_PROTOCOL_DEFAULTS = (
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


def test_protocol_defaults_are_identical() -> None:
    assert CONSOLIDATED_PROTOCOL_DEFAULTS == EXPECTED_PROTOCOL_DEFAULTS


def test_consolidated_model_matches_completed_modular_model() -> None:
    torch.manual_seed(17)
    modular = ModularGridPermutedOurs(
        student_channels=12,
        teacher_channels=(4, 8, 12),
        num_student_blocks=3,
        num_heads=4,
        grid_resize_mode="larger",
        permutation_seed=1,
    ).eval()
    torch.manual_seed(17)
    consolidated = ConsolidatedGridPermutedOurs(
        student_channels=12,
        teacher_channels=(4, 8, 12),
        num_student_blocks=3,
        num_heads=4,
        grid_resize_mode="larger",
        permutation_seed=1,
    ).eval()

    student = [torch.randn(1, 12, 4, 4) for _ in range(3)]
    teacher = [
        torch.randn(1, 4, 8, 8),
        torch.randn(1, 8, 4, 4),
        torch.randn(1, 12, 2, 2),
    ]

    with torch.no_grad():
        modular_outputs = modular(student, teacher)
        consolidated_outputs = consolidated(student, teacher)

    assert torch.equal(modular_outputs[0], consolidated_outputs[0])
    assert torch.equal(modular_outputs[1], consolidated_outputs[1])
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
            assert torch.equal(modular_tensor, consolidated_tensor)

    for stage in range(3):
        name = f"table4_grid_permutation_stage_{stage}"
        assert torch.equal(
            getattr(modular, name),
            getattr(consolidated, name),
        )
