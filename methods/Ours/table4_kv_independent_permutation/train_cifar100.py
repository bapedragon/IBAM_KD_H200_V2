#!/usr/bin/env python3
"""Run the Table 4 independent-K/V global permutation control."""

from __future__ import annotations

import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from methods.Ours import core
from methods.Ours.table4_kv_independent_permutation.model import (
    IndependentKVGridPermutedOurs,
)


PERMUTATION_SEED = 1
VALUE_SEED_OFFSET = 1000
PROTOCOL_DEFAULTS = (
    ("--protocol-name", "table4_kv_independent_cifar100_researcher_sync_v1"),
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


def has_option(option: str) -> bool:
    return any(
        argument == option or argument.startswith(f"{option}=")
        for argument in sys.argv[1:]
    )


def build_kv_independent_ours(
    *args: object,
    **kwargs: object,
) -> IndependentKVGridPermutedOurs:
    return IndependentKVGridPermutedOurs(
        *args,
        permutation_seed=PERMUTATION_SEED,
        value_seed_offset=VALUE_SEED_OFFSET,
        **kwargs,
    )


if __name__ == "__main__":
    if has_option("--dataset"):
        raise SystemExit("This wrapper fixes --dataset cifar100; remove --dataset.")
    sys.argv[1:1] = ["--dataset", "cifar100"]
    for option, value in reversed(PROTOCOL_DEFAULTS):
        if not has_option(option):
            sys.argv[1:1] = [option, value]

    print(
        "[TABLE4_CONTROL] reference_full_top1=82.90% "
        "reference_same_kv_permutation_top1=81.79% "
        "only_change=independent_value_permutation "
        "k_seed=1 v_seed=1001",
        flush=True,
    )
    core.Ours = build_kv_independent_ours  # type: ignore[assignment]
    core.cli_main()
