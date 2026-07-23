#!/usr/bin/env python3
"""Run the K/V-only grid-permutation control for Ours V2 on CIFAR-100."""

from __future__ import annotations

import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from methods.OursV2.cifar100.protocol import install_cifar100_defaults
from methods.OursV2.core import cli_main
from methods.OursV2.grid_permutation.model import GridPermutedOurs


PROTOCOL_NAME = "cifar100_deit_ti_ours_v2_grid_permutation_v1"
PERMUTATION_SEED = 1


def build_grid_permuted_ours(
    *args: object,
    **kwargs: object,
) -> GridPermutedOurs:
    return GridPermutedOurs(
        *args,
        permutation_seed=PERMUTATION_SEED,
        **kwargs,
    )


if __name__ == "__main__":
    install_cifar100_defaults(PROTOCOL_NAME)
    cli_main(
        model_factory=build_grid_permuted_ours,
        variant="grid_permutation_v1",
    )
