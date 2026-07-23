#!/usr/bin/env python3
"""Run the position-agnostic token-space control for Ours V2 on CIFAR-100."""

from __future__ import annotations

import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from methods.OursV2.cifar100.protocol import install_cifar100_defaults
from methods.OursV2.core import cli_main
from methods.OursV2.token_space.model import TokenSpaceOurs


PROTOCOL_NAME = "cifar100_deit_ti_ours_v2_token_space_v1"


if __name__ == "__main__":
    install_cifar100_defaults(PROTOCOL_NAME)
    cli_main(
        model_factory=TokenSpaceOurs,
        variant="token_space_v1",
    )
