#!/usr/bin/env python3
"""Run position-aware Ours V2 on CIFAR-100."""

from __future__ import annotations

import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from methods.OursV2.cifar100.protocol import install_cifar100_defaults
from methods.OursV2.core import cli_main


PROTOCOL_NAME = "cifar100_deit_ti_ours_v2_relative_position_v1"


if __name__ == "__main__":
    install_cifar100_defaults(PROTOCOL_NAME)
    cli_main()
