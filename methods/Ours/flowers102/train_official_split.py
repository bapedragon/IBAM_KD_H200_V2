#!/usr/bin/env python3
"""Run Ours on the strict official Flowers-102 train/val/test split."""

from __future__ import annotations

import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from methods.Ours.core import cli_main
from methods.Ours.flowers102.train import PROTOCOL_DEFAULTS as BASE_DEFAULTS


PROTOCOL_DEFAULTS = tuple(
    (option, value)
    for option, value in BASE_DEFAULTS
    if option != "--protocol-name"
) + (
    (
        "--protocol-name",
        "flowers102_deit_ti_ours_researcher_sync_v2_official_split",
    ),
    ("--flowers-split-policy", "official_three_way"),
)


def has_option(option: str) -> bool:
    return any(
        argument == option or argument.startswith(f"{option}=")
        for argument in sys.argv[1:]
    )


if __name__ == "__main__":
    if has_option("--dataset"):
        raise SystemExit("This wrapper fixes --dataset flowers102; remove --dataset.")
    sys.argv[1:1] = ["--dataset", "flowers102"]
    for option, value in reversed(PROTOCOL_DEFAULTS):
        if not has_option(option):
            sys.argv[1:1] = [option, value]
    cli_main()
