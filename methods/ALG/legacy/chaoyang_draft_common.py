#!/usr/bin/env python3
"""Archived noncanonical ALG/Ours-common diagnostic; not a paper ALG entry."""

from __future__ import annotations

import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from methods.ALG.core import cli_main


PROTOCOL_DEFAULTS = (
    ("--protocol-name", "chaoyang_deit_ti_alg_draft_common_v1"),
    ("--student-epochs", "300"),
    ("--batch-size", "128"),
    ("--eval-batch-size", "128"),
    ("--lr", "0.0005"),
    ("--min-lr", "0.0"),
    ("--weight-decay", "0.05"),
    ("--warmup-epochs", "20"),
    ("--warmup-factor", "0.001"),
    ("--label-smoothing", "0.1"),
    ("--drop-path-rate", "0.0"),
    ("--teacher-image-size", "32"),
    ("--beta", "2.5"),
    ("--alg-threshold", "-0.02"),
    ("--alg-smoothing-window", "50"),
    ("--base-protocol", "draft_common"),
    ("--eval-resize-mode", "center_crop"),
    ("--seed", "42"),
)
PROTOCOL_FLAGS = ("--amp",)


def has_option(option: str) -> bool:
    positive_or_negative = (option, f"--no-{option.removeprefix('--')}")
    return any(
        argument in positive_or_negative or argument.startswith(f"{option}=")
        for argument in sys.argv[1:]
    )


if __name__ == "__main__":
    if has_option("--dataset"):
        raise SystemExit("This wrapper fixes --dataset chaoyang; remove --dataset.")
    sys.argv[1:1] = ["--dataset", "chaoyang"]
    for option, value in reversed(PROTOCOL_DEFAULTS):
        if not has_option(option):
            sys.argv[1:1] = [option, value]
    for option in reversed(PROTOCOL_FLAGS):
        if not has_option(option):
            sys.argv[1:1] = [option]
    cli_main()
