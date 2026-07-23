#!/usr/bin/env python3
"""Run Ours ResNet56-to-DeiT-Ti training on CUB-200-2011."""

from __future__ import annotations

import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from methods.Ours.core import cli_main  # noqa: E402


PROTOCOL_DEFAULTS = (
    ("--protocol-name", "cub200_deit_ti_ours_scratch_teacher_v1"),
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


if __name__ == "__main__":
    if has_option("--dataset"):
        raise SystemExit("This wrapper fixes --dataset cub200; remove --dataset.")
    sys.argv[1:1] = ["--dataset", "cub200"]
    for option, value in reversed(PROTOCOL_DEFAULTS):
        if not has_option(option):
            sys.argv[1:1] = [option, value]
    cli_main()
