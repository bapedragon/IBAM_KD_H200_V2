#!/usr/bin/env python3
"""Run pure ALG-paper/public-LG settings on Chaoyang."""

from __future__ import annotations

import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from methods.ALG.core import cli_main


PROTOCOL_DEFAULTS = (
    ("--protocol-name", "chaoyang_deit_ti_alg_paper_lg_v2"),
    ("--student-epochs", "300"),
    ("--batch-size", "128"),
    ("--eval-batch-size", "200"),
    ("--lr", "0.0005"),
    ("--min-lr", "0.000005"),
    ("--weight-decay", "0.05"),
    ("--warmup-epochs", "20"),
    ("--warmup-factor", "0.001"),
    ("--label-smoothing", "0.0"),
    ("--drop-path-rate", "0.1"),
    ("--teacher-image-size", "32"),
    ("--beta", "2.5"),
    ("--alg-threshold", "-0.02"),
    ("--alg-smoothing-window", "50"),
    ("--alg-warmup-epochs", "0"),
    ("--alg-stop-comparison", "paper_ge"),
    ("--alg-derivative-mode", "paper_equations"),
    ("--base-protocol", "lg_official"),
    ("--eval-resize-mode", "direct"),
    ("--seed", "1"),
)


def has_option(option: str) -> bool:
    return any(
        argument == option or argument.startswith(f"{option}=")
        for argument in sys.argv[1:]
    )


if __name__ == "__main__":
    if has_option("--dataset"):
        raise SystemExit("This wrapper fixes --dataset chaoyang; remove --dataset.")
    sys.argv[1:1] = ["--dataset", "chaoyang"]
    for option, value in reversed(PROTOCOL_DEFAULTS):
        if not has_option(option):
            sys.argv[1:1] = [option, value]
    cli_main()
