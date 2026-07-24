"""Dataset-locked entry-point helper for official LG."""

from __future__ import annotations

import sys

from methods.LG.core import cli_main


OFFICIAL_LG_DEFAULTS = (
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
    ("--base-protocol", "lg_official"),
    ("--eval-resize-mode", "direct"),
    ("--seed", "1"),
)


def has_option(option: str) -> bool:
    return any(
        argument == option or argument.startswith(f"{option}=")
        for argument in sys.argv[1:]
    )


def run_dataset(dataset: str, protocol_name: str) -> None:
    if has_option("--dataset"):
        raise SystemExit(f"This wrapper fixes --dataset {dataset}; remove --dataset.")
    sys.argv[1:1] = ["--dataset", dataset]
    defaults = (
        ("--protocol-name", protocol_name),
        *OFFICIAL_LG_DEFAULTS,
    )
    if dataset == "flowers102":
        defaults = (
            *defaults,
            ("--flowers-split-policy", "trainval_test_best"),
        )
    for option, value in reversed(defaults):
        if not has_option(option):
            sys.argv[1:1] = [option, value]
    cli_main()
