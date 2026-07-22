#!/usr/bin/env python3
"""Run one Table 7 CIFAR-100 lambda control with the locked Ours protocol."""

from __future__ import annotations

import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


ALLOWED_LAMBDAS = {
    "0": (0.0, "0"),
    "0.0": (0.0, "0"),
    "0.00": (0.0, "0"),
    "0.25": (0.25, "0p25"),
    ".25": (0.25, "0p25"),
    "0.5": (0.5, "0p5"),
    ".5": (0.5, "0p5"),
    "0.50": (0.5, "0p5"),
    "0.75": (0.75, "0p75"),
    ".75": (0.75, "0p75"),
    "1": (1.0, "1"),
    "1.0": (1.0, "1"),
    "1.00": (1.0, "1"),
}

LOCKED_PROTOCOL_DEFAULTS = (
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


def has_option(arguments: list[str], option: str) -> bool:
    return any(
        argument == option or argument.startswith(f"{option}=")
        for argument in arguments
    )


def extract_lambda_value(arguments: list[str]) -> tuple[float, str, list[str]]:
    """Extract and validate the Table 7 control value from command arguments."""

    values: list[str] = []
    cleaned: list[str] = []
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if argument == "--lambda-value":
            if index + 1 >= len(arguments):
                raise ValueError("--lambda-value requires a value.")
            values.append(arguments[index + 1])
            index += 2
            continue
        if argument.startswith("--lambda-value="):
            values.append(argument.split("=", 1)[1])
            index += 1
            continue
        cleaned.append(argument)
        index += 1

    if len(values) != 1:
        raise ValueError("Supply --lambda-value exactly once.")
    if values[0] not in ALLOWED_LAMBDAS:
        raise ValueError(
            "Table 7 convex sweep permits only lambda in "
            "{0, 0.25, 0.5, 0.75, 1.0}."
        )
    value, name = ALLOWED_LAMBDAS[values[0]]
    return value, name, cleaned


def inject_locked_defaults(
    arguments: list[str], lambda_value: float, lambda_name: str
) -> list[str]:
    """Inject the normal 82.90%-run protocol and the sole lambda change."""

    if has_option(arguments, "--dataset"):
        raise ValueError("This wrapper fixes --dataset cifar100; remove --dataset.")
    if has_option(arguments, "--fusion-ratio"):
        raise ValueError(
            "Do not pass --fusion-ratio directly; use the audited --lambda-value."
        )
    if has_option(arguments, "--protocol-name"):
        raise ValueError("This wrapper fixes --protocol-name for Table 7.")

    injected = list(arguments)
    dynamic_defaults = (
        (
            "--protocol-name",
            f"table7_loss_balance_lambda_{lambda_name}_cifar100_researcher_sync_v1",
        ),
        ("--fusion-ratio", str(lambda_value)),
        ("--dataset", "cifar100"),
        *LOCKED_PROTOCOL_DEFAULTS,
    )
    for option, value in reversed(dynamic_defaults):
        if not has_option(injected, option):
            injected[0:0] = [option, value]
    return injected


def main() -> None:
    try:
        lambda_value, lambda_name, remaining = extract_lambda_value(sys.argv[1:])
        sys.argv[1:] = inject_locked_defaults(
            remaining,
            lambda_value,
            lambda_name,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error

    align_weight = 1.0 - lambda_value
    print(
        "[TABLE7_CONTROL] reference_lambda=0.5 reference_full_top1=82.90% "
        f"only_change=lambda lambda={lambda_value:g}",
        flush=True,
    )
    print(
        "[TABLE7_LOSS] "
        f"feature_loss={lambda_value:g}*L_fuse+{align_weight:g}*L_align "
        "adaptive_beta=unchanged controller_observes=weighted_feature_loss",
        flush=True,
    )

    # Delayed import keeps argument-validation tests independent of torch/timm.
    from methods.Ours.core import cli_main

    cli_main()


if __name__ == "__main__":
    main()
