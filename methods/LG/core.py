#!/usr/bin/env python3
"""Run static official Locality Guidance."""

from __future__ import annotations

from methods.LG.runtime import (  # noqa: F401
    OFFICIAL_LG_COMMIT,
    StaticGuidanceController,
    build_lg_loaders,
    build_lg_loaders_with_final_test,
    create_official_lg_optimizer,
    create_scheduler,
    create_student,
    forward_student_features,
    forward_teacher_features,
    official_lg_parameter_groups,
)
from methods.LG.runtime import cli_main as _runtime_cli_main
from methods.LG.runtime import main as _runtime_main


def main() -> None:
    _runtime_main("LG")


def cli_main() -> None:
    _runtime_cli_main("LG")


if __name__ == "__main__":
    cli_main()
