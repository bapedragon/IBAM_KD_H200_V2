#!/usr/bin/env python3
"""Run official LG ResNet56-to-DeiT-Ti on CUB-200-2011."""

from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from methods.LG.entrypoint import run_dataset


if __name__ == "__main__":
    run_dataset("cub200", "cub200_deit_ti_official_lg_v1")
