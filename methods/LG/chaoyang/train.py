#!/usr/bin/env python3
"""Run official LG ResNet56-to-DeiT-Ti on Chaoyang."""

from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from methods.LG.entrypoint import run_dataset


if __name__ == "__main__":
    run_dataset("chaoyang", "chaoyang_deit_ti_official_lg_v1")
