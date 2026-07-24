#!/usr/bin/env python3
"""ALG compatibility facade over the official LG runtime."""

from __future__ import annotations

import sys

from methods.LG import runtime as _runtime


# Preserve module-level patching/import behavior used by historical tests and
# runners while keeping the single training base under methods/LG.
sys.modules[__name__] = _runtime


if __name__ == "__main__":
    _runtime.cli_main("ALG")
