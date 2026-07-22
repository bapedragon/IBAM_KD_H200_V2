#!/usr/bin/env python3
"""Dependency-free tests for the isolated Table 7 wrapper contract."""

from __future__ import annotations

import unittest

from methods.Ours.table7_loss_balance.train_cifar100 import (
    extract_lambda_value,
    inject_locked_defaults,
)


class Table7WrapperTest(unittest.TestCase):
    def test_lambda_zero_is_alignment_only(self) -> None:
        value, name, remaining = extract_lambda_value(
            ["--lambda-value", "0", "--timing-run"]
        )
        self.assertEqual(value, 0.0)
        self.assertEqual(name, "0")
        self.assertEqual(remaining, ["--timing-run"])

        injected = inject_locked_defaults(remaining, value, name)
        ratio_index = injected.index("--fusion-ratio")
        self.assertEqual(injected[ratio_index + 1], "0.0")
        self.assertIn("table7_loss_balance_lambda_0_cifar100", " ".join(injected))

    def test_all_convex_sweep_values_are_accepted(self) -> None:
        for raw, expected in (
            ("0", 0.0),
            ("0.25", 0.25),
            ("0.5", 0.5),
            ("0.75", 0.75),
            ("1.0", 1.0),
        ):
            with self.subTest(raw=raw):
                value, _, _ = extract_lambda_value([f"--lambda-value={raw}"])
                self.assertEqual(value, expected)

    def test_unplanned_lambda_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "permits only"):
            extract_lambda_value(["--lambda-value", "0.4"])

    def test_direct_fusion_ratio_override_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Do not pass --fusion-ratio"):
            inject_locked_defaults(["--fusion-ratio", "0.25"], 0.0, "0")


if __name__ == "__main__":
    unittest.main()
