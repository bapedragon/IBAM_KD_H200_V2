from __future__ import annotations

import unittest
from pathlib import Path

from methods.OursV2.cifar100.protocol import BASE_PROTOCOL_DEFAULTS
from methods.OursV2.table7_loss_balance.train_cifar100 import (
    extract_lambda_value,
    inject_locked_defaults,
)


def option_value(arguments: list[str], option: str) -> str:
    index = arguments.index(option)
    return arguments[index + 1]


class OursV2Table7WrapperTest(unittest.TestCase):
    def test_lambda_zero_is_alignment_only_on_v2_protocol(self) -> None:
        value, name, remaining = extract_lambda_value(
            ["--lambda-value", "0", "--num-workers", "4"]
        )
        self.assertEqual(value, 0.0)
        self.assertEqual(name, "0")
        self.assertEqual(remaining, ["--num-workers", "4"])

        injected = inject_locked_defaults(remaining, value, name)
        self.assertEqual(option_value(injected, "--fusion-ratio"), "0.0")
        self.assertEqual(option_value(injected, "--dataset"), "cifar100")
        self.assertEqual(
            option_value(injected, "--protocol-name"),
            "table7_loss_balance_"
            "lambda_0_cifar100_ours_v2_relative_position_v1",
        )

    def test_all_locked_v2_defaults_are_injected(self) -> None:
        injected = inject_locked_defaults([], 0.0, "0")
        for option, value in BASE_PROTOCOL_DEFAULTS:
            self.assertEqual(option_value(injected, option), value)

    def test_convex_sweep_validation_matches_table7(self) -> None:
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
        with self.assertRaisesRegex(ValueError, "permits only"):
            extract_lambda_value(["--lambda-value", "0.4"])

    def test_direct_identity_overrides_are_rejected(self) -> None:
        for option, value in (
            ("--fusion-ratio", "0.5"),
            ("--dataset", "cifar100"),
            ("--protocol-name", "manual"),
        ):
            with self.subTest(option=option):
                with self.assertRaises(ValueError):
                    inject_locked_defaults([option, value], 0.0, "0")

    def test_issue_contains_copyable_full_run(self) -> None:
        issue = (
            Path(__file__).resolve().parents[1]
            / "methods/OursV2/table7_loss_balance/H200_ISSUE.md"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "python methods/OursV2/table7_loss_balance/train_cifar100.py "
            "--lambda-value 0",
            issue,
        )
        self.assertIn("relative_position_v1", issue)
        self.assertIn("args.fusion_ratio = 0.0", issue)


if __name__ == "__main__":
    unittest.main()
