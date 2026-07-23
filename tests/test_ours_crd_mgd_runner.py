"""Ordering checks for the CIFAR-100 Ours/CRD/MGD H200 runner."""

from __future__ import annotations

import unittest

from methods.run_cifar100_ours_crd_mgd import (
    MEASURED_FULL_SECONDS,
    METHODS,
    POD_LIMIT_SECONDS,
    validate_runtime_plan,
)


class OursCrdMgdRunnerTest(unittest.TestCase):
    def test_ours_runs_first(self) -> None:
        self.assertEqual([method for method, _, _ in METHODS], ["Ours", "CRD", "MGD"])

    def test_output_prefixes_are_unique(self) -> None:
        prefixes = [prefix for _, _, prefix in METHODS]
        self.assertEqual(len(prefixes), len(set(prefixes)))

    def test_measured_all_three_exceed_pod_limit(self) -> None:
        total = sum(MEASURED_FULL_SECONDS.values())
        self.assertEqual(total, 10 * 3600 + 27 * 60 + 7)
        self.assertGreater(total, POD_LIMIT_SECONDS)

    def test_recommended_split_fits(self) -> None:
        self.assertLess(MEASURED_FULL_SECONDS["Ours"], POD_LIMIT_SECONDS)
        self.assertLess(
            MEASURED_FULL_SECONDS["CRD"] + MEASURED_FULL_SECONDS["MGD"],
            POD_LIMIT_SECONDS,
        )

    def test_full_three_method_run_is_blocked_by_default(self) -> None:
        with self.assertRaises(RuntimeError):
            validate_runtime_plan(
                ["Ours", "CRD", "MGD"],
                full_run=True,
                allow_over_limit=False,
            )

    def test_full_three_method_run_can_be_explicitly_allowed(self) -> None:
        total = validate_runtime_plan(
            ["Ours", "CRD", "MGD"],
            full_run=True,
            allow_over_limit=True,
        )
        self.assertEqual(total, 10 * 3600 + 27 * 60 + 7)


if __name__ == "__main__":
    unittest.main()
