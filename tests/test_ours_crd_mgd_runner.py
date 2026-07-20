"""Ordering checks for the CIFAR-100 Ours/CRD/MGD H200 runner."""

from __future__ import annotations

import unittest

from methods.run_cifar100_ours_crd_mgd import METHODS


class OursCrdMgdRunnerTest(unittest.TestCase):
    def test_ours_runs_first(self) -> None:
        self.assertEqual([method for method, _, _ in METHODS], ["Ours", "CRD", "MGD"])

    def test_output_prefixes_are_unique(self) -> None:
        prefixes = [prefix for _, _, prefix in METHODS]
        self.assertEqual(len(prefixes), len(set(prefixes)))


if __name__ == "__main__":
    unittest.main()
