"""Plan checks for the combined H200 full batch."""

from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path

from methods.run_combined_full_batch import (
    CHAOYANG_FIVE_METHOD_ESTIMATE,
    CIFAR_METHOD_ESTIMATES,
    FLOWERS_FIVE_METHOD_ESTIMATE,
    POD_LIMIT_SECONDS,
    build_tasks,
)


class CombinedBatchPlanTest(unittest.TestCase):
    def test_kd_plan_order_paths_and_margin(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory)
            args = argparse.Namespace(
                cifar_method="KD",
                teacher_root=Path("teachers/checkpoints"),
                num_workers=4,
                seed=42,
            )
            tasks = build_tasks(args, output_root)

        self.assertEqual(
            [task["dataset"] for task in tasks],
            ["chaoyang", "flowers102", "cifar100"],
        )
        self.assertEqual(len(tasks[0]["methods"]), 5)
        self.assertEqual(len(tasks[1]["methods"]), 5)
        self.assertEqual(tasks[2]["methods"], ["KD"])
        self.assertEqual(tasks[2]["output_dir"].name, "cifar100_kd")

        estimated_total = (
            CHAOYANG_FIVE_METHOD_ESTIMATE
            + FLOWERS_FIVE_METHOD_ESTIMATE
            + CIFAR_METHOD_ESTIMATES["KD"]
        )
        self.assertEqual(estimated_total, 7 * 3600 + 45 * 60 + 59)
        self.assertGreater(POD_LIMIT_SECONDS - estimated_total, 2 * 3600)


if __name__ == "__main__":
    unittest.main()
