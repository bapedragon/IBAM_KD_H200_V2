from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path

from methods.run_flowers_chaoyang_300ep import (
    DATASETS,
    METHODS,
    PLANNED_EPOCHS,
    POD_LIMIT_SECONDS,
    SAFETY_MARGIN_SECONDS,
    build_tasks,
    estimated_total_seconds,
)


class GenericKd300EpochRunnerTest(unittest.TestCase):
    def test_plan_keeps_epoch_only_override_and_safe_margin(self) -> None:
        args = argparse.Namespace(
            teacher_root=Path("teachers/checkpoints"),
            num_workers=4,
            seed=42,
        )
        with tempfile.TemporaryDirectory() as directory:
            tasks = build_tasks(args, Path(directory))

        self.assertEqual([task["dataset"] for task in tasks], list(DATASETS))
        self.assertEqual(tasks[0]["dataset"], "chaoyang")
        self.assertEqual(tasks[1]["dataset"], "flowers102")
        for task in tasks:
            self.assertEqual(task["methods"], list(METHODS))
            self.assertIn("--planned-epochs", task["command"])
            self.assertIn(str(PLANNED_EPOCHS), task["command"])

        estimate = estimated_total_seconds()
        self.assertLess(estimate, POD_LIMIT_SECONDS)
        self.assertGreater(POD_LIMIT_SECONDS - estimate, SAFETY_MARGIN_SECONDS)
        self.assertAlmostEqual(estimate, 8 * 3600 + 18 * 60 + 18, delta=1.0)


if __name__ == "__main__":
    unittest.main()
