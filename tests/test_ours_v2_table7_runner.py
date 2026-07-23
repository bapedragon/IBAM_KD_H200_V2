from __future__ import annotations

import argparse
import unittest
from pathlib import Path

from methods.OursV2.table7_loss_balance.run_cifar100 import (
    POD_LIMIT_SECONDS,
    TASKS,
    build_command,
    format_duration,
)


class OursV2Table7RunnerTest(unittest.TestCase):
    def test_task_order_is_lambda_zero_then_lambda_half(self) -> None:
        self.assertEqual(
            [task.lambda_value for task in TASKS],
            [0.0, 0.5],
        )

    def test_timing_commands_share_v2_model_and_isolate_outputs(self) -> None:
        args = argparse.Namespace(
            timing_run=True,
            full_run=False,
            data_dir=Path("/data"),
            teacher_root=Path("/teachers"),
            num_workers=4,
        )
        output_root = Path("/tmp/ours-v2-table7-test")
        run_dirs: list[Path] = []
        for task in TASKS:
            command, run_dir = build_command(task, args, output_root)
            run_dirs.append(run_dir)
            self.assertIn("--timing-run", command)
            self.assertIn("--lambda-value", command)
            self.assertIn(task.lambda_cli, command)
            self.assertIn(task.lambda_name, str(run_dir))
            self.assertIn("--num-workers", command)
            self.assertIn("4", command)
        self.assertNotEqual(run_dirs[0], run_dirs[1])

    def test_pod_limit_and_duration_format(self) -> None:
        self.assertEqual(POD_LIMIT_SECONDS, 600 * 60)
        self.assertEqual(format_duration(4 * 3600 + 5 * 60 + 6), "4h 05m 06s")

    def test_issue_contains_copyable_paired_timing_request(self) -> None:
        issue = (
            Path(__file__).resolve().parents[1]
            / "methods/OursV2/table7_loss_balance/H200_ISSUE.md"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "python methods/OursV2/table7_loss_balance/run_cifar100.py "
            "--timing-run --num-workers 4",
            issue,
        )
        self.assertIn("[SEQUENCE] lambda_0 -> lambda_0p5", issue)
        self.assertIn("completed_tasks=2/2", issue)
        self.assertIn("[POD_LIMIT_CHECK] status=PASS|FAIL", issue)


if __name__ == "__main__":
    unittest.main()
