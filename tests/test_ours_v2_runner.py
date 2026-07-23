from __future__ import annotations

import argparse
import unittest
from pathlib import Path

from methods.OursV2.run_cifar100_controls import (
    POD_LIMIT_SECONDS,
    TASKS,
    build_command,
    format_duration,
)


class OursV2ControlRunnerTest(unittest.TestCase):
    def test_task_order_is_grid_permutation_then_token_space(self) -> None:
        self.assertEqual(
            [task.variant for task in TASKS],
            ["grid_permutation_v1", "token_space_v1"],
        )

    def test_timing_commands_are_isolated_and_keep_exact_protocol_names(
        self,
    ) -> None:
        args = argparse.Namespace(
            timing_run=True,
            full_run=False,
            data_dir=Path("/data"),
            teacher_root=Path("/teachers"),
            num_workers=4,
        )
        output_root = Path("/tmp/ours-v2-test")
        run_dirs: list[Path] = []
        for task in TASKS:
            command, run_dir = build_command(task, args, output_root)
            run_dirs.append(run_dir)
            self.assertIn("--timing-run", command)
            self.assertIn("--protocol-name", command)
            self.assertIn(task.protocol_name, command)
            self.assertIn(task.variant, str(run_dir))
            self.assertIn("--num-workers", command)
            self.assertIn("4", command)
        self.assertNotEqual(run_dirs[0], run_dirs[1])

    def test_pod_limit_and_duration_format(self) -> None:
        self.assertEqual(POD_LIMIT_SECONDS, 600 * 60)
        self.assertEqual(format_duration(3 * 3600 + 4 * 60 + 5), "3h 04m 05s")

    def test_issue_contains_copyable_timing_request(self) -> None:
        issue = (
            Path(__file__).resolve().parents[1]
            / "methods/OursV2/H200_ISSUE.md"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "python methods/OursV2/run_cifar100_controls.py "
            "--timing-run --num-workers 4",
            issue,
        )
        self.assertIn("completed_tasks=2/2", issue)
        self.assertIn("[POD_LIMIT_CHECK] status=PASS|FAIL", issue)


if __name__ == "__main__":
    unittest.main()
