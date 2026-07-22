from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

from methods.run_alg_batch_ablation_ours_chaoyang import TASKS, build_command


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def read_protocol_defaults() -> dict[str, str]:
    source = (
        REPOSITORY_ROOT / "methods/ALG/chaoyang/train_pure_alg.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "PROTOCOL_DEFAULTS"
            for target in node.targets
        ):
            return dict(ast.literal_eval(node.value))
    raise AssertionError("PROTOCOL_DEFAULTS was not found")


class PureAlgAndBatchRunnerTest(unittest.TestCase):
    def test_pure_alg_chaoyang_protocol_is_paper_equation_based(self) -> None:
        defaults = read_protocol_defaults()
        self.assertEqual(defaults["--student-epochs"], "300")
        self.assertEqual(defaults["--alg-warmup-epochs"], "0")
        self.assertEqual(defaults["--alg-stop-comparison"], "paper_ge")
        self.assertEqual(defaults["--alg-derivative-mode"], "paper_equations")
        self.assertEqual(defaults["--base-protocol"], "lg_official")
        self.assertEqual(defaults["--seed"], "1")

    def test_sequence_contains_requested_four_independent_tasks(self) -> None:
        self.assertEqual(
            [(task.method, task.dataset, task.batch_size) for task in TASKS],
            [
                ("ALG", "flowers102", 64),
                ("ALG", "chaoyang", 128),
                ("ALG", "chaoyang", 64),
                ("Ours", "chaoyang", 64),
            ],
        )
        self.assertEqual(len({task.protocol_name for task in TASKS}), 4)

    def test_commands_use_shared_chaoyang_mount_and_unique_run_dirs(self) -> None:
        args = SimpleNamespace(
            timing_run=True,
            data_dir=Path("data"),
            chaoyang_data_dir=Path("/app/data/chaoyang"),
            teacher_root=Path("teachers/checkpoints"),
            num_workers=4,
        )
        output_root = Path("/tmp/results")
        run_dirs = set()
        for task in TASKS:
            command, run_dir = build_command(task, args, output_root)
            run_dirs.add(run_dir)
            batch_index = command.index("--batch-size")
            self.assertEqual(command[batch_index + 1], str(task.batch_size))
            if task.dataset == "chaoyang":
                data_index = command.index("--data-dir")
                self.assertEqual(command[data_index + 1], "/app/data/chaoyang")
            self.assertIn("--timing-run", command)
        self.assertEqual(len(run_dirs), 4)


if __name__ == "__main__":
    unittest.main()
