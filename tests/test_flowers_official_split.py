from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from contextlib import redirect_stdout
import io
import inspect
import json
import sys
import tempfile
import unittest
from unittest import mock

import torch
from torch.utils.data import Dataset

from methods.ALG import core as alg_core
from methods.ALG.flowers102.train_official_split import (
    PROTOCOL_DEFAULTS as ALG_DEFAULTS,
)
from methods.Ours.flowers102.train_official_split import (
    PROTOCOL_DEFAULTS as OURS_DEFAULTS,
)
from methods.run_flowers_official_split_ours_alg import TASKS, main as sequence_main


class FakeFlowers(Dataset[tuple[torch.Tensor, int]]):
    SIZES = {"train": 1020, "val": 1020, "test": 6149}

    def __init__(self, *, split: str, transform: object, **_: object) -> None:
        self.split = split
        self.transform = transform

    def __len__(self) -> int:
        return self.SIZES[self.split]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return torch.zeros(3, 224, 224), index % 102


class FlowersOfficialSplitTest(unittest.TestCase):
    def test_wrappers_lock_official_three_way_policy(self) -> None:
        alg = dict(ALG_DEFAULTS)
        ours = dict(OURS_DEFAULTS)
        for values in (alg, ours):
            self.assertEqual(values["--flowers-split-policy"], "official_three_way")
            self.assertEqual(values["--student-epochs"], "300")
            self.assertEqual(values["--batch-size"], "128")
            self.assertEqual(values["--warmup-epochs"], "20")
            self.assertEqual(values["--seed"], "1")
            self.assertIn("official_split", values["--protocol-name"])
        self.assertEqual(alg["--alg-warmup-epochs"], "0")
        self.assertEqual(alg["--alg-stop-comparison"], "paper_ge")
        self.assertEqual(alg["--alg-derivative-mode"], "paper_equations")
        self.assertEqual(ours["--alg-warmup-epochs"], "20")
        self.assertEqual(ours["--grid-resize-mode"], "larger")

    def test_runner_executes_alg_then_ours(self) -> None:
        self.assertEqual([method for method, _ in TASKS], ["ALG", "Ours"])
        self.assertTrue(all("official_split" in str(path) for _, path in TASKS))

    def test_runner_prints_both_final_best_results(self) -> None:
        source = inspect.getsource(sequence_main)
        self.assertIn('for method in ("ALG", "Ours")', source)
        self.assertIn("[FINAL_BEST]", source)
        self.assertIn("[FINAL_COMPARISON]", source)

    def test_runner_emits_both_best_values_at_end(self) -> None:
        results = {
            "alg": (68.0, 67.5),
            "ours": (70.0, 69.25),
        }

        def fake_run(command: list[str], **_: object) -> None:
            output_dir = Path(command[command.index("--output-dir") + 1])
            run_name = command[command.index("--run-name") + 1]
            method = "ours" if run_name.startswith("ours_") else "alg"
            run_dir = output_dir / run_name
            run_dir.mkdir(parents=True, exist_ok=True)
            best_val, final_test = results[method]
            (run_dir / "summary.json").write_text(
                json.dumps(
                    {
                        "best_validation_top1": best_val,
                        "final_test_top1": final_test,
                        "estimated_planned_seconds": 60.0,
                    }
                ),
                encoding="utf-8",
            )

        with tempfile.TemporaryDirectory() as directory:
            stdout = io.StringIO()
            argv = [
                "run_flowers_official_split_ours_alg.py",
                "--timing-run",
                "--output-dir",
                directory,
                "--num-workers",
                "0",
            ]
            with mock.patch.object(sys, "argv", argv), mock.patch(
                "methods.run_flowers_official_split_ours_alg.subprocess.run",
                side_effect=fake_run,
            ), redirect_stdout(stdout):
                sequence_main()

        output = stdout.getvalue()
        self.assertIn(
            "[FINAL_BEST][ALG] best_val_top1=68.0% "
            "selected_checkpoint_test_top1=67.5%",
            output,
        )
        self.assertIn(
            "[FINAL_BEST][Ours] best_val_top1=70.0% "
            "selected_checkpoint_test_top1=69.25%",
            output,
        )
        self.assertLess(output.index("[FINAL_BEST][ALG]"), output.index("[DONE]"))
        self.assertLess(output.index("[FINAL_BEST][Ours]"), output.index("[DONE]"))

    def test_loader_keeps_train_val_and_test_disjoint(self) -> None:
        args = SimpleNamespace(
            dataset="flowers102",
            data_dir=Path("data"),
            flowers_split_policy="official_three_way",
            smoke=False,
            seed=1,
            smoke_train_samples=16,
            smoke_test_samples=8,
            batch_size=64,
            eval_batch_size=200,
            num_workers=0,
        )
        fake_timm = SimpleNamespace(
            data=SimpleNamespace(create_transform=lambda **_: mock.sentinel.train)
        )
        with mock.patch(
            "teachers.train_teacher_flowers.ensure_scipy"
        ), mock.patch(
            "teachers.train_teacher_flowers.ensure_flowers"
        ), mock.patch.object(
            alg_core, "Flowers102", FakeFlowers
        ):
            train_loader, val_loader, test_loader = (
                alg_core.build_alg_loaders_with_final_test(
                    args, torch.device("cpu"), fake_timm
                )
            )
        self.assertEqual(len(train_loader.dataset), 1020)
        self.assertEqual(len(val_loader.dataset), 1020)
        self.assertIsNotNone(test_loader)
        assert test_loader is not None
        self.assertEqual(len(test_loader.dataset), 6149)
        self.assertEqual(train_loader.dataset.split, "train")
        self.assertEqual(val_loader.dataset.split, "val")
        self.assertEqual(test_loader.dataset.split, "test")


if __name__ == "__main__":
    unittest.main()
