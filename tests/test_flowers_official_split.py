from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
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
from methods.run_flowers_official_split_ours_alg import TASKS


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
        for defaults in (ALG_DEFAULTS, OURS_DEFAULTS):
            values = dict(defaults)
            self.assertEqual(values["--flowers-split-policy"], "official_three_way")
            self.assertEqual(values["--student-epochs"], "300")
            self.assertEqual(values["--batch-size"], "64")
            self.assertEqual(values["--warmup-epochs"], "20")
            self.assertEqual(values["--seed"], "1")
            self.assertIn("official_split", values["--protocol-name"])

    def test_runner_executes_ours_then_alg(self) -> None:
        self.assertEqual([method for method, _ in TASKS], ["Ours", "ALG"])
        self.assertTrue(all("official_split" in str(path) for _, path in TASKS))

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
