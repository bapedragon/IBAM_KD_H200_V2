from __future__ import annotations

from pathlib import Path
import unittest
from unittest import mock

import torch

from methods.ALG.chaoyang.train import PROTOCOL_DEFAULTS as PUBLIC_DEFAULTS
from methods.ALG.cifar100.train import PROTOCOL_DEFAULTS as CIFAR_DEFAULTS
from methods.ALG.flowers102.train import PROTOCOL_DEFAULTS as FLOWERS_DEFAULTS
from methods.ALG import core as alg_core
from methods.ALG.cub200.train import PROTOCOL_DEFAULTS as CUB_DEFAULTS


def as_map(values: tuple[tuple[str, str], ...]) -> dict[str, str]:
    return dict(values)


class AlgProtocolVariantsTest(unittest.TestCase):
    def test_canonical_lg_alg_family_is_locked(self) -> None:
        defaults = as_map(PUBLIC_DEFAULTS)
        self.assertEqual(defaults["--base-protocol"], "lg_official")
        self.assertEqual(defaults["--min-lr"], "0.000005")
        self.assertEqual(defaults["--label-smoothing"], "0.0")
        self.assertEqual(defaults["--drop-path-rate"], "0.1")
        self.assertEqual(defaults["--eval-resize-mode"], "direct")
        self.assertEqual(defaults["--seed"], "1")
        self.assertEqual(defaults["--batch-size"], "128")
        self.assertEqual(defaults["--eval-batch-size"], "200")
        self.assertEqual(defaults["--alg-warmup-epochs"], "0")
        self.assertEqual(defaults["--alg-stop-comparison"], "paper_ge")
        self.assertEqual(defaults["--alg-derivative-mode"], "paper_equations")
        self.assertEqual(
            defaults["--protocol-name"],
            "chaoyang_deit_ti_alg_paper_official_lg_v1",
        )

    def test_flowers_uses_canonical_base(self) -> None:
        defaults = as_map(FLOWERS_DEFAULTS)
        public = as_map(PUBLIC_DEFAULTS)
        self.assertEqual(
            defaults["--protocol-name"],
            "flowers102_deit_ti_alg_paper_official_lg_v1",
        )
        for option in (
            "--student-epochs",
            "--batch-size",
            "--eval-batch-size",
            "--lr",
            "--min-lr",
            "--weight-decay",
            "--warmup-epochs",
            "--warmup-factor",
            "--label-smoothing",
            "--drop-path-rate",
            "--teacher-image-size",
            "--beta",
            "--alg-threshold",
            "--alg-smoothing-window",
            "--alg-warmup-epochs",
            "--base-protocol",
            "--eval-resize-mode",
            "--seed",
        ):
            self.assertEqual(defaults[option], public[option], option)

    def test_cifar100_uses_canonical_base(self) -> None:
        defaults = as_map(CIFAR_DEFAULTS)
        public = as_map(PUBLIC_DEFAULTS)
        self.assertEqual(
            defaults["--protocol-name"],
            "cifar100_deit_ti_alg_paper_official_lg_v1",
        )
        for option in (
            "--student-epochs",
            "--batch-size",
            "--eval-batch-size",
            "--lr",
            "--min-lr",
            "--weight-decay",
            "--warmup-epochs",
            "--warmup-factor",
            "--label-smoothing",
            "--drop-path-rate",
            "--teacher-image-size",
            "--beta",
            "--alg-threshold",
            "--alg-smoothing-window",
            "--alg-warmup-epochs",
            "--base-protocol",
            "--eval-resize-mode",
            "--seed",
        ):
            self.assertEqual(defaults[option], public[option], option)

    def test_cub_uses_same_canonical_alg_settings(self) -> None:
        cub = as_map(CUB_DEFAULTS)
        public = as_map(PUBLIC_DEFAULTS)
        for option in (
            "--student-epochs",
            "--batch-size",
            "--eval-batch-size",
            "--lr",
            "--min-lr",
            "--weight-decay",
            "--warmup-epochs",
            "--warmup-factor",
            "--label-smoothing",
            "--drop-path-rate",
            "--teacher-image-size",
            "--beta",
            "--alg-threshold",
            "--alg-smoothing-window",
            "--alg-warmup-epochs",
            "--alg-stop-comparison",
            "--alg-derivative-mode",
            "--base-protocol",
            "--eval-resize-mode",
            "--seed",
        ):
            self.assertEqual(cub[option], public[option], option)

    def test_noncanonical_alg_variants_are_rejected(self) -> None:
        with mock.patch(
            "sys.argv",
            [
                "train.py",
                "--dataset",
                "cifar100",
                "--alg-warmup-epochs",
                "20",
            ],
        ):
            args = alg_core.parse_args()
        args.method = "ALG"
        with self.assertRaisesRegex(ValueError, "Canonical ALG"):
            alg_core.finalize_args(args)

    def test_alg_core_accepts_cifar100_dataset(self) -> None:
        with mock.patch("sys.argv", ["train.py", "--dataset", "cifar100"]):
            args = alg_core.parse_args()
        self.assertEqual(args.dataset, "cifar100")
        self.assertEqual(alg_core.REFERENCE_TOP1["cifar100"]["alg"], 81.98)
        self.assertIn("cifar100", alg_core.PAPER_GUIDANCE_STOP_EPOCH)

    def test_cifar100_native_teacher_audit_uses_official_test_split(self) -> None:
        fake_dataset = torch.utils.data.TensorDataset(
            torch.zeros(4, 3, 32, 32),
            torch.zeros(4, dtype=torch.long),
        )
        args = mock.Mock(
            dataset="cifar100",
            data_dir=Path("data"),
            smoke=False,
            seed=1,
            smoke_test_samples=2,
            eval_batch_size=2,
            num_workers=0,
        )
        with mock.patch.object(
            alg_core, "official_test_transform", return_value=mock.sentinel.transform
        ), mock.patch.object(
            alg_core, "CIFAR100", return_value=fake_dataset
        ) as cifar100:
            loader = alg_core.build_native_teacher_audit_loader(
                args, torch.device("cpu")
            )
        cifar100.assert_called_once_with(
            root=Path("data"),
            train=False,
            transform=mock.sentinel.transform,
            download=False,
        )
        self.assertEqual(len(loader.dataset), 4)


if __name__ == "__main__":
    unittest.main()
