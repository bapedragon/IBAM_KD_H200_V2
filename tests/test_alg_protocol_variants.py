from __future__ import annotations

import unittest

import torch

from methods.ALG.chaoyang.train import PROTOCOL_DEFAULTS as PUBLIC_DEFAULTS
from methods.ALG.flowers102.train import PROTOCOL_DEFAULTS as FLOWERS_DEFAULTS
from methods.ALG.chaoyang.train_draft_common import (
    PROTOCOL_DEFAULTS as DRAFT_COMMON_DEFAULTS,
    PROTOCOL_FLAGS as DRAFT_COMMON_FLAGS,
)
from methods.ALG.core import create_draft_common_scheduler
from methods.Ours.core import create_ours_scheduler


def as_map(values: tuple[tuple[str, str], ...]) -> dict[str, str]:
    return dict(values)


class AlgProtocolVariantsTest(unittest.TestCase):
    def test_shared_paper_visible_values_are_identical(self) -> None:
        public = as_map(PUBLIC_DEFAULTS)
        draft = as_map(DRAFT_COMMON_DEFAULTS)
        for option in (
            "--student-epochs",
            "--lr",
            "--weight-decay",
            "--warmup-epochs",
            "--teacher-image-size",
            "--beta",
            "--alg-threshold",
            "--alg-smoothing-window",
        ):
            self.assertEqual(public[option], draft[option], option)

    def test_public_lg_alg_family_is_locked(self) -> None:
        defaults = as_map(PUBLIC_DEFAULTS)
        self.assertEqual(defaults["--base-protocol"], "lg_official")
        self.assertEqual(defaults["--min-lr"], "0.000005")
        self.assertEqual(defaults["--label-smoothing"], "0.0")
        self.assertEqual(defaults["--drop-path-rate"], "0.1")
        self.assertEqual(defaults["--eval-resize-mode"], "direct")
        self.assertEqual(defaults["--seed"], "1")
        self.assertEqual(defaults["--batch-size"], "64")
        self.assertEqual(defaults["--eval-batch-size"], "200")
        self.assertEqual(defaults["--alg-warmup-epochs"], "20")
        self.assertEqual(
            defaults["--protocol-name"],
            "chaoyang_deit_ti_alg_researcher_sync_v1",
        )

    def test_draft_common_family_matches_historical_ours_base(self) -> None:
        defaults = as_map(DRAFT_COMMON_DEFAULTS)
        self.assertEqual(defaults["--base-protocol"], "draft_common")
        self.assertEqual(defaults["--min-lr"], "0.0")
        self.assertEqual(defaults["--label-smoothing"], "0.1")
        self.assertEqual(defaults["--drop-path-rate"], "0.0")
        self.assertEqual(defaults["--eval-resize-mode"], "center_crop")
        self.assertEqual(defaults["--seed"], "42")
        self.assertEqual(defaults["--batch-size"], "128")
        self.assertEqual(DRAFT_COMMON_FLAGS, ("--amp",))

    def test_draft_common_lr_curve_is_identical_to_historical_ours(self) -> None:
        parameter_a = torch.nn.Parameter(torch.zeros(()))
        parameter_b = torch.nn.Parameter(torch.zeros(()))
        optimizer_a = torch.optim.SGD([parameter_a], lr=5e-4)
        optimizer_b = torch.optim.SGD([parameter_b], lr=5e-4)
        alg_scheduler, alg_warmup = create_draft_common_scheduler(
            optimizer_a, 300, 20, 0.0, 5e-4
        )
        ours_scheduler, ours_warmup = create_ours_scheduler(
            optimizer_b, 300, 20, 0.0, 5e-4
        )
        self.assertEqual(alg_warmup, ours_warmup)
        for epoch_index in (0, 1, 19, 20, 21, 100, 200, 299, 300):
            self.assertAlmostEqual(
                alg_scheduler.lr_lambdas[0](epoch_index),
                ours_scheduler.lr_lambdas[0](epoch_index),
                places=12,
            )

    def test_flowers_uses_researcher_sync_base(self) -> None:
        defaults = as_map(FLOWERS_DEFAULTS)
        public = as_map(PUBLIC_DEFAULTS)
        self.assertEqual(
            defaults["--protocol-name"],
            "flowers102_deit_ti_alg_researcher_sync_v1",
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


if __name__ == "__main__":
    unittest.main()
