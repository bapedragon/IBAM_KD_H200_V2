from __future__ import annotations

import unittest
from unittest.mock import patch

from torchvision import transforms

from methods.KD.core import STUDENT_IMAGE_SIZE, build_eval_transform
from methods.Ours.chaoyang.train import PROTOCOL_DEFAULTS as CHAOYANG_DEFAULTS
from methods.Ours.cifar100.train import PROTOCOL_DEFAULTS as CIFAR_DEFAULTS
from methods.Ours.core import parse_args
from methods.Ours.flowers102.train import PROTOCOL_DEFAULTS as FLOWERS_DEFAULTS


def defaults_map(defaults: tuple[tuple[str, str], ...]) -> dict[str, str]:
    return dict(defaults)


class OursSourceGridProtocolTest(unittest.TestCase):
    def test_shared_ours_defaults_match_loss_and_grid_contract(self) -> None:
        with patch("sys.argv", ["ours"]):
            args = parse_args()
        self.assertEqual(args.student_epochs, 300)
        self.assertEqual(args.fusion_ratio, 0.5)
        self.assertEqual(args.beta_schedule, "alg")
        self.assertEqual(args.beta_on, 2.5)
        self.assertEqual(args.alg_threshold, -0.02)
        self.assertEqual(args.alg_smoothing_window, 50)
        self.assertEqual(args.alg_warmup_epochs, 20)
        self.assertEqual(args.grid_resize_mode, "larger")
        self.assertFalse(args.amp)

    def test_cifar_and_flowers_follow_supplied_source_larger_grid_policy(self) -> None:
        for defaults in (CIFAR_DEFAULTS, FLOWERS_DEFAULTS):
            with self.subTest(protocol=defaults_map(defaults)["--protocol-name"]):
                self.assertEqual(
                    defaults_map(defaults)["--grid-resize-mode"], "larger"
                )
                self.assertEqual(
                    defaults_map(defaults)["--eval-resize-mode"], "direct"
                )

    def test_chaoyang_researcher_sync_uses_larger_grid(self) -> None:
        defaults = defaults_map(CHAOYANG_DEFAULTS)
        self.assertEqual(defaults["--grid-resize-mode"], "larger")
        self.assertEqual(defaults["--eval-resize-mode"], "direct")

    def test_direct_eval_preserves_full_224px_field_of_view(self) -> None:
        transform = build_eval_transform("chaoyang", resize_mode="direct")
        self.assertEqual(len(transform.transforms), 3)
        self.assertIsInstance(transform.transforms[0], transforms.Resize)
        self.assertEqual(
            transform.transforms[0].size,
            (STUDENT_IMAGE_SIZE, STUDENT_IMAGE_SIZE),
        )
        self.assertFalse(
            any(isinstance(item, transforms.CenterCrop) for item in transform.transforms)
        )

    def test_generic_kd_center_crop_transform_remains_available(self) -> None:
        transform = build_eval_transform("chaoyang", resize_mode="center_crop")
        self.assertTrue(
            any(isinstance(item, transforms.CenterCrop) for item in transform.transforms)
        )

    def test_chaoyang_protocol_is_locked(self) -> None:
        defaults = defaults_map(CHAOYANG_DEFAULTS)
        self.assertEqual(
            defaults["--protocol-name"],
            "chaoyang_deit_ti_ours_researcher_sync_v1",
        )
        self.assertEqual(defaults["--student-epochs"], "300")
        self.assertEqual(defaults["--batch-size"], "64")
        self.assertEqual(defaults["--warmup-epochs"], "20")
        self.assertEqual(defaults["--warmup-factor"], "0.001")
        self.assertEqual(defaults["--min-lr"], "0.000005")
        self.assertEqual(defaults["--label-smoothing"], "0.0")
        self.assertEqual(defaults["--drop-path-rate"], "0.1")
        self.assertEqual(defaults["--eval-batch-size"], "200")
        self.assertEqual(defaults["--seed"], "1")
        self.assertEqual(defaults["--base-protocol"], "lg_official")
        self.assertEqual(defaults["--teacher-image-size"], "32")
        self.assertEqual(defaults["--beta-schedule"], "alg")
        self.assertEqual(defaults["--beta-on"], "2.5")
        self.assertEqual(defaults["--alg-threshold"], "-0.02")
        self.assertEqual(defaults["--alg-smoothing-window"], "50")
        self.assertEqual(defaults["--alg-warmup-epochs"], "20")
        self.assertEqual(defaults["--grid-resize-mode"], "larger")
        self.assertEqual(defaults["--eval-resize-mode"], "direct")


if __name__ == "__main__":
    unittest.main()
