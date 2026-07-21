from __future__ import annotations

import unittest

from methods.Ours.chaoyang.train import PROTOCOL_DEFAULTS as CHAOYANG_DEFAULTS
from methods.Ours.cifar100.train import PROTOCOL_DEFAULTS as CIFAR_DEFAULTS
from methods.Ours.flowers102.train import PROTOCOL_DEFAULTS as FLOWERS_DEFAULTS


def defaults_map(defaults: tuple[tuple[str, str], ...]) -> dict[str, str]:
    return dict(defaults)


class OursPaperGridProtocolTest(unittest.TestCase):
    def test_all_table_wrappers_follow_v3_teacher_grid_policy(self) -> None:
        for defaults in (CIFAR_DEFAULTS, FLOWERS_DEFAULTS, CHAOYANG_DEFAULTS):
            with self.subTest(protocol=defaults_map(defaults)["--protocol-name"]):
                self.assertEqual(
                    defaults_map(defaults)["--grid-resize-mode"], "teacher"
                )

    def test_chaoyang_protocol_is_locked(self) -> None:
        defaults = defaults_map(CHAOYANG_DEFAULTS)
        self.assertEqual(
            defaults["--protocol-name"],
            "chaoyang_deit_ti_ours_papergrid_v1",
        )
        self.assertEqual(defaults["--student-epochs"], "100")
        self.assertEqual(defaults["--batch-size"], "64")
        self.assertEqual(defaults["--warmup-epochs"], "5")
        self.assertEqual(defaults["--teacher-image-size"], "32")
        self.assertEqual(defaults["--beta-schedule"], "alg")
        self.assertEqual(defaults["--beta-on"], "2.5")
        self.assertEqual(defaults["--alg-threshold"], "-0.02")
        self.assertEqual(defaults["--alg-smoothing-window"], "50")
        self.assertEqual(defaults["--grid-resize-mode"], "teacher")


if __name__ == "__main__":
    unittest.main()
