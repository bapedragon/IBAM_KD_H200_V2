from __future__ import annotations

import unittest

from methods.run_cub200_lg_alg_ours import (
    STUDENT_RUN_NAMES,
    STUDENT_SCRIPTS,
)


class Cub200LgAlgRunnerTest(unittest.TestCase):
    def test_students_run_official_lg_paper_alg_then_unchanged_ours(self) -> None:
        self.assertEqual(list(STUDENT_SCRIPTS), ["LG", "ALG", "Ours"])
        self.assertEqual(
            STUDENT_SCRIPTS,
            {
                "LG": "methods/LG/cub200/train.py",
                "ALG": "methods/ALG/cub200/train.py",
                "Ours": "methods/Ours/cub200/train.py",
            },
        )

    def test_all_outputs_are_method_labeled(self) -> None:
        self.assertEqual(set(STUDENT_RUN_NAMES), {"LG", "ALG", "Ours"})
        self.assertEqual(len(set(STUDENT_RUN_NAMES.values())), 3)
        self.assertTrue(STUDENT_RUN_NAMES["LG"].startswith("lg_"))
        self.assertTrue(STUDENT_RUN_NAMES["ALG"].startswith("alg_"))
        self.assertTrue(STUDENT_RUN_NAMES["Ours"].startswith("ours_"))


if __name__ == "__main__":
    unittest.main()
