from __future__ import annotations

import unittest

from methods.run_cub200_lg_alg_ours import (
    STUDENT_RUN_NAMES,
    STUDENT_SCRIPTS,
    collect_best_top1,
    format_final_top1_summary,
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

    def test_final_log_compacts_all_best_top1_results(self) -> None:
        summaries = {
            "teacher": {"best_top1": 51.234},
            "LG": {"best_top1": 62.345},
            "ALG": {"best_top1": 63.456},
            "Ours": {"best_top1": 64.567},
        }
        results = collect_best_top1(summaries)
        self.assertEqual(
            format_final_top1_summary(results),
            (
                "[FINAL_TOP1_SUMMARY] Teacher=51.23% LG=62.34% "
                "ALG=63.46% Ours=64.57%"
            ),
        )

    def test_final_log_rejects_an_incomplete_sequence(self) -> None:
        with self.assertRaisesRegex(KeyError, "Missing completed summary for Ours"):
            collect_best_top1(
                {
                    "teacher": {"best_top1": 1.0},
                    "LG": {"best_top1": 2.0},
                    "ALG": {"best_top1": 3.0},
                }
            )


if __name__ == "__main__":
    unittest.main()
