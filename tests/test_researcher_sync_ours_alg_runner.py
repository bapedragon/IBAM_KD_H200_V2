from __future__ import annotations

import unittest

from methods.run_researcher_sync_ours_alg import TASKS, task_run_name


class ResearcherSyncOursAlgRunnerTest(unittest.TestCase):
    def test_sequence_contains_exactly_requested_tasks(self) -> None:
        self.assertEqual(
            [(method, dataset) for method, dataset, _ in TASKS],
            [
                ("Ours", "cifar100"),
                ("Ours", "flowers102"),
                ("ALG", "cifar100"),
            ],
        )

    def test_outputs_are_unique_and_provenance_labeled(self) -> None:
        full_names = [
            task_run_name(method, dataset, False)
            for method, dataset, _ in TASKS
        ]
        timing_names = [
            task_run_name(method, dataset, True)
            for method, dataset, _ in TASKS
        ]
        self.assertEqual(len(full_names), len(set(full_names)))
        self.assertEqual(len(timing_names), len(set(timing_names)))
        self.assertTrue(all("researcher_sync" in name for name in full_names))
        self.assertTrue(all("300ep_seed1" in name for name in full_names))
        self.assertTrue(all("timing_2ep_seed1" in name for name in timing_names))


if __name__ == "__main__":
    unittest.main()
