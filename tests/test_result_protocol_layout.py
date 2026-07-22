from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


class ResultProtocolLayoutTest(unittest.TestCase):
    def test_no_artifact_is_stored_directly_under_dataset_directory(self) -> None:
        methods = {"KD", "CRD", "ReviewKD", "MGD", "OFA", "ALG", "Ours"}
        for method_dir in RESULTS.iterdir():
            if not method_dir.is_dir() or method_dir.name not in methods:
                continue
            for dataset_dir in method_dir.iterdir():
                if not dataset_dir.is_dir():
                    continue
                direct_artifacts = [
                    path.name
                    for path in dataset_dir.iterdir()
                    if path.suffix in {".pt", ".json"}
                ]
                self.assertEqual(
                    direct_artifacts,
                    [],
                    f"Artifacts require a protocol-ID directory: {dataset_dir}",
                )

    def test_each_committed_protocol_directory_is_self_contained(self) -> None:
        summaries = sorted(RESULTS.glob("*/*/*/run_summary.json"))
        self.assertEqual(len(summaries), 33)
        for summary_path in summaries:
            run_dir = summary_path.parent
            checkpoint_path = run_dir / "student_best.pt"
            self.assertTrue(checkpoint_path.is_file(), run_dir)
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            method, dataset = summary_path.parts[-4], summary_path.parts[-3]
            self.assertEqual(payload["method"], method)
            self.assertEqual(payload["dataset"], dataset)
            args = payload["args"]
            protocol_id = run_dir.name
            self.assertIn(str(args["student_epochs"]), protocol_id)
            self.assertIn(f"seed{args['seed']}", protocol_id)

    def test_researcher_sync_import_destinations_are_explicit(self) -> None:
        expected = (
            RESULTS / "Ours/cifar100/researcher_sync_v1_300ep_seed1",
            RESULTS / "Ours/flowers102/researcher_sync_v1_300ep_seed1",
            RESULTS / "ALG/flowers102/researcher_sync_v1_300ep_seed1",
        )
        for run_dir in expected:
            self.assertTrue((run_dir / "run_summary.json").is_file(), run_dir)
            self.assertTrue((run_dir / "student_best.pt").is_file(), run_dir)

    def test_consolidated_table_uses_only_current_reporting_results(self) -> None:
        root_readme = (ROOT / "README.md").read_text(encoding="utf-8")
        results_readme = (RESULTS / "README.md").read_text(encoding="utf-8")
        expected_rows = (
            "| KD | Logits | 69.10 | 48.95 | 62.79 |",
            "| CRD | Pooled contrastive | 68.59 | 49.06 | 79.85 |",
            "| ReviewKD | Projected fusion | 75.65 | 61.88 | 82.75 |",
            "| MGD | Masked reconstruction | 75.68 | 54.66 | 81.81 |",
            "| OFA | Logit-space projection | 67.73 | 46.41 | 78.03 |",
            "| LG | Direct match (static) |  |  |  |",
            "| ALG | Scheduled match (static) |  |  |  |",
            "| **Ours** | **Grid-space, learnable** | **82.90** | **74.81** | **81.95\\*** |",
        )
        for row in expected_rows:
            self.assertIn(row, root_readme)
            self.assertIn(row, results_readme)

        pending = (RESULTS / "PENDING_IMPORTS.md").read_text(encoding="utf-8")
        self.assertIn("Ours | Chaoyang", pending)
        self.assertIn("researcher_sync_v1_300ep_seed1", pending)
        self.assertIn("81.95%", pending)


if __name__ == "__main__":
    unittest.main()
