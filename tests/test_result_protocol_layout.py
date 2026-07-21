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
        self.assertEqual(len(summaries), 18)
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

    def test_researcher_sync_pending_destinations_are_explicit(self) -> None:
        pending = (RESULTS / "PENDING_IMPORTS.md").read_text(encoding="utf-8")
        self.assertIn("results/Ours/cifar100/researcher_sync_v1_300ep_seed1/", pending)
        self.assertIn("results/Ours/flowers102/researcher_sync_v1_300ep_seed1/", pending)
        self.assertIn("results/ALG/flowers102/researcher_sync_v1_300ep_seed1/", pending)


if __name__ == "__main__":
    unittest.main()
