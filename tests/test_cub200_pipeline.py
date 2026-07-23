from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from methods.Ours import core as ours_core
from methods.Ours.cub200.dataset import CUB200Dataset, read_records
from methods.Ours.cub200.train import PROTOCOL_DEFAULTS
from methods.Ours.cub200.train_teacher import ResNet56CUB200
from teachers.verify_checkpoints import load_teacher


class Cub200PipelineTest(unittest.TestCase):
    def make_synthetic_dataset(self, parent: Path) -> Path:
        root = parent / "CUB_200_2011"
        image_root = root / "images"
        entries = (
            (1, "001.Class_one/image_0001.jpg", 1, 1),
            (2, "001.Class_one/image_0002.jpg", 1, 0),
            (3, "002.Class_two/image_0003.jpg", 2, 1),
            (4, "002.Class_two/image_0004.jpg", 2, 0),
        )
        for _, relative_path, _, _ in entries:
            path = image_root / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (8, 6), color=(50, 100, 150)).save(path)
        (root / "images.txt").write_text(
            "".join(f"{image_id} {path}\n" for image_id, path, _, _ in entries),
            encoding="utf-8",
        )
        (root / "image_class_labels.txt").write_text(
            "".join(
                f"{image_id} {class_id}\n"
                for image_id, _, class_id, _ in entries
            ),
            encoding="utf-8",
        )
        (root / "train_test_split.txt").write_text(
            "".join(
                f"{image_id} {is_train}\n"
                for image_id, _, _, is_train in entries
            ),
            encoding="utf-8",
        )
        return root

    def test_official_metadata_parser_and_zero_based_targets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            root = self.make_synthetic_dataset(parent)
            records = read_records(root)
            self.assertEqual([record.image_id for record in records], [1, 2, 3, 4])
            train = CUB200Dataset(parent, split="train")
            test = CUB200Dataset(root, split="test")
            self.assertEqual(len(train), 2)
            self.assertEqual(len(test), 2)
            image, target = train[0]
            self.assertEqual(image.mode, "RGB")
            self.assertEqual(target, 0)
            self.assertEqual(test[1][1], 1)

    def test_metadata_parser_rejects_unsafe_image_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = self.make_synthetic_dataset(Path(temporary))
            (root / "images.txt").write_text(
                "1 ../outside.jpg\n2 safe.jpg\n3 safe2.jpg\n4 safe3.jpg\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                read_records(root)

    def test_ours_registry_and_locked_defaults_include_cub200(self) -> None:
        self.assertEqual(ours_core.NUM_CLASSES["cub200"], 200)
        self.assertIsNone(ours_core.VANILLA_TOP1["cub200"]["deit_ti"])
        defaults = dict(PROTOCOL_DEFAULTS)
        self.assertEqual(defaults["--base-protocol"], "lg_official")
        self.assertEqual(defaults["--student-epochs"], "300")
        self.assertEqual(defaults["--batch-size"], "64")
        self.assertEqual(defaults["--seed"], "1")

    def test_generated_teacher_manifest_loads_and_hash_verifies(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "teacher_resnet56_cub200_32_best.pt"
            model = ResNet56CUB200()
            payload = {
                "epoch": 3,
                "accuracy": 12.5,
                "dataset": "cub200",
                "num_classes": 200,
                "model": model.state_dict(),
            }
            torch.save(payload, checkpoint)
            digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            manifest = {
                "version": 1,
                "teachers": {
                    "cub200": {
                        "selected_kind": "best",
                        "checkpoint": checkpoint.name,
                        "sha256": digest,
                        "epoch": 3,
                        "top1": 12.5,
                        "num_classes": 200,
                        "input_resolution": 32,
                    }
                },
            }
            (root / "manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            loaded, loaded_payload, spec = load_teacher(
                "cub200", checkpoint_root=root
            )
            self.assertEqual(loaded_payload["epoch"], 3)
            self.assertEqual(spec["num_classes"], 200)
            self.assertFalse(any(parameter.requires_grad for parameter in loaded.parameters()))
            with torch.inference_mode():
                logits = loaded(torch.zeros(2, 3, 32, 32))
            self.assertEqual(tuple(logits.shape), (2, 200))


if __name__ == "__main__":
    unittest.main()
