"""Shape tests for all five generic KD methods on the fixed V2 teachers."""

from __future__ import annotations

import unittest

import torch

from methods.KD.core import create_student, student_view_to_teacher_view
from methods.MGD.core import (
    STUDENT_CHANNELS,
    TEACHER_CHANNELS,
    forward_student_feature,
    forward_teacher_feature,
)
from methods.MGD.official_mgd import MGDLoss
from methods.OFA.core import (
    STUDENT_EMBED_DIM,
    forward_student_features,
)
from methods.OFA.official_ofa import OFAProjector, ofa_loss
from teachers.verify_checkpoints import load_teacher


class FiveMethodBridgeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import timm

        cls.timm = timm

    def check_dataset(self, dataset: str, num_classes: int) -> None:
        teacher, payload, spec = load_teacher(dataset)
        student = create_student(self.timm, "deit_ti", num_classes).eval()
        student_view = torch.zeros(2, 3, 224, 224)
        teacher_view = student_view_to_teacher_view(student_view, dataset)
        self.assertEqual(tuple(teacher_view.shape), (2, 3, 32, 32))
        self.assertEqual(int(payload["num_classes"]), num_classes)
        self.assertEqual(int(spec["num_classes"]), num_classes)

        mgd = MGDLoss(
            STUDENT_CHANNELS,
            TEACHER_CHANNELS,
            alpha_mgd=0.00007,
            lambda_mgd=0.15,
        ).eval()
        with torch.inference_mode():
            teacher_feature = forward_teacher_feature(teacher, teacher_view, 14)
            student_feature, student_logits = forward_student_feature(
                student, student_view, 11
            )
            aligned = mgd.align_feature(student_feature)
        self.assertEqual(tuple(teacher_feature.shape), (2, 64, 14, 14))
        self.assertEqual(tuple(student_feature.shape), (2, 192, 14, 14))
        self.assertEqual(tuple(aligned.shape), (2, 64, 14, 14))
        self.assertEqual(tuple(student_logits.shape), (2, num_classes))

        stages = (1, 2, 3, 4)
        projector = OFAProjector(
            stages=stages,
            embed_dim=STUDENT_EMBED_DIM,
            patch_grid=14,
            num_classes=num_classes,
        ).eval()
        with torch.inference_mode():
            teacher_logits = teacher(teacher_view)
            tokens, student_logits = forward_student_features(
                student, student_view, stages
            )
            projected = projector(tokens)
        self.assertEqual(tuple(teacher_logits.shape), (2, num_classes))
        self.assertEqual(tuple(student_logits.shape), (2, num_classes))
        self.assertEqual([tuple(item.shape) for item in tokens], [(2, 197, 192)] * 4)
        self.assertEqual(
            [tuple(item.shape) for item in projected],
            [(2, num_classes)] * 4,
        )

        # The timing probe must also exercise the trainable heterogeneous
        # adapters, not only their inference shapes.
        mgd_train = MGDLoss(
            STUDENT_CHANNELS,
            TEACHER_CHANNELS,
            alpha_mgd=0.00007,
            lambda_mgd=0.15,
        )
        synthetic_student_feature = torch.randn(
            2, STUDENT_CHANNELS, 14, 14, requires_grad=True
        )
        synthetic_teacher_feature = torch.randn(2, TEACHER_CHANNELS, 14, 14)
        mgd_loss = mgd_train(synthetic_student_feature, synthetic_teacher_feature)
        self.assertTrue(torch.isfinite(mgd_loss))
        mgd_loss.backward()
        self.assertIsNotNone(synthetic_student_feature.grad)

        ofa_train = OFAProjector(
            stages=stages,
            embed_dim=STUDENT_EMBED_DIM,
            patch_grid=14,
            num_classes=num_classes,
        )
        synthetic_tokens = [
            torch.randn(2, 197, STUDENT_EMBED_DIM, requires_grad=True)
            for _ in stages
        ]
        synthetic_teacher_logits = torch.randn(2, num_classes)
        targets = torch.tensor([0, num_classes - 1])
        target_mask = torch.nn.functional.one_hot(
            targets, num_classes=num_classes
        ).float()
        ofa_total = sum(
            ofa_loss(item, synthetic_teacher_logits, target_mask, 1.0)
            for item in ofa_train(synthetic_tokens)
        )
        self.assertTrue(torch.isfinite(ofa_total))
        ofa_total.backward()
        self.assertTrue(all(item.grad is not None for item in synthetic_tokens))

    def test_flowers102_bridges(self) -> None:
        self.check_dataset("flowers102", 102)

    def test_chaoyang_bridges(self) -> None:
        self.check_dataset("chaoyang", 4)


if __name__ == "__main__":
    unittest.main()
