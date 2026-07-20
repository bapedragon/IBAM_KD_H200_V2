"""Shape and checkpoint tests for the V2 CIFAR-100 distillation bridge."""

from __future__ import annotations

import unittest

import torch

from methods.CRD.core import (
    TEACHER_DIM,
    forward_student_with_rep,
    forward_teacher_with_rep,
)
from methods.KD.core import create_student, student_view_to_teacher_view
from methods.ReviewKD.core import (
    STUDENT_CHANNELS,
    TEACHER_CHANNELS,
    forward_student_features,
    forward_teacher_features,
)
from methods.ReviewKD.official_reviewkd import ReviewKDAdapter
from teachers.verify_checkpoints import load_teacher


class Cifar100MethodBridgeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import timm

        cls.teacher, cls.payload, cls.spec = load_teacher("cifar100")
        cls.student = create_student(timm, "deit_ti", 100)
        cls.student.eval()
        cls.student_view = torch.zeros(2, 3, 224, 224)
        cls.teacher_view = student_view_to_teacher_view(cls.student_view)

    def test_fixed_teacher_and_resolution_bridge(self) -> None:
        self.assertEqual(tuple(self.teacher_view.shape), (2, 3, 32, 32))
        self.assertEqual(int(self.payload["epoch"]), int(self.spec["epoch"]))
        with torch.inference_mode():
            logits = self.teacher(self.teacher_view)
        self.assertEqual(tuple(logits.shape), (2, 100))
        self.assertTrue(bool(torch.isfinite(logits).all()))

    def test_crd_representation_bridge(self) -> None:
        with torch.inference_mode():
            teacher_rep, teacher_logits = forward_teacher_with_rep(
                self.teacher, self.teacher_view
            )
            student_rep, student_logits = forward_student_with_rep(
                self.student, self.student_view
            )
        self.assertEqual(tuple(teacher_rep.shape), (2, TEACHER_DIM))
        self.assertEqual(tuple(student_rep.shape), (2, 192))
        self.assertEqual(tuple(teacher_logits.shape), (2, 100))
        self.assertEqual(tuple(student_logits.shape), (2, 100))

    def test_reviewkd_bilinear_grid_bridge(self) -> None:
        adapter = ReviewKDAdapter(
            STUDENT_CHANNELS,
            TEACHER_CHANNELS,
            mid_channels=192,
        ).eval()
        with torch.inference_mode():
            teacher_features = forward_teacher_features(
                self.teacher, self.teacher_view, feature_grid=14
            )
            student_features, student_logits = forward_student_features(
                self.student, self.student_view, (3, 7, 11)
            )
            reviewed = adapter(student_features)
        expected_teacher = [(2, 16, 14, 14), (2, 32, 14, 14), (2, 64, 14, 14)]
        self.assertEqual([tuple(item.shape) for item in teacher_features], expected_teacher)
        self.assertEqual(
            [tuple(item.shape) for item in student_features],
            [(2, 192, 14, 14)] * 3,
        )
        self.assertEqual([tuple(item.shape) for item in reviewed], expected_teacher)
        self.assertEqual(tuple(student_logits.shape), (2, 100))


if __name__ == "__main__":
    unittest.main()
