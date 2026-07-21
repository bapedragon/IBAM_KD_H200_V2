from __future__ import annotations

import math
import unittest

import torch
import torch.nn.functional as F

from methods.ALG.core import AdaptiveGuidanceController
from methods.ALG.official_lg import LocalityGuidance


def reference_derivatives(
    losses: list[float], window: int
) -> tuple[list[float], list[float]]:
    raw: list[float] = []
    smoothed: list[float] = []
    for index, current in enumerate(losses):
        epoch = index + 1
        if epoch == 1:
            derivative = 0.0
        elif epoch <= window:
            derivative = (current - sum(losses[:index]) / index) / epoch
        else:
            derivative = (current - losses[index - window]) / window
        raw.append(derivative)
        recent = raw[max(0, len(raw) - window) :]
        smoothed.append(sum(recent) / len(recent))
    return raw, smoothed


class AlgControllerTest(unittest.TestCase):
    def test_matches_published_equations(self) -> None:
        losses = [10.0 - 0.1 * epoch for epoch in range(120)]
        expected_raw, expected_smoothed = reference_derivatives(losses, 50)
        controller = AdaptiveGuidanceController(
            beta=2.5,
            threshold=1.0,
            smoothing_window=50,
        )
        for epoch, loss in enumerate(losses, 1):
            self.assertEqual(controller.beta_for_epoch(epoch), 2.5)
            controller.observe(epoch, loss)

        for actual, expected in zip(
            controller.derivative_history,
            expected_raw,
            strict=True,
        ):
            self.assertAlmostEqual(actual, expected, places=12)
        for actual, expected in zip(
            controller.smoothed_derivative_history,
            expected_smoothed,
            strict=True,
        ):
            self.assertAlmostEqual(actual, expected, places=12)

    def test_crossing_epoch_is_last_guided_epoch(self) -> None:
        controller = AdaptiveGuidanceController(
            beta=2.5,
            threshold=-0.02,
            smoothing_window=3,
        )
        for epoch, loss in enumerate(
            [4.0, 3.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
            1,
        ):
            beta = controller.beta_for_epoch(epoch)
            if beta == 0.0:
                break
            controller.observe(epoch, loss)
        self.assertIsNotNone(controller.stop_epoch)
        assert controller.stop_epoch is not None
        self.assertEqual(controller.beta_history[controller.stop_epoch - 1], 2.5)
        self.assertEqual(controller.beta_for_epoch(controller.stop_epoch + 1), 0.0)

    def test_early_increase_cannot_stop_before_descent(self) -> None:
        controller = AdaptiveGuidanceController(
            beta=2.5,
            threshold=-0.02,
            smoothing_window=50,
        )
        for epoch, loss in enumerate([4.0, 4.2], 1):
            self.assertEqual(controller.beta_for_epoch(epoch), 2.5)
            controller.observe(epoch, loss)
        self.assertFalse(controller.descent_observed)
        self.assertTrue(controller.active)
        self.assertIsNone(controller.stop_epoch)


class OfficialLgPortTest(unittest.TestCase):
    def test_shapes_and_loss_match_public_implementation(self) -> None:
        torch.manual_seed(7)
        student = [
            torch.randn(2, 192, 14, 14, requires_grad=True) for _ in range(3)
        ]
        teacher = [
            torch.randn(2, 16, 32, 32),
            torch.randn(2, 32, 16, 16),
            torch.randn(2, 64, 8, 8),
        ]
        module = LocalityGuidance()
        loss, aligned_student, aligned_teacher = module(student, teacher)
        expected_shapes = [
            (2, 16, 32, 32),
            (2, 32, 16, 16),
            (2, 64, 14, 14),
        ]
        self.assertEqual(
            [tuple(value.shape) for value in aligned_student], expected_shapes
        )
        self.assertEqual(
            [tuple(value.shape) for value in aligned_teacher], expected_shapes
        )
        manual = sum(
            F.mse_loss(student_value, teacher_value)
            for student_value, teacher_value in zip(
                aligned_student,
                aligned_teacher,
                strict=True,
            )
        )
        self.assertTrue(
            math.isclose(
                float(loss.detach()),
                float(manual.detach()),
                rel_tol=1e-7,
            )
        )
        loss.backward()
        self.assertIsNotNone(student[0].grad)


if __name__ == "__main__":
    unittest.main()
