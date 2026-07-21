from __future__ import annotations

import inspect
import math
import unittest

import torch
import torch.nn.functional as F

from methods.ALG.core import AdaptiveGuidanceController, main
from methods.ALG.official_lg import LocalityGuidance


def delta_at(losses: list[float], epoch: int, window: int) -> float | None:
    if epoch < 2:
        return None
    if epoch <= window:
        previous_mean = sum(losses[: epoch - 1]) / (epoch - 1)
        return (losses[epoch - 1] - previous_mean) / epoch
    return (losses[epoch - 1] - losses[epoch - window - 1]) / window


def researcher_smoothed(
    losses: list[float], epoch: int, window: int
) -> float | None:
    if epoch < 2:
        return None
    if epoch <= window:
        values = [delta_at(losses, index, window) for index in range(2, epoch + 1)]
        return sum(float(value) for value in values if value is not None) / len(values)
    if epoch < 2 * window:
        values = [
            delta_at(losses, index, window)
            for index in range(epoch - window + 1, epoch + 1)
        ]
        return sum(float(value) for value in values if value is not None) / len(values)
    total = sum(
        losses[index - 1] - losses[index - window - 1]
        for index in range(epoch - window + 1, epoch + 1)
    )
    return total / (window**2)


class AlgControllerTest(unittest.TestCase):
    def test_matches_researcher_three_case_equations(self) -> None:
        losses = [10.0 - 0.1 * epoch for epoch in range(120)]
        controller = AdaptiveGuidanceController(
            beta=2.5,
            threshold=1.0,
            smoothing_window=50,
            warm_up=20,
        )
        for epoch, loss in enumerate(losses, 1):
            self.assertEqual(controller.beta_for_epoch(epoch), 2.5)
            controller.observe(epoch, loss)

        for epoch, actual in enumerate(
            controller.smoothed_derivative_history, 1
        ):
            if epoch < 20:
                self.assertIsNone(actual)
            else:
                expected = researcher_smoothed(losses, epoch, 50)
                self.assertIsNotNone(actual)
                self.assertAlmostEqual(float(actual), float(expected), places=12)

    def test_controller_observes_complete_alg_lg_loss(self) -> None:
        source = inspect.getsource(main)
        self.assertIn("controller.observe(epoch, lg_loss)", source)

    def test_warmup_blocks_early_stop(self) -> None:
        controller = AdaptiveGuidanceController(
            beta=2.5,
            threshold=-0.02,
            smoothing_window=50,
            warm_up=4,
        )
        for epoch, loss in enumerate([4.0, 4.2, 4.4], 1):
            self.assertEqual(controller.beta_for_epoch(epoch), 2.5)
            controller.observe(epoch, loss)
            self.assertTrue(controller.active)
        self.assertEqual(controller.beta_for_epoch(4), 2.5)
        controller.observe(4, 4.6)
        self.assertFalse(controller.active)
        self.assertEqual(controller.stop_epoch, 4)

    def test_no_descent_guard_matches_researcher_code(self) -> None:
        controller = AdaptiveGuidanceController(
            beta=2.5,
            threshold=-0.02,
            smoothing_window=50,
            warm_up=2,
        )
        controller.beta_for_epoch(1)
        controller.observe(1, 4.0)
        controller.beta_for_epoch(2)
        controller.observe(2, 4.2)
        self.assertFalse(controller.active)
        self.assertEqual(controller.stop_epoch, 2)

    def test_strict_greater_than_threshold(self) -> None:
        controller = AdaptiveGuidanceController(
            beta=2.5,
            threshold=0.0,
            smoothing_window=50,
            warm_up=2,
        )
        controller.beta_for_epoch(1)
        controller.observe(1, 4.0)
        controller.beta_for_epoch(2)
        controller.observe(2, 4.0)
        self.assertTrue(controller.active)
        self.assertIsNone(controller.stop_epoch)

    def test_crossing_epoch_is_last_guided_epoch(self) -> None:
        controller = AdaptiveGuidanceController(
            beta=2.5,
            threshold=-0.02,
            smoothing_window=3,
            warm_up=4,
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
