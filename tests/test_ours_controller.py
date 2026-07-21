from __future__ import annotations

import argparse
import inspect
import unittest

from methods.Ours.core import AdaptiveGuidanceController, main


def controller_args(**overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "beta_schedule": "alg",
        "beta_on": 2.5,
        "alg_threshold": -0.02,
        "alg_smoothing_window": 50,
        "alg_warmup_epochs": 20,
        "guidance_stop_epoch": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


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


class AdaptiveGuidanceControllerTest(unittest.TestCase):
    def test_matches_researcher_three_case_equations(self) -> None:
        losses = [10.0 - 0.1 * epoch for epoch in range(120)]
        controller = AdaptiveGuidanceController(
            controller_args(alg_threshold=1.0)
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

    def test_controller_observes_complete_feature_loss(self) -> None:
        source = inspect.getsource(main)
        self.assertIn("controller.observe(epoch, feature_loss)", source)
        self.assertNotIn("controller.observe(epoch, alignment_loss)", source)

    def test_warmup_blocks_early_stop(self) -> None:
        controller = AdaptiveGuidanceController(
            controller_args(alg_warmup_epochs=4)
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
            controller_args(alg_warmup_epochs=2)
        )
        controller.beta_for_epoch(1)
        controller.observe(1, 4.0)
        controller.beta_for_epoch(2)
        controller.observe(2, 4.2)
        self.assertFalse(controller.active)
        self.assertEqual(controller.stop_epoch, 2)

    def test_strict_greater_than_threshold(self) -> None:
        controller = AdaptiveGuidanceController(
            controller_args(alg_threshold=0.0, alg_warmup_epochs=2)
        )
        controller.beta_for_epoch(1)
        controller.observe(1, 4.0)
        controller.beta_for_epoch(2)
        controller.observe(2, 4.0)
        self.assertTrue(controller.active)
        self.assertIsNone(controller.stop_epoch)

    def test_crossing_epoch_is_last_guided_epoch(self) -> None:
        controller = AdaptiveGuidanceController(
            controller_args(alg_smoothing_window=3, alg_warmup_epochs=4)
        )
        for epoch, loss in enumerate([4.0, 3.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0], 1):
            beta = controller.beta_for_epoch(epoch)
            if beta == 0.0:
                break
            controller.observe(epoch, loss)

        self.assertEqual(controller.stop_epoch, 8)
        self.assertEqual(controller.beta_history[7], 2.5)
        self.assertEqual(controller.beta_for_epoch(9), 0.0)


if __name__ == "__main__":
    unittest.main()
