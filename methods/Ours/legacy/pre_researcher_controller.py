"""Archived controller used by Ours runs through Git commit ee2dc55.

This module is intentionally not imported by the active training code.
"""

from __future__ import annotations

from typing import Any


class PreResearcherAdaptiveGuidanceController:
    """Historical descent-guard controller retained for result provenance."""

    def __init__(self, *, beta: float = 2.5, tau: float = -0.02, window: int = 50):
        self.beta = float(beta)
        self.tau = float(tau)
        self.window = int(window)
        self.active = True
        self.descent_observed = False
        self.stop_epoch: int | None = None
        self.loss_history: list[float] = []
        self.derivative_history: list[float] = []

    def beta_for_epoch(self) -> float:
        return self.beta if self.active else 0.0

    def observe_alignment_loss(self, epoch: int, alignment_loss: float) -> dict[str, Any]:
        self.loss_history.append(float(alignment_loss))
        if epoch == 1:
            derivative = 0.0
        elif epoch <= self.window:
            previous_mean = sum(self.loss_history[:-1]) / (epoch - 1)
            derivative = (self.loss_history[-1] - previous_mean) / epoch
        else:
            derivative = (
                self.loss_history[-1] - self.loss_history[-1 - self.window]
            ) / self.window
        self.derivative_history.append(derivative)
        recent = self.derivative_history[-self.window :]
        smoothed = sum(recent) / len(recent)
        if epoch > 1 and smoothed < self.tau:
            self.descent_observed = True
        if epoch > 1 and self.descent_observed and smoothed >= self.tau:
            self.active = False
            self.stop_epoch = epoch
        return {
            "active": self.active,
            "descent_observed": self.descent_observed,
            "stop_epoch": self.stop_epoch,
            "smoothed_derivative": smoothed,
        }
