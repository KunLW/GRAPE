from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def clip_amplitudes(amplitudes, lower=None, upper=None):
    return np.clip(amplitudes, lower, upper)


@dataclass(frozen=True)
class PulseConstraints:
    amplitude_lower: np.ndarray | float | None = None
    amplitude_upper: np.ndarray | float | None = None
    max_delta: np.ndarray | float | None = None

    @classmethod
    def from_slew_rate(cls, dt, max_slew_rate, amplitude_lower=None, amplitude_upper=None):
        return cls(
            amplitude_lower=amplitude_lower,
            amplitude_upper=amplitude_upper,
            max_delta=np.asarray(max_slew_rate, dtype=float) * dt,
        )

    def is_feasible(self, amplitudes, atol=1e-12):
        amplitudes = np.asarray(amplitudes, dtype=float)
        if self.amplitude_lower is not None:
            lower = np.broadcast_to(np.asarray(self.amplitude_lower, dtype=float), amplitudes.shape)
            if np.any(amplitudes < lower - atol):
                return False
        if self.amplitude_upper is not None:
            upper = np.broadcast_to(np.asarray(self.amplitude_upper, dtype=float), amplitudes.shape)
            if np.any(amplitudes > upper + atol):
                return False
        if self.max_delta is not None and amplitudes.shape[0] > 1:
            max_delta = np.broadcast_to(
                np.asarray(self.max_delta, dtype=float),
                amplitudes[1:].shape,
            )
            if np.any(np.abs(np.diff(amplitudes, axis=0)) > max_delta + atol):
                return False
        return True

    def violation(self, amplitudes):
        amplitudes = np.asarray(amplitudes, dtype=float)
        violations = []
        if self.amplitude_lower is not None:
            lower = np.broadcast_to(np.asarray(self.amplitude_lower, dtype=float), amplitudes.shape)
            violations.append(np.maximum(lower - amplitudes, 0.0))
        if self.amplitude_upper is not None:
            upper = np.broadcast_to(np.asarray(self.amplitude_upper, dtype=float), amplitudes.shape)
            violations.append(np.maximum(amplitudes - upper, 0.0))
        if self.max_delta is not None and amplitudes.shape[0] > 1:
            delta = np.diff(amplitudes, axis=0)
            max_delta = np.broadcast_to(np.asarray(self.max_delta, dtype=float), delta.shape)
            violations.append(np.maximum(np.abs(delta) - max_delta, 0.0))
        if not violations:
            return 0.0
        return float(sum(np.sum(item**2) for item in violations))

    def penalty(self, amplitudes, weight=1.0):
        return weight * self.violation(amplitudes)

    def penalty_gradient(self, amplitudes, weight=1.0):
        amplitudes = np.asarray(amplitudes, dtype=float)
        gradient = np.zeros_like(amplitudes)
        if self.amplitude_lower is not None:
            lower = np.broadcast_to(np.asarray(self.amplitude_lower, dtype=float), amplitudes.shape)
            below = amplitudes < lower
            gradient[below] += -2.0 * (lower[below] - amplitudes[below])
        if self.amplitude_upper is not None:
            upper = np.broadcast_to(np.asarray(self.amplitude_upper, dtype=float), amplitudes.shape)
            above = amplitudes > upper
            gradient[above] += 2.0 * (amplitudes[above] - upper[above])
        if self.max_delta is not None and amplitudes.shape[0] > 1:
            delta = np.diff(amplitudes, axis=0)
            max_delta = np.broadcast_to(np.asarray(self.max_delta, dtype=float), delta.shape)
            excess = np.maximum(np.abs(delta) - max_delta, 0.0)
            delta_gradient = 2.0 * excess * np.sign(delta)
            gradient[:-1] -= delta_gradient
            gradient[1:] += delta_gradient
        return weight * gradient

    def scipy_constraints(self, parameterization):
        constraints = []
        if self.max_delta is None:
            return constraints

        max_delta = np.asarray(self.max_delta, dtype=float)

        def positive_delta(parameters):
            amplitudes = parameterization.to_physical(parameters)
            return np.broadcast_to(max_delta, amplitudes[1:].shape) - np.diff(amplitudes, axis=0)

        def negative_delta(parameters):
            amplitudes = parameterization.to_physical(parameters)
            return np.broadcast_to(max_delta, amplitudes[1:].shape) + np.diff(amplitudes, axis=0)

        constraints.extend(
            [
                {"type": "ineq", "fun": lambda parameters: positive_delta(parameters).reshape(-1)},
                {"type": "ineq", "fun": lambda parameters: negative_delta(parameters).reshape(-1)},
            ]
        )
        return constraints
