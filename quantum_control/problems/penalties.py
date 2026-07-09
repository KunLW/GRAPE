from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ParameterSmoothPenalty:
    l1_weight: float = 0.0
    l2_weight: float = 0.0

    def value(self, parameters, parameter_shape):
        return self.l1_value(parameters, parameter_shape) + self.l2_value(
            parameters,
            parameter_shape,
        )

    def l1_value(self, parameters, parameter_shape):
        parameters = self._reshape(parameters, parameter_shape)
        if parameters.shape[0] < 2 or not self.l1_weight:
            return 0.0
        first_difference = np.diff(parameters, axis=0)
        return float(self.l1_weight * np.sum(np.abs(first_difference)))

    def l2_value(self, parameters, parameter_shape):
        parameters = self._reshape(parameters, parameter_shape)
        if parameters.shape[0] < 3 or not self.l2_weight:
            return 0.0
        second_difference = np.diff(parameters, n=2, axis=0)
        return float(self.l2_weight * np.sum(second_difference**2))

    def gradient(self, parameters, parameter_shape):
        parameters = self._reshape(parameters, parameter_shape)
        gradient = np.zeros_like(parameters)

        if parameters.shape[0] >= 2 and self.l1_weight:
            first_difference = np.diff(parameters, axis=0)
            first_gradient = self.l1_weight * np.sign(first_difference)
            gradient[:-1] -= first_gradient
            gradient[1:] += first_gradient

        if parameters.shape[0] >= 3 and self.l2_weight:
            second_difference = np.diff(parameters, n=2, axis=0)
            second_gradient = 2.0 * self.l2_weight * second_difference
            gradient[:-2] += second_gradient
            gradient[1:-1] -= 2.0 * second_gradient
            gradient[2:] += second_gradient

        return gradient.reshape(np.shape(parameters))

    @staticmethod
    def _reshape(parameters, parameter_shape):
        return np.asarray(parameters, dtype=float).reshape(parameter_shape)


class PenalizedParameterizedProblem:
    """Parameter-space problem minus smoothness penalties.

    Convention: problems are fidelities to be *maximized*, so penalties are
    always subtracted here; the optimizer's ``maximize`` flag only flips the
    sign handed to scipy. ``raw_value`` exposes the un-penalized fidelity for
    reporting (there is deliberately no ``raw_gradient`` — nothing consumes
    it).
    """

    def __init__(self, problem, penalty, parameter_shape=None):
        self.problem = problem
        self.penalty = penalty
        self.parameter_shape = (
            tuple(problem.initial_parameters().shape)
            if parameter_shape is None
            else tuple(parameter_shape)
        )

    def initial_parameters(self, pulse=None):
        return self.problem.initial_parameters(pulse)

    def pulse_from_parameters(self, parameters):
        return self.problem.pulse_from_parameters(self._reshape(parameters))

    def parameter_bounds(self):
        return self.problem.parameter_bounds()

    def value(self, parameters):
        parameters = self._reshape(parameters)
        return self.raw_value(parameters) - self.penalty.value(parameters, self.parameter_shape)

    def gradient(self, parameters):
        parameters = self._reshape(parameters)
        return self.problem.gradient(parameters) - self.penalty.gradient(
            parameters,
            self.parameter_shape,
        )

    def value_and_gradient(self, parameters):
        """Both at once; the wrapped problem evolves only once."""
        parameters = self._reshape(parameters)
        value, gradient = self.problem.value_and_gradient(parameters)
        return (
            value - self.penalty.value(parameters, self.parameter_shape),
            gradient - self.penalty.gradient(parameters, self.parameter_shape),
        )

    def raw_value(self, parameters):
        return self.problem.value(self._reshape(parameters))

    def _reshape(self, parameters):
        return np.asarray(parameters, dtype=float).reshape(self.parameter_shape)
