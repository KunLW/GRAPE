from __future__ import annotations

import numpy as np

from quantum_control.differentiators.base import Differentiator


class FiniteDifferenceDifferentiator(Differentiator):
    def __init__(self, evolution, objective, epsilon=1e-6):
        self.evolution = evolution
        self.objective = objective
        self.epsilon = epsilon

    def gradient(self, system, pulse, context, result=None):
        gradient = np.zeros_like(pulse.amplitudes)
        for step_index in range(pulse.n_steps):
            for control_index in range(pulse.n_controls):
                plus = np.array(pulse.amplitudes, copy=True)
                minus = np.array(pulse.amplitudes, copy=True)
                plus[step_index, control_index] += self.epsilon
                minus[step_index, control_index] -= self.epsilon
                plus_value = self.objective.evaluate(
                    self.evolution.evolve(system, pulse.with_amplitudes(plus), context)
                )
                minus_value = self.objective.evaluate(
                    self.evolution.evolve(system, pulse.with_amplitudes(minus), context)
                )
                gradient[step_index, control_index] = (
                    plus_value - minus_value
                ) / (2.0 * self.epsilon)
        return gradient
