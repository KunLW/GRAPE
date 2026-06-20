from __future__ import annotations

import numpy as np


class AdamOptimizer:
    def __init__(self, learning_rate=1e-2, beta1=0.9, beta2=0.999, epsilon=1e-8, steps=100):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.steps = steps

    def optimize(self, problem, initial_pulse=None):
        pulse = initial_pulse or problem.pulse
        amplitudes = np.array(pulse.amplitudes, copy=True)
        first_moment = np.zeros_like(amplitudes)
        second_moment = np.zeros_like(amplitudes)
        history = []

        for iteration in range(1, self.steps + 1):
            trial_pulse = pulse.with_amplitudes(amplitudes)
            value = problem.value(trial_pulse)
            gradient = problem.gradient(trial_pulse)
            first_moment = self.beta1 * first_moment + (1.0 - self.beta1) * gradient
            second_moment = self.beta2 * second_moment + (1.0 - self.beta2) * gradient**2
            m_hat = first_moment / (1.0 - self.beta1**iteration)
            v_hat = second_moment / (1.0 - self.beta2**iteration)
            amplitudes = amplitudes + self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
            history.append(value)

        return {
            "optimized_pulse": pulse.with_amplitudes(amplitudes),
            "history": history,
        }

    def optimize_parameters(self, problem, initial_parameters=None):
        parameters = (
            problem.initial_parameters().reshape(-1)
            if initial_parameters is None
            else np.asarray(initial_parameters, dtype=float).reshape(-1)
        )
        first_moment = np.zeros_like(parameters)
        second_moment = np.zeros_like(parameters)
        history = []

        for iteration in range(1, self.steps + 1):
            value = problem.value(parameters)
            gradient = problem.gradient(parameters).reshape(-1)
            first_moment = self.beta1 * first_moment + (1.0 - self.beta1) * gradient
            second_moment = self.beta2 * second_moment + (1.0 - self.beta2) * gradient**2
            m_hat = first_moment / (1.0 - self.beta1**iteration)
            v_hat = second_moment / (1.0 - self.beta2**iteration)
            parameters = parameters + self.learning_rate * m_hat / (
                np.sqrt(v_hat) + self.epsilon
            )
            history.append(value)

        return {
            "optimized_parameters": parameters,
            "optimized_pulse": problem.pulse_from_parameters(parameters),
            "history": history,
        }
