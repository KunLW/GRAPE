from __future__ import annotations

import numpy as np

from quantum_control.differentiators.base import Differentiator


class PerturbativeExpansionDifferentiator(Differentiator):
    def __init__(self, step_builder, objective):
        self.step_builder = step_builder
        self.objective = objective

    def gradient(self, system, pulse, context, result):
        if result.backward is None:
            raise ValueError("Backward states are required for expansion gradients.")

        amplitudes = self.objective.amplitudes(result)
        gradient = np.zeros_like(pulse.amplitudes)

        for step_index, step in enumerate(result.steps):
            previous_forward = result.forward[step_index].components
            next_backward = result.backward[step_index + 1].components
            controls = pulse.controls_at(step_index)
            t = step_index * pulse.dt

            for control_index in range(pulse.n_controls):
                derivative_step = self.step_builder.derivative_step(
                    system,
                    controls,
                    pulse.dt,
                    control_index,
                    step,
                    t=t,
                )
                local_derivatives = self._local_component_derivatives(
                    derivative_step,
                    previous_forward,
                    result.max_order,
                )
                derivative_amplitudes = self._derivative_amplitudes(
                    local_derivatives,
                    next_backward,
                    result.max_order,
                )
                derivative_value = self.objective.contract(
                    amplitudes,
                    derivative_amplitudes=derivative_amplitudes,
                )
                gradient[step_index, control_index] = np.real_if_close(derivative_value).real

        return gradient

    @staticmethod
    def _local_component_derivatives(derivative_step, previous_forward, max_order):
        derivatives = {}
        for order in range(max_order + 1):
            value = derivative_step.W @ previous_forward[order]
            if order > 0:
                value = value + derivative_step.V @ previous_forward[order - 1]
            derivatives[order] = value
        return derivatives

    @staticmethod
    def _derivative_amplitudes(local_derivatives, next_backward, max_order):
        derivative_amplitudes = {}
        for final_order in range(max_order + 1):
            amplitude = 0.0 + 0.0j
            for local_order in range(final_order + 1):
                future_order = final_order - local_order
                amplitude = amplitude + np.vdot(
                    next_backward[future_order],
                    local_derivatives[local_order],
                )
            derivative_amplitudes[final_order] = amplitude
        return derivative_amplitudes
