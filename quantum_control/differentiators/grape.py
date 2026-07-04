from __future__ import annotations

import numpy as np

from quantum_control.steps.unitary_step import UnitaryStep


class GrapeDifferentiator:
    def __init__(self, step_builder):
        self.step_builder = step_builder

    def gradient(self, system, pulse, context, result):
        if not hasattr(result, "W_steps"):
            raise ValueError("GrapeDifferentiator requires a nominal unitary result.")
        if context.target_state is None:
            raise ValueError("GrapeDifferentiator requires context.target_state.")

        initial_state = np.asarray(context.initial_state, dtype=complex)
        target_state = np.asarray(context.target_state, dtype=complex)
        forward_states = self._forward_states(result.W_steps, initial_state)
        backward_states = self._backward_states(result.W_steps, target_state)
        final_amplitude = np.vdot(target_state, forward_states[-1])
        gradient = np.zeros_like(pulse.amplitudes)

        for step_index, step_w in enumerate(result.W_steps):
            step = UnitaryStep(W=step_w)
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
                derivative_state = derivative_step.W @ forward_states[step_index]
                derivative_amplitude = np.vdot(
                    backward_states[step_index + 1],
                    derivative_state,
                )
                derivative_value = 2.0 * np.real(
                    np.conj(final_amplitude) * derivative_amplitude
                )
                gradient[step_index, control_index] = derivative_value

        return gradient

    @staticmethod
    def _forward_states(steps, initial_state):
        """Return states[k] = W_k ... W_1 |initial_state>."""
        states = [initial_state]
        for step_w in steps:
            states.append(step_w @ states[-1])
        return states

    @staticmethod
    def _backward_states(steps, target_state):
        """Return adjoint-column backward states.

        The mathematical backward object is the bra
        <B_k| = <target_state| W_N ... W_{k+1}.  For NumPy contractions we
        store its adjoint column, |bar B_k> = W_{k+1}^dag ... W_N^dag
        |target_state>, so np.vdot(states[k], x) evaluates <B_k|x>.
        """
        states = [None] * (len(steps) + 1)
        states[-1] = target_state
        for step_index in reversed(range(len(steps))):
            states[step_index] = steps[step_index].conj().T @ states[step_index + 1]
        return states
