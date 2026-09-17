from __future__ import annotations

import numpy as np

from quantum_control.evolution.base import Evolution
from quantum_control.results.expansion_result import ExpansionResult, ExpansionState


class PerturbativeExpansionEvolution(Evolution):
    """Propagate independent quasi-static channels through second order.

    Higher-order states carry a leading noise-channel axis when there are
    multiple sources. Order two contains only same-channel double insertions;
    mixed insertions average to zero. Nominal propagation is shared.
    """

    def __init__(self, step_builder, max_order=2):
        if max_order < 0:
            raise ValueError("max_order must be non-negative.")
        self.step_builder = step_builder
        self.max_order = max_order

    def evolve(self, system, pulse, context):
        steps = [
            self.step_builder.build_step(
                system,
                pulse.controls_at(step_index),
                pulse.dt,
                t=step_index * pulse.dt,
            )
            for step_index in range(pulse.n_steps)
        ]
        if steps and steps[0].V.ndim == 3 and self.max_order > 2:
            raise ValueError("Independent multi-channel expansion supports max_order <= 2.")
        forward = self._forward_states(steps, context.initial_state)
        backward = None
        if context.compute_backward and context.target_state is not None:
            backward = self._backward_states(steps, context.target_state)
        return ExpansionResult(
            steps=steps,
            forward=forward,
            backward=backward,
            max_order=self.max_order,
            metadata={
                "dt": pulse.dt,
                "target_state": context.target_state,
                "fluctuation_average": "independent_second_order",
            },
        )

    def _forward_states(self, steps, initial_state, seed_components=None):
        states = [ExpansionState({0: np.asarray(initial_state, dtype=complex)})]
        for order in range(1, self.max_order + 1):
            states[0].components[order] = _zero_component(steps, initial_state)
        if seed_components:
            for order, state in seed_components.items():
                states[0].components[order] = np.asarray(state, dtype=complex)

        for step in steps:
            previous = states[-1].components
            components = {}
            for order in range(self.max_order + 1):
                propagated = apply_operator(step.W, previous[order])
                if order > 0:
                    propagated = propagated + apply_operator(step.V, previous[order - 1])
                components[order] = propagated
            states.append(ExpansionState(components))
        return states

    def _backward_states(self, steps, target_state):
        states_by_index = [None] * (len(steps) + 1)
        final_components = {0: np.asarray(target_state, dtype=complex)}
        for order in range(1, self.max_order + 1):
            final_components[order] = _zero_component(steps, target_state)
        states_by_index[-1] = ExpansionState(final_components)

        for step_index in range(len(steps) - 1, -1, -1):
            step = steps[step_index]
            next_components = states_by_index[step_index + 1].components
            components = {}
            for order in range(self.max_order + 1):
                propagated = apply_operator(step.W.conj().T, next_components[order])
                if order > 0:
                    propagated = propagated + apply_operator(
                        step.V.conj().swapaxes(-1, -2), next_components[order - 1]
                    )
                components[order] = propagated
            states_by_index[step_index] = ExpansionState(components)
        return states_by_index


def apply_operator(operator, state):
    """Matrix-vector product, preserving an optional independent-channel axis."""
    return np.einsum("...ij,...j->...i", operator, state)


def _zero_component(steps, state):
    shape = np.shape(state)
    if steps and steps[0].V.ndim == 3:
        shape = (steps[0].V.shape[0],) + shape
    return np.zeros(shape, dtype=complex)
