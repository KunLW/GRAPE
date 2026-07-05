from __future__ import annotations

import numpy as np

from quantum_control.evolution.base import Evolution
from quantum_control.results.expansion_result import ExpansionResult, ExpansionState


class PerturbativeExpansionEvolution(Evolution):
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
        forward = self._forward_states(steps, context.initial_state)
        backward = None
        if context.compute_backward and context.target_state is not None:
            backward = self._backward_states(steps, context.target_state)
        return ExpansionResult(
            steps=steps,
            forward=forward,
            backward=backward,
            max_order=self.max_order,
            metadata={"dt": pulse.dt},
        )

    def _forward_states(self, steps, initial_state, seed_components=None):
        states = [ExpansionState({0: np.asarray(initial_state, dtype=complex)})]
        for order in range(1, self.max_order + 1):
            states[0].components[order] = np.zeros_like(states[0].components[0])
        if seed_components:
            for order, state in seed_components.items():
                states[0].components[order] = np.asarray(state, dtype=complex)

        for step in steps:
            previous = states[-1].components
            components = {}
            for order in range(self.max_order + 1):
                propagated = step.W @ previous[order]
                if order > 0:
                    propagated = propagated + step.V @ previous[order - 1]
                components[order] = propagated
            states.append(ExpansionState(components))
        return states

    def _backward_states(self, steps, target_state):
        states_by_index = [None] * (len(steps) + 1)
        final_components = {0: np.asarray(target_state, dtype=complex)}
        for order in range(1, self.max_order + 1):
            final_components[order] = np.zeros_like(final_components[0])
        states_by_index[-1] = ExpansionState(final_components)

        for step_index in range(len(steps) - 1, -1, -1):
            step = steps[step_index]
            next_components = states_by_index[step_index + 1].components
            components = {}
            for order in range(self.max_order + 1):
                propagated = step.W.conj().T @ next_components[order]
                if order > 0:
                    propagated = propagated + step.V.conj().T @ next_components[order - 1]
                components[order] = propagated
            states_by_index[step_index] = ExpansionState(components)
        return states_by_index
