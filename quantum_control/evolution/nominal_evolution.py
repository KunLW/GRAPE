from __future__ import annotations

import numpy as np

from quantum_control.evolution.base import Evolution
from quantum_control.results.nominal_result import NominalUnitaryResult


class NominalUnitaryEvolution(Evolution):
    def __init__(self, step_builder):
        self.step_builder = step_builder

    def evolve(self, system, pulse, context):
        steps = []
        dimension = context.initial_state.shape[0]
        total = np.eye(dimension, dtype=complex)
        for step_index in range(pulse.n_steps):
            step = self.step_builder.build_step(
                system,
                pulse.controls_at(step_index),
                pulse.dt,
                t=step_index * pulse.dt,
            )
            steps.append(step.W)
            total = step.W @ total
        return NominalUnitaryResult(
            W_steps=steps,
            U_total=total,
            metadata={
                "initial_state": context.initial_state,
                "target_state": context.target_state,
            },
        )
