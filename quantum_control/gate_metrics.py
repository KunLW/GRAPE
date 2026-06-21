from __future__ import annotations

import numpy as np

from quantum_control.context import EvolutionContext
from quantum_control.evolution.expansion_evolution import PerturbativeExpansionEvolution
from quantum_control.evolution.nominal_evolution import NominalUnitaryEvolution
from quantum_control.objectives.expansion_fidelity import ExpansionFidelity
from quantum_control.objectives.state_fidelity import StateTransferFidelity
from quantum_control.state_average import ExpansionStateAverageFidelity, StatePair
from quantum_control.steps.perturbative_step import PerturbativeStepBuilder
from quantum_control.steps.unitary_step import UnitaryStepBuilder


def single_qubit_logical_test_states():
    zero = np.array([1.0, 0.0], dtype=complex)
    one = np.array([0.0, 1.0], dtype=complex)
    plus = (zero + one) / np.sqrt(2.0)
    plus_i = (zero + 1j * one) / np.sqrt(2.0)
    return (zero, one, plus, plus_i)


def two_qubit_logical_test_states():
    states = single_qubit_logical_test_states()
    return tuple(np.kron(left, right) for left in states for right in states)


def ms_xx_pi_over_2_gate():
    sx = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    xx = np.kron(sx, sx)
    return (np.eye(4, dtype=complex) - 1j * xx) / np.sqrt(2.0)


def motion_resolved_gate_state_pairs(target_gate, n_levels):
    if n_levels < 1:
        raise ValueError("n_levels must be at least 1.")
    target_gate = np.asarray(target_gate, dtype=complex)
    if target_gate.shape != (4, 4):
        raise ValueError("target_gate must have shape (4, 4).")

    motion_ground = _motion_basis(0, n_levels)
    weight = 1.0 / len(two_qubit_logical_test_states())
    pairs = []
    for spin_state in two_qubit_logical_test_states():
        initial_state = np.kron(spin_state, motion_ground)
        target_spin = target_gate @ spin_state
        for motion_index in range(n_levels):
            target_state = np.kron(target_spin, _motion_basis(motion_index, n_levels))
            pairs.append(StatePair(initial_state, target_state, weight))
    return tuple(pairs)


def closed_gate_fidelity(system, pulse, target_gate, n_levels):
    evolution = NominalUnitaryEvolution(UnitaryStepBuilder())
    objective_values = []
    for pair in motion_resolved_gate_state_pairs(target_gate, n_levels):
        context = EvolutionContext(
            initial_state=pair.initial_state,
            target_state=pair.target_state,
        )
        result = evolution.evolve(system, pulse, context)
        objective_values.append(
            pair.weight * StateTransferFidelity(pair.target_state).evaluate(result)
        )
    return float(np.sum(objective_values))


def open_gate_fidelity(system, pulse, target_gate, n_levels):
    step_builder = PerturbativeStepBuilder()
    objective = ExpansionFidelity(max_order=2, drop_odd_average=True)
    averaged = ExpansionStateAverageFidelity(
        system=system,
        pulse=pulse,
        evolution=PerturbativeExpansionEvolution(step_builder, max_order=2),
        objective=objective,
        differentiator=None,
        state_pairs=motion_resolved_gate_state_pairs(target_gate, n_levels),
        normalize_weights=False,
    )
    return averaged.value()


def _motion_basis(index, n_levels):
    state = np.zeros(n_levels, dtype=complex)
    state[index] = 1.0
    return state
