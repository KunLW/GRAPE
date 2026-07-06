from __future__ import annotations

import numpy as np

from quantum_control.context import EvolutionContext
from quantum_control.evolution.expansion_evolution import PerturbativeExpansionEvolution
from quantum_control.evolution.nominal_evolution import NominalUnitaryEvolution
from quantum_control.objectives.expansion_fidelity import ExpansionFidelity
from quantum_control.objectives.state_fidelity import StateTransferFidelity
from quantum_control.state_average import ExpansionStateAverageFidelity
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


def closed_gate_fidelity(system, pulse, state_pairs):
    """Average closed-system transfer fidelity over weighted ``state_pairs``.

    ``state_pairs`` is any iterable of ``StatePair`` (e.g. from a system
    definition's ``state_pairs()``); this function is agnostic to the
    physical structure behind them.
    """
    evolution = NominalUnitaryEvolution(UnitaryStepBuilder())
    objective_values = []
    for pair in state_pairs:
        context = EvolutionContext(
            initial_state=pair.initial_state,
            target_state=pair.target_state,
        )
        result = evolution.evolve(system, pulse, context)
        objective_values.append(
            pair.weight * StateTransferFidelity(pair.target_state).evaluate(result)
        )
    return float(np.sum(objective_values))


def open_gate_fidelity(system, pulse, state_pairs, n_workers=1):
    """Second-order expansion fidelity averaged over weighted ``state_pairs``."""
    step_builder = PerturbativeStepBuilder()
    objective = ExpansionFidelity(max_order=2, drop_odd_average=True)
    averaged = ExpansionStateAverageFidelity(
        system=system,
        pulse=pulse,
        evolution=PerturbativeExpansionEvolution(step_builder, max_order=2),
        objective=objective,
        differentiator=None,
        state_pairs=tuple(state_pairs),
        normalize_weights=False,
        n_workers=n_workers,
    )
    try:
        return averaged.value()
    finally:
        averaged.shutdown()
