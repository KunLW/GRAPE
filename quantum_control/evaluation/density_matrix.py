"""Faithful open-system evaluation: exact Lindblad + Gauss-Hermite averaging.

Unlike the perturbative metrics in ``gate_metrics``, this module propagates
the full density matrix under the exact Lindblad master equation and averages
the coherent quasi-static fluctuations by Gauss-Hermite quadrature over their
Gaussian distribution. It is therefore valid beyond small expansion
parameters and serves as the independent check on
``noisy_gate_fidelity`` (second-order expansion + first-order Lindblad
correction).

Cost: one matrix exponential of the d^2 x d^2 Liouvillian per pulse step and
per quadrature node, with ``hermite_points ** n_fluctuation_terms`` nodes.
This is an *evaluation* tool for deliberate, occasional use on modest Hilbert
spaces — never part of the optimization loop.
"""

from __future__ import annotations

import itertools

import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.linalg import expm


def _dissipator_superoperator(collapse_operators, dimension):
    """Vectorized Lindblad dissipator, row-major ``vec(rho)`` convention."""
    identity = np.eye(dimension, dtype=complex)
    dissipator = np.zeros((dimension**2, dimension**2), dtype=complex)
    for operator in collapse_operators:
        operator = np.asarray(operator, dtype=complex)
        product = operator.conj().T @ operator
        dissipator = dissipator + (
            np.kron(operator, operator.conj())
            - 0.5 * np.kron(product, identity)
            - 0.5 * np.kron(identity, product.T)
        )
    return dissipator


def _node_superpropagator(system, pulse, xi, fluctuation_terms, dissipator):
    """Total superpropagator for one draw ``xi`` of the fluctuation variables.

    Per step the coherent Hamiltonian is the nominal one plus
    ``sum_j xi_j * matrix_j`` for static terms and
    ``sum_i xi_i * control_i(t) * matrix_i`` for control terms — the same
    convention as ``OpenSystem.fluctuation_hamiltonian`` with unit variables
    replaced by the quadrature node.
    """
    dimension = np.asarray(system.drift).shape[0]
    identity = np.eye(dimension, dtype=complex)
    static_parts = [
        (value, term.matrix)
        for value, term in zip(xi, fluctuation_terms)
        if term.kind == "static"
    ]
    control_parts = [
        (value, term.matrix)
        for value, term in zip(xi, fluctuation_terms)
        if term.kind == "control"
    ]
    total = np.eye(dimension**2, dtype=complex)
    for step_index in range(pulse.n_steps):
        controls = pulse.controls_at(step_index)
        hamiltonian = np.asarray(system.nominal_hamiltonian(controls), dtype=complex)
        for value, matrix in static_parts:
            hamiltonian = hamiltonian + value * matrix
        for (value, matrix), amplitude in zip(control_parts, controls):
            hamiltonian = hamiltonian + value * amplitude * matrix
        liouvillian = (
            -1j * (np.kron(hamiltonian, identity) - np.kron(identity, hamiltonian.T))
            + dissipator
        )
        total = expm(pulse.dt * liouvillian) @ total
    return total


def faithful_gate_fidelity(system, pulse, state_pairs, hermite_points=5):
    """Exactly averaged open-system fidelity over weighted ``state_pairs``.

    The noise model is read off the system itself: quasi-static coherent
    fluctuations from ``system.fluctuation_terms`` (Gauss-Hermite averaged
    over their independent standard-normal variables) and Markovian
    decoherence from ``system.collapse_operators`` (exact Lindblad
    propagation). A plain ``ClosedSystem`` degenerates to nominal unitary
    propagation.

    Same convention as the perturbative metrics: returns
    ``sum_pairs weight * <target| rho_final |target>`` with the weights used
    as given, so the value is directly comparable to
    ``closed_gate_fidelity`` / ``noisy_gate_fidelity``.
    """
    state_pairs = tuple(state_pairs)
    fluctuation_terms = tuple(getattr(system, "fluctuation_terms", ()))
    collapse_operators = tuple(getattr(system, "collapse_operators", ()))
    dimension = np.asarray(system.drift).shape[0]
    dissipator = _dissipator_superoperator(collapse_operators, dimension)

    # Gauss-Hermite nodes/weights for each independent standard normal:
    # integral N(0,1) f = sum_k (w_k / sqrt(pi)) f(sqrt(2) x_k).
    if fluctuation_terms:
        nodes, weights = hermgauss(int(hermite_points))
        nodes = np.sqrt(2.0) * nodes
        weights = weights / np.sqrt(np.pi)
        grid = itertools.product(range(len(nodes)), repeat=len(fluctuation_terms))
    else:
        nodes, weights = np.zeros(1), np.ones(1)
        grid = [()]

    initial_vecs = np.column_stack(
        [np.outer(pair.initial_state, pair.initial_state.conj()).reshape(-1) for pair in state_pairs]
    )
    pair_weights = np.array([pair.weight for pair in state_pairs], dtype=float)
    targets = [np.asarray(pair.target_state, dtype=complex) for pair in state_pairs]

    fidelity = 0.0
    for index_combo in grid:
        xi = np.array([nodes[k] for k in index_combo], dtype=float)
        node_weight = float(np.prod([weights[k] for k in index_combo])) if index_combo else 1.0
        total = _node_superpropagator(system, pulse, xi, fluctuation_terms, dissipator)
        final_vecs = total @ initial_vecs
        pair_fidelities = np.array(
            [
                float(
                    np.real(
                        target.conj()
                        @ final_vecs[:, column].reshape(dimension, dimension)
                        @ target
                    )
                )
                for column, target in enumerate(targets)
            ]
        )
        fidelity += node_weight * float(pair_weights @ pair_fidelities)
    return fidelity
