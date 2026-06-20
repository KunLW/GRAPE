from __future__ import annotations

import numpy as np

from quantum_control.systems.closed_system import ClosedSystem


def annihilation_operator(n_levels):
    if n_levels < 1:
        raise ValueError("n_levels must be at least 1.")
    operator = np.zeros((n_levels, n_levels), dtype=complex)
    for level in range(1, n_levels):
        operator[level - 1, level] = np.sqrt(level)
    return operator


def creation_operator(n_levels):
    return annihilation_operator(n_levels).conj().T


def number_operator(n_levels):
    a = annihilation_operator(n_levels)
    return creation_operator(n_levels) @ a


def spin_phase_operator(phi_s):
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    return np.cos(phi_s) * sx + np.sin(phi_s) * sy


def spin_boson_control_system(n_levels, phi_s):
    spin_identity = np.eye(2, dtype=complex)
    a = annihilation_operator(n_levels)
    adag = a.conj().T
    dimension = 2 * n_levels

    return ClosedSystem(
        drift=np.zeros((dimension, dimension), dtype=complex),
        controls=[
            np.kron(spin_identity, adag @ a),
            np.kron(spin_phase_operator(phi_s), a + adag),
        ],
    )
