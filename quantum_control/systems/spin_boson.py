from __future__ import annotations

import numpy as np

from quantum_control.pulses.parameterization import BoundedAmplitudeParameterization
from quantum_control.pulses.pulse import PiecewiseConstantPulse
from quantum_control.systems.closed_system import FluctuatingClosedSystem


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


def spin_boson_control_system(
    n_levels,
    phi_s,
    static_fluctuations=(),
    control_fluctuations=(),
):
    """Build the two-channel spin-boson Hamiltonian with optional fluctuations.

    The nominal Hamiltonian is
    ``alpha_1 I_spin ⊗ a†a + alpha_2 S_phi ⊗ (a + a†)``. Fluctuation terms use
    the same long-correlation convention as the perturbative expansion path:
    static matrices are already-scaled ``sigma_xi H_xi`` terms and control
    fluctuation matrices are already-scaled ``sigma_chi_i H_chi_i`` terms.
    """

    spin_identity = np.eye(2, dtype=complex)
    a = annihilation_operator(n_levels)
    adag = a.conj().T
    dimension = 2 * n_levels

    return FluctuatingClosedSystem(
        drift=np.zeros((dimension, dimension), dtype=complex),
        controls=[
            np.kron(spin_identity, adag @ a),
            np.kron(spin_phase_operator(phi_s), a + adag),
        ],
        static_fluctuations=static_fluctuations,
        control_fluctuations=control_fluctuations,
    )


def spin_boson_initial_pulse(
    n_steps=200,
    total_time_us=225.8,
    alpha1_khz_bounds=(1.0, 600.0),
    alpha2_khz_bounds=(0.0, 20.0),
):
    if n_steps < 1:
        raise ValueError("n_steps must be at least 1.")
    total_time = float(total_time_us) * 1e-6
    if total_time <= 0.0:
        raise ValueError("total_time_us must be positive.")

    alpha1_lower, alpha1_upper = _khz_bounds_to_rad_s(alpha1_khz_bounds)
    alpha2_lower, alpha2_upper = _khz_bounds_to_rad_s(alpha2_khz_bounds)
    dt = total_time / n_steps
    phase = np.pi * (np.arange(n_steps, dtype=float) + 0.5) / n_steps

    alpha1_center = 0.5 * (alpha1_upper + alpha1_lower)
    alpha1_scale = 0.5 * (alpha1_upper - alpha1_lower)
    alpha1 = alpha1_center + alpha1_scale * np.cos(phase)
    alpha2 = alpha2_lower + (alpha2_upper - alpha2_lower) * np.sin(phase)

    return PiecewiseConstantPulse(
        amplitudes=np.column_stack([alpha1, alpha2]),
        dt=dt,
    )


def spin_boson_parameterization(
    n_steps=200,
    alpha1_khz_bounds=(1.0, 600.0),
    alpha2_khz_bounds=(0.0, 20.0),
):
    if n_steps < 1:
        raise ValueError("n_steps must be at least 1.")
    alpha1_lower, alpha1_upper = _khz_bounds_to_rad_s(alpha1_khz_bounds)
    alpha2_lower, alpha2_upper = _khz_bounds_to_rad_s(alpha2_khz_bounds)
    return BoundedAmplitudeParameterization(
        lower=np.array([alpha1_lower, alpha2_lower], dtype=float),
        upper=np.array([alpha1_upper, alpha2_upper], dtype=float),
    )


def _khz_bounds_to_rad_s(bounds):
    lower, upper = np.asarray(bounds, dtype=float)
    if upper <= lower:
        raise ValueError("upper bounds must be greater than lower bounds.")
    return 2.0 * np.pi * 1000.0 * lower, 2.0 * np.pi * 1000.0 * upper
