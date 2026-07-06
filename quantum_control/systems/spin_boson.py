from __future__ import annotations

import numpy as np

from quantum_control.gate_metrics import two_qubit_logical_test_states
from quantum_control.pulses.parameterization import BoundedAmplitudeParameterization
from quantum_control.pulses.pulse import PiecewiseConstantPulse
from quantum_control.state_average import StatePair
from quantum_control.systems.closed_system import FluctuatingClosedSystem
from quantum_control.systems.open_system import LindbladOpenSystem
from quantum_control.units import khz_bounds_to_rad_s

DEFAULT_LAMB_DICKE_ETA = 0.075
DEFAULT_ALPHA1_KHZ_BOUNDS = (1.0, 60.0)
DEFAULT_ALPHA2_KHZ_BOUNDS = (0.0, 200.0)


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


def two_qubit_spin_phase_mode(phi_s, mode_vector):
    mode_vector = np.asarray(mode_vector, dtype=float)
    if mode_vector.shape != (2,):
        raise ValueError("mode_vector must contain exactly two ion weights.")
    single_spin = spin_phase_operator(phi_s)
    identity = np.eye(2, dtype=complex)
    return (
        mode_vector[0] * np.kron(single_spin, identity)
        + mode_vector[1] * np.kron(identity, single_spin)
    )


def two_qubit_spin_phase_difference(phi_s):
    return two_qubit_spin_phase_mode(phi_s, (0.5, -0.5))


def motion_resolved_gate_state_pairs(target_gate, n_levels):
    """Expand a 4x4 spin gate into weighted spin ⊗ motion ``StatePair``s.

    Initial states are the 16 two-qubit logical test states with the motion
    in its ground state; targets are resolved over all ``n_levels`` motional
    levels, so the average measures the gate fidelity on the spins while
    tracking where motional population ends up.
    """
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


def _motion_basis(index, n_levels):
    state = np.zeros(n_levels, dtype=complex)
    state[index] = 1.0
    return state


def spin_boson_collapse_operators(
    n_levels,
    gamma_heating=0.0,
    gamma_motional_dephasing=0.0,
    gamma_spin_dephasing=0.0,
):
    r"""Build scaled jump operators $L = \sqrt{\gamma}\,A$ for the spin-boson system.

    Heating uses $A = I_{\mathrm{spin}} \otimes a^\dagger$, motional dephasing
    $A = I_{\mathrm{spin}} \otimes a^\dagger a$, and collective spin dephasing
    $A = \tfrac{1}{2}(\sigma_z \otimes I + I \otimes \sigma_z) \otimes I_{\mathrm{motion}}$.
    Rates are angular frequencies (rad/s); zero-rate channels are omitted.
    """

    spin_identity = np.eye(4, dtype=complex)
    single_identity = np.eye(2, dtype=complex)
    motion_identity = np.eye(n_levels, dtype=complex)
    sz = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    sz_collective = 0.5 * (np.kron(sz, single_identity) + np.kron(single_identity, sz))

    channels = [
        (gamma_heating, np.kron(spin_identity, creation_operator(n_levels))),
        (gamma_motional_dephasing, np.kron(spin_identity, number_operator(n_levels))),
        (gamma_spin_dephasing, np.kron(sz_collective, motion_identity)),
    ]
    operators = []
    for gamma, operator in channels:
        gamma = float(gamma)
        if gamma < 0.0:
            raise ValueError("decoherence rates must be non-negative.")
        if gamma > 0.0:
            operators.append(np.sqrt(gamma) * operator)
    return operators


def spin_boson_control_system(
    n_levels,
    phi_s,
    mode_vector=(0.5, -0.5),
    eta=DEFAULT_LAMB_DICKE_ETA,
    static_fluctuations=(),
    control_fluctuations=(),
    collapse_operators=(),
):
    """Build the two-channel spin-boson Hamiltonian with optional fluctuations.

    The nominal Hamiltonian is
    ``alpha_1 I_spin ⊗ a†a + alpha_2 eta S_phi ⊗ X1`` with
    ``X1 = (a + a†) / 2`` and two-qubit
    ``S_phi = b_1 sigma_phi ⊗ I + b_2 I ⊗ sigma_phi``. The default
    ``mode_vector`` is the stretch-mode vector ``(1, -1) / 2``; use
    ``(1, 1) / 2`` for the COM mode. The default Lamb-Dicke factor is
    ``eta = 0.075``. Fluctuation terms use the same long-correlation convention
    as the perturbative expansion path: static matrices are already-scaled
    ``sigma_xi H_xi`` terms and control fluctuation matrices are already-scaled
    ``sigma_chi_i H_chi_i`` terms.
    """

    spin_identity = np.eye(4, dtype=complex)
    a = annihilation_operator(n_levels)
    adag = a.conj().T
    x1 = 0.5 * (a + adag)
    dimension = 4 * n_levels

    drift = np.zeros((dimension, dimension), dtype=complex)
    controls = [
        np.kron(spin_identity, adag @ a),
        float(eta) * np.kron(two_qubit_spin_phase_mode(phi_s, mode_vector), x1),
    ]
    if len(collapse_operators):
        return LindbladOpenSystem(
            drift=drift,
            controls=controls,
            static_fluctuations=static_fluctuations,
            control_fluctuations=control_fluctuations,
            collapse_operators=tuple(collapse_operators),
        )
    return FluctuatingClosedSystem(
        drift=drift,
        controls=controls,
        static_fluctuations=static_fluctuations,
        control_fluctuations=control_fluctuations,
    )


def spin_boson_initial_pulse(
    n_steps=200,
    total_time_us=225.8,
    alpha1_khz_bounds=DEFAULT_ALPHA1_KHZ_BOUNDS,
    alpha2_khz_bounds=DEFAULT_ALPHA2_KHZ_BOUNDS,
    alpha1_cycles=1.0,
):
    if n_steps < 1:
        raise ValueError("n_steps must be at least 1.")
    total_time = float(total_time_us) * 1e-6
    if total_time <= 0.0:
        raise ValueError("total_time_us must be positive.")
    if alpha1_cycles <= 0.0:
        raise ValueError("alpha1_cycles must be positive.")

    alpha1_lower, alpha1_upper = khz_bounds_to_rad_s(alpha1_khz_bounds)
    alpha2_lower, alpha2_upper = khz_bounds_to_rad_s(alpha2_khz_bounds)
    dt = total_time / n_steps
    normalized_time = (np.arange(n_steps, dtype=float) + 0.5) / n_steps

    alpha1_center = 0.5 * (alpha1_upper + alpha1_lower)
    alpha1_scale = 0.5 * (alpha1_upper - alpha1_lower)
    alpha1 = alpha1_center + 0.7 * alpha1_scale + 0.3 * alpha1_scale * np.cos(
        2.0 * np.pi * alpha1_cycles * normalized_time
    )
    alpha2 = alpha2_lower + (alpha2_upper - alpha2_lower) * np.sin(
        np.pi * normalized_time
    )

    return PiecewiseConstantPulse(
        amplitudes=np.column_stack([alpha1, alpha2]),
        dt=dt,
    )


def spin_boson_parameterization(
    n_steps=200,
    alpha1_khz_bounds=DEFAULT_ALPHA1_KHZ_BOUNDS,
    alpha2_khz_bounds=DEFAULT_ALPHA2_KHZ_BOUNDS,
):
    if n_steps < 1:
        raise ValueError("n_steps must be at least 1.")
    alpha1_lower, alpha1_upper = khz_bounds_to_rad_s(alpha1_khz_bounds)
    alpha2_lower, alpha2_upper = khz_bounds_to_rad_s(alpha2_khz_bounds)
    return BoundedAmplitudeParameterization(
        lower=np.array([alpha1_lower, alpha2_lower], dtype=float),
        upper=np.array([alpha1_upper, alpha2_upper], dtype=float),
    )

