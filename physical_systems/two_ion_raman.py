"""Two-ion Raman gate (6 lasers, 4 motional modes) physics and system definition.

Everything two-ion-Raman lives in this one file: the operators and Hamiltonian
builders, the noise-term declarations, the fidelity definition
(``state_pairs``), and the ``TwoIonRamanDefinition`` adapter registered under
``system.type: two_ion_raman``. Shared oscillator/gate/test-state conventions
are imported from ``physical_systems.spin_boson`` (the reference module) so the
two systems cannot drift apart.

Physical model (Wu, Wang & Duan, PRA 97, 062325 (2018))
-------------------------------------------------------
Two ions in a linear Paul trap, each individually addressed by its own
phase-insensitive three-beam set: a reference beam plus a red- and a
blue-detuned beam forming that ion's two Raman pairs at beat detunings
``+/- mu`` ("2 Raman transitions" = one per ion, 6 lasers total; each beam
shines exactly one ion). All beams are transverse, so all four collective
transverse phonon modes are kept in the Hilbert space: mode 1 = x-COM,
mode 2 = x-rocking, mode 3 = y-COM, mode 4 = y-rocking; ``Delta k`` lies in
the transverse (x-y) plane and projects on both radial directions. The
Hilbert space is ``(2 x 2) tensor Fock(n_levels)^4`` with every
operator built as ``kron(spin_part, motion_part)``; the flat index of
``|s>|n1 n2 n3 n4>`` is ``s*n^4 + ((n1*n + n2)*n + n3)*n + n4``.

After adiabatic elimination of the excited state, first-order Lamb-Dicke
expansion, dropping the carrier (oscillates at ``+/- mu``, paper Eq. 10) and
the Molmer-Sorensen RWA, in the frame rotating at ``mu`` for the modes the
blue pair of ion ``j`` survives as the anti-Jaynes-Cummings coupling
``B_j = sigma_j^{+,phi_s} A_j^dag + h.c.`` and the red pair as the JC coupling
``R_j = sigma_j^{+,phi_s} A_j + h.c.``, with the per-ion collective mode
operator ``A_j = sum_k eta_k b_j^k a_k`` (``b_1^k = 0.5``, ``b_2^k =
0.5 * s_k``, the per-ion split of spin_boson's ``mode_vector = (0.5, +/-0.5)``
convention). The drift carries the per-mode detunings ``delta_k = omega_k -
mu`` on the number operators plus optional deterministic per-ion differential
AC-Stark shifts.

Control channels (12 = 2 per beam x 6 beams)
--------------------------------------------
Each beam contributes a (rabi, detuning) channel pair, and every channel
multiplies a *constant* Hamiltonian (no control parameter ever sits inside an
exponential), which requires the approximations below:

- **rabi** (effective pair-Rabi units, kHz): a pair's coupling is bilinear in
  its two beams' fields (``Omega_1 Omega_2 / 2 Delta``); the linear model uses
  the contribution convention — each pair's effective Rabi is the *sum* of its
  two beams' channel amplitudes (``u_i ~ Omega_bar_partner * Omega_i / (4
  Delta)``), the symmetric first-order linearization of the product (exact
  under common relative modulation; drive off at zero controls). The reference
  beam belongs to both of its ion's pairs, so ``rabi_ref_j -> (B_j + R_j)/2``
  while ``rabi_blue_j -> B_j/2`` and ``rabi_red_j -> R_j/2``. The
  drive-proportional differential AC-Stark shift rides on the same channels:
  the shift is ``stark_rabi_ratio`` times the effective pair Rabi (linear in
  the drive, ~1e-4 for 355 nm Yb+ per the paper Sec. III.B.5), so each rabi
  channel operator carries an extra ``kappa * Z_j`` term with ``Z_j =
  kron(sigma_z^(j)/2, I_motion)`` — ``2 kappa`` on ``rabi_ref_j`` (it feeds
  both pairs), ``kappa`` on ``rabi_blue_j`` / ``rabi_red_j``.
- **detuning** (kHz): a beam-frequency offset shifts its pairs' two-photon
  detunings (blue pair: ``d/d delta_ref = +1``, ``d/d delta_blue = -1``; red
  pair: ``d/d delta_red = +1``, ``d/d delta_ref = -1``). Per ion this
  decomposes into a qubit-like common part -> ``Z_j = kron(sigma_z^(j)/2,
  I_motion)`` and a drive-detuning part -> the alpha1-type generator
  ``N = kron(I_4, sum_k n_k)``: ``det_ref_j -> N``, ``det_blue_j -> -(Z_j +
  N)/2``, ``det_red_j -> (Z_j - N)/2``. There is no separate alpha1 channel —
  its role is played by the ``det_ref`` channels. Note ``det_ref1`` and
  ``det_ref2`` share the operator ``N``: a per-ion drive detuning has no exact
  static per-ion generator, the global ``N`` is the model's first-order
  approximation (per-ion *differential* mu shifts are outside the model).

Amplitudes are specified in kHz in the YAML config and converted to angular
frequency (rad/s) internally; the default target is the maximally entangling
Molmer-Sorensen XX(pi/2) gate (shared with spin_boson).

Noise enters in two independent ways, mirrored by the two ``system.noise``
subsections:

- ``fluctuations``: coherent quasi-static errors (shot-to-shot collective spin
  dephasing, a single common motional-frequency/mu drift on ``N``, and relative
  amplitude noise per control channel keyed by role). With 14 fluctuation
  terms the exact ``--faithful`` check costs ``hermite_points**14`` — run
  faithful checks with noise subsets (zero out sigmas) or few hermite points.
- ``decoherence``: incoherent Lindblad channels — per-mode heating, motional
  dephasing, collective spin dephasing, and spontaneous emission from the
  adiabatically eliminated excited state (per-ion Raman spin flips at rate
  ``gamma/2`` each plus Rayleigh dephasing; supply the constant rates
  ``~ gamma_e Omega^2 / Delta^2`` via the config).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import override

import numpy as np

from physical_systems.common import (
    ControlChannel,
    DecoherenceConfigBase,
    FluctuationsConfigBase,
    PopulationStructure,
    StateProbe,
    SystemDefinitionBase,
    basis_state,
    validate_pulse_config,
)
from physical_systems.spin_boson import (
    annihilation_operator,
    creation_operator,
    number_operator,
    resolve_target_gate,
    two_qubit_logical_test_states,
)
from quantum_control import RAD_S_PER_KHZ, khz_bounds_to_rad_s
from quantum_control.problems.state_average import StatePair
from quantum_control.pulses.parameterization import BoundedAmplitudeParameterization
from quantum_control.pulses.pulse import PiecewiseConstantPulse
from quantum_control.systems import (
    ClosedSystem,
    DecoherenceChannel,
    FluctuationTerm,
    OpenSystem,
)

N_MODES = 4
N_IONS = 2
N_CONTROLS = 12

# Bare transverse mode frequencies omega_k relative to the two-photon carrier
# (no drive detuning baked in); illustrative set from omega_x/2pi = 3.0 MHz,
# omega_y/2pi = 2.9 MHz, omega_z/2pi = 0.5 MHz with rocking_t =
# sqrt(omega_t^2 - omega_z^2) per transverse direction. The nominal beat
# detuning mu is a separate knob; the drift carries delta_k = omega_k - mu.
# Override with the actual trap's numbers in the YAML config.
DEFAULT_MODE_FREQUENCIES_KHZ = (3000.0, 2958.0, 2900.0, 2857.0)
DEFAULT_MU_KHZ = 3010.0
# Delta k in the transverse plane projects on both x and y; eta_k scales with
# the projection and 1/sqrt(omega_k). Illustrative values.
DEFAULT_ETAS = (0.075, 0.076, 0.053, 0.054)
# Differential AC-Stark shift per unit effective pair Rabi (dimensionless);
# ~1e-4 for 355 nm Yb+ per the reference paper (Sec. III.B.5).
DEFAULT_STARK_RABI_RATIO = 1.0e-4
DEFAULT_MODE_SIGNS = (1.0, -1.0, 1.0, -1.0)
DEFAULT_RABI_KHZ_BOUNDS = (0.0, 200.0)
DEFAULT_DETUNING_KHZ_BOUNDS = (-60.0, 60.0)

# Channel order: per ion, per beam (ref, blue, red), the (rabi, det) pair.
# Role of channel i is CONTROL_ROLES[i % 2].
CONTROL_ROLES = ("rabi", "det")
CONTROL_LABELS = tuple(
    f"{role}_{beam}{ion}"
    for ion in (1, 2)
    for beam in ("ref", "blue", "red")
    for role in CONTROL_ROLES
)
RABI_COLUMNS = tuple(i for i, label in enumerate(CONTROL_LABELS) if label.startswith("rabi"))


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------

def _validate_mode_tuple(values, name):
    values = np.asarray(values, dtype=float)
    if values.shape != (N_MODES,):
        raise ValueError(f"{name} must contain exactly {N_MODES} per-mode values.")
    return values


def _validate_ion_index(ion_index):
    if ion_index not in (0, 1):
        raise ValueError("ion_index must be 0 or 1.")


def mode_operator(single_mode_operator, mode_index, n_levels):
    """Embed a single-mode operator at ``mode_index`` in ``Fock(n_levels)^4``."""
    if not 0 <= mode_index < N_MODES:
        raise ValueError(f"mode_index must be in range(0, {N_MODES}).")
    factors = [np.eye(n_levels, dtype=complex)] * N_MODES
    factors[mode_index] = np.asarray(single_mode_operator, dtype=complex)
    result = factors[0]
    for factor in factors[1:]:
        result = np.kron(result, factor)
    return result


def mode_number_operator(mode_index, n_levels):
    return mode_operator(number_operator(n_levels), mode_index, n_levels)


def total_number_operator(n_levels):
    """``sum_k n_k`` on the four-mode motion space (the ``N`` generator)."""
    return sum(mode_number_operator(k, n_levels) for k in range(N_MODES))


def collective_lowering_operator(etas, mode_signs, ion_index, n_levels):
    """``A_j = sum_k eta_k b_j^k a_k`` on the motion space, ``b_j^k`` from
    ``(0.5, 0.5 * s_k)``."""
    etas = _validate_mode_tuple(etas, "etas")
    mode_signs = _validate_mode_tuple(mode_signs, "mode_signs")
    _validate_ion_index(ion_index)
    weights = 0.5 * etas if ion_index == 0 else 0.5 * mode_signs * etas
    a = annihilation_operator(n_levels)
    return sum(
        weights[k] * mode_operator(a, k, n_levels) for k in range(N_MODES)
    )


def single_ion_operator(single_qubit_operator, ion_index):
    """Embed a 2x2 operator on ion ``ion_index`` in the 4-dim two-qubit space."""
    _validate_ion_index(ion_index)
    single_qubit_operator = np.asarray(single_qubit_operator, dtype=complex)
    identity = np.eye(2, dtype=complex)
    if ion_index == 0:
        return np.kron(single_qubit_operator, identity)
    return np.kron(identity, single_qubit_operator)


def collective_sz():
    """``0.5 * (sz x I + I x sz)`` on the two-qubit spin space."""
    sz = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    return 0.5 * (single_ion_operator(sz, 0) + single_ion_operator(sz, 1))


def pair_coupling_operators(phi_s, etas, mode_signs, ion_index, n_levels):
    """The full-space Hermitian pair couplings ``(B_j, R_j)`` of ion ``j``.

    ``B_j = sigma_j^{+,phi_s} A_j^dag + h.c.`` (blue / anti-Jaynes-Cummings)
    and ``R_j = sigma_j^{+,phi_s} A_j + h.c.`` (red / Jaynes-Cummings), with
    ``sigma^{+,phi} = e^{i phi} sigma_+`` so that ``B + R = 2 * kron(
    sigma_phi^(j), sum_k eta_k b_j^k X1_k)`` recovers the spin_boson
    ``spin_phase_operator`` convention.
    """
    lowering = collective_lowering_operator(etas, mode_signs, ion_index, n_levels)
    raising = lowering.conj().T
    sigma_plus = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=complex)
    spin_raising = np.exp(1j * phi_s) * single_ion_operator(sigma_plus, ion_index)
    blue_half = np.kron(spin_raising, raising)
    red_half = np.kron(spin_raising, lowering)
    return (
        blue_half + blue_half.conj().T,
        red_half + red_half.conj().T,
    )


# ---------------------------------------------------------------------------
# Fidelity definition
# ---------------------------------------------------------------------------

def motion_indices(n_levels, total_phonon_cutoff):
    """Fock-index tuples ``(n1, n2, n3, n4)`` with ``sum n_k <= cutoff``.

    Ordered like ``itertools.product`` (i.e. flat kron order); a cutoff of
    ``4 * (n_levels - 1)`` or larger keeps the whole basis.
    """
    if n_levels < 1:
        raise ValueError("n_levels must be at least 1.")
    if total_phonon_cutoff < 0:
        raise ValueError("total_phonon_cutoff must be non-negative.")
    return tuple(
        indices
        for indices in itertools.product(range(n_levels), repeat=N_MODES)
        if sum(indices) <= total_phonon_cutoff
    )


def motion_basis_state(indices, n_levels):
    """The basis vector ``|n1 n2 n3 n4>`` of the ``n_levels**4`` motion space."""
    flat_index = 0
    for index in indices:
        if not 0 <= index < n_levels:
            raise ValueError("each motional index must be in range(n_levels).")
        flat_index = flat_index * n_levels + index
    return basis_state(flat_index, n_levels ** N_MODES)


def motion_resolved_gate_state_pairs_4modes(target_gate, n_levels, total_phonon_cutoff):
    """Expand a 4x4 spin gate into weighted spin x 4-mode-motion ``StatePair``s.

    Initial states are the 16 two-qubit logical test states with all four
    modes in their ground state; targets are resolved over every motional
    basis state with total phonon number ``<= total_phonon_cutoff``. Because
    the motional targets at fixed spin target are orthonormal, the weighted
    fidelity sum equals the motion-traced spin-averaged gate fidelity when the
    cutoff keeps the full basis (``>= 4 * (n_levels - 1)``); with a smaller
    cutoff it is a strict lower bound that additionally penalizes population
    left above the cutoff (a converged MS pulse saturates the bound).
    """
    target_gate = np.asarray(target_gate, dtype=complex)
    if target_gate.shape != (4, 4):
        raise ValueError("target_gate must have shape (4, 4).")

    motion_ground = motion_basis_state((0,) * N_MODES, n_levels)
    resolved_indices = motion_indices(n_levels, total_phonon_cutoff)
    weight = 1.0 / len(two_qubit_logical_test_states())
    pairs = []
    for spin_state in two_qubit_logical_test_states():
        initial_state = np.kron(spin_state, motion_ground)
        target_spin = target_gate @ spin_state
        for indices in resolved_indices:
            target_state = np.kron(target_spin, motion_basis_state(indices, n_levels))
            pairs.append(StatePair(initial_state, target_state, weight))
    return tuple(pairs)


def ms_bell_target_motion_ground(n_levels):
    """Bell state ``(|00> - i|11>)/sqrt(2)`` with all four modes in the ground
    state; where MS_XX(pi/2) sends ``|00>|0000>`` (the driver's probe target)."""
    motion_dimension = n_levels ** N_MODES
    target = np.zeros(4 * motion_dimension, dtype=complex)
    target[0] = 1.0 / np.sqrt(2.0)
    target[3 * motion_dimension] = -1j / np.sqrt(2.0)
    return target


# ---------------------------------------------------------------------------
# System builders
# ---------------------------------------------------------------------------

def _drift_and_controls(
    n_levels,
    phi_s,
    mode_frequencies_rad_s,
    mu_rad_s,
    etas,
    mode_signs,
    stark_shifts_rad_s,
    stark_rabi_ratio,
):
    """The drift and the 12 control Hamiltonians, in ``CONTROL_LABELS`` order.

    The drift carries ``delta_k = omega_k - mu`` per mode; the rabi channel
    operators include the drive-proportional differential AC-Stark component
    ``stark_rabi_ratio * Z_j`` (doubled on the reference-beam channel). Shared
    by ``two_ion_raman_control_system`` and
    ``TwoIonRamanDefinition.fluctuation_terms`` so the control-kind
    fluctuation operators are byte-identical to the control Hamiltonians.
    """
    mode_frequencies = _validate_mode_tuple(
        mode_frequencies_rad_s, "mode_frequencies_rad_s"
    )
    mode_detunings = mode_frequencies - float(mu_rad_s)
    stark_rabi_ratio = float(stark_rabi_ratio)
    stark_shifts = np.asarray(stark_shifts_rad_s, dtype=float)
    if stark_shifts.shape != (N_IONS,):
        raise ValueError(f"stark_shifts_rad_s must contain exactly {N_IONS} values.")

    spin_identity = np.eye(4, dtype=complex)
    motion_identity = np.eye(n_levels ** N_MODES, dtype=complex)
    sz = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    number_total = np.kron(spin_identity, total_number_operator(n_levels))

    drift = sum(
        mode_detunings[k] * np.kron(spin_identity, mode_number_operator(k, n_levels))
        for k in range(N_MODES)
    )
    z_operators = []
    for ion_index in range(N_IONS):
        z_operator = 0.5 * np.kron(single_ion_operator(sz, ion_index), motion_identity)
        z_operators.append(z_operator)
        drift = drift + stark_shifts[ion_index] * z_operator

    controls = []
    for ion_index in range(N_IONS):
        blue, red = pair_coupling_operators(phi_s, etas, mode_signs, ion_index, n_levels)
        z_operator = z_operators[ion_index]
        stark = stark_rabi_ratio * z_operator
        controls += [
            0.5 * (blue + red) + 2.0 * stark,  # rabi_ref: feeds both pairs
            number_total,                      # det_ref: pure drive-detuning
            0.5 * blue + stark,                # rabi_blue
            -0.5 * (z_operator + number_total),  # det_blue
            0.5 * red + stark,                 # rabi_red
            0.5 * (z_operator - number_total),   # det_red
        ]
    return drift, controls


def two_ion_raman_collapse_operators(
    n_levels,
    gamma_heating=(0.0, 0.0, 0.0, 0.0),
    gamma_motional_dephasing=0.0,
    gamma_spin_dephasing=0.0,
    gamma_raman=(0.0, 0.0),
    gamma_rayleigh=(0.0, 0.0),
):
    r"""Build scaled jump operators $L = \sqrt{\gamma}\,A$ for the two-ion system.

    Heating uses per-mode $A = I_{\mathrm{spin}} \otimes a_k^\dagger$ (one rate
    per mode — heating is strongly mode-dependent), motional dephasing one
    common rate on each $I_{\mathrm{spin}} \otimes n_k$, spin dephasing the
    collective $\sigma_z$, spontaneous Raman scattering per-ion spin flips
    $\sigma_\pm^{(j)}$ at rate $\gamma_j/2$ each, and Rayleigh scattering
    per-ion dephasing $\sigma_z^{(j)}/2$. Rates are angular frequencies
    (rad/s); zero-rate channels are omitted.
    """
    gamma_heating = _validate_mode_tuple(gamma_heating, "gamma_heating")
    gamma_raman = np.asarray(gamma_raman, dtype=float)
    gamma_rayleigh = np.asarray(gamma_rayleigh, dtype=float)
    if gamma_raman.shape != (N_IONS,) or gamma_rayleigh.shape != (N_IONS,):
        raise ValueError(f"per-ion rate tuples must contain exactly {N_IONS} values.")

    spin_identity = np.eye(4, dtype=complex)
    motion_identity = np.eye(n_levels ** N_MODES, dtype=complex)
    sz = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    sigma_plus = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=complex)
    sigma_minus = sigma_plus.conj().T
    adag = creation_operator(n_levels)

    channels = [
        (gamma_heating[k], np.kron(spin_identity, mode_operator(adag, k, n_levels)))
        for k in range(N_MODES)
    ]
    channels += [
        (gamma_motional_dephasing, np.kron(spin_identity, mode_number_operator(k, n_levels)))
        for k in range(N_MODES)
    ]
    channels.append(
        (gamma_spin_dephasing, np.kron(collective_sz(), motion_identity))
    )
    for ion_index in range(N_IONS):
        channels += [
            (0.5 * gamma_raman[ion_index],
             np.kron(single_ion_operator(sigma_plus, ion_index), motion_identity)),
            (0.5 * gamma_raman[ion_index],
             np.kron(single_ion_operator(sigma_minus, ion_index), motion_identity)),
            (gamma_rayleigh[ion_index],
             0.5 * np.kron(single_ion_operator(sz, ion_index), motion_identity)),
        ]
    operators = []
    for gamma, operator in channels:
        gamma = float(gamma)
        if gamma < 0.0:
            raise ValueError("decoherence rates must be non-negative.")
        if gamma > 0.0:
            operators.append(np.sqrt(gamma) * operator)
    return operators


def two_ion_raman_control_system(
    n_levels,
    phi_s,
    mode_frequencies_rad_s=tuple(RAD_S_PER_KHZ * f for f in DEFAULT_MODE_FREQUENCIES_KHZ),
    mu_rad_s=RAD_S_PER_KHZ * DEFAULT_MU_KHZ,
    etas=DEFAULT_ETAS,
    mode_signs=DEFAULT_MODE_SIGNS,
    stark_shifts_rad_s=(0.0, 0.0),
    stark_rabi_ratio=DEFAULT_STARK_RABI_RATIO,
    static_fluctuations=(),
    control_fluctuations=(),
    collapse_operators=(),
):
    """Build the 12-channel two-ion-Raman system with optional noise.

    The nominal Hamiltonian is the drift (per-mode detunings ``omega_k - mu``
    + constant per-ion Stark offsets) plus the 12 per-beam (rabi, detuning)
    control terms in ``CONTROL_LABELS`` order, with the drive-proportional
    AC-Stark component riding on the rabi channels; see the module docstring
    for each channel's operator.

    With no noise arguments this returns a plain ``ClosedSystem``. The noise
    arguments take *already-scaled* matrices (``sigma_xi H_xi`` for static,
    ``sigma_chi_i H_chi_i`` per control channel, ``sqrt(gamma) L`` jump
    operators) and are wrapped into unit-strength ``NoiseTerm``s on an
    ``OpenSystem``, matching the spin_boson convention exactly.
    """
    drift, controls = _drift_and_controls(
        n_levels,
        phi_s,
        mode_frequencies_rad_s,
        mu_rad_s,
        etas,
        mode_signs,
        stark_shifts_rad_s,
        stark_rabi_ratio,
    )

    noise_terms = [
        FluctuationTerm(
            name=f"static[{index}]",
            operator=matrix,
            definition="pre-scaled static fluctuation matrix",
            coefficient=1.0,
            kind="static",
        )
        for index, matrix in enumerate(static_fluctuations)
    ]
    noise_terms += [
        FluctuationTerm(
            name=f"control[{index}]",
            operator=matrix,
            definition="pre-scaled control fluctuation matrix",
            coefficient=1.0,
            kind="control",
        )
        for index, matrix in enumerate(control_fluctuations)
    ]
    noise_terms += [
        DecoherenceChannel(
            name=f"collapse[{index}]",
            operator=operator,
            definition="pre-scaled jump operator",
            rate=1.0,
        )
        for index, operator in enumerate(collapse_operators)
    ]
    if noise_terms:
        return OpenSystem(drift=drift, controls=controls, noise_terms=tuple(noise_terms))
    return ClosedSystem(drift=drift, controls=controls)


def two_ion_raman_initial_pulse(
    n_steps=200,
    total_time_us=225.8,
    rabi_khz_bounds=DEFAULT_RABI_KHZ_BOUNDS,
    detuning_khz_bounds=DEFAULT_DETUNING_KHZ_BOUNDS,
    detuning_cycles=1.0,
):
    """Deterministic "cosine" initial guess: the six rabi columns share a
    half-amplitude ``sin(pi t)`` arch, the two ``det_ref`` columns a cosine
    around an offset midpoint (the spin_boson alpha1 shape), and the remaining
    detuning columns start at zero."""
    if n_steps < 1:
        raise ValueError("n_steps must be at least 1.")
    total_time = float(total_time_us) * 1e-6
    if total_time <= 0.0:
        raise ValueError("total_time_us must be positive.")
    if detuning_cycles <= 0.0:
        raise ValueError("detuning_cycles must be positive.")

    rabi_lower, rabi_upper = khz_bounds_to_rad_s(rabi_khz_bounds)
    detuning_lower, detuning_upper = khz_bounds_to_rad_s(detuning_khz_bounds)
    dt = total_time / n_steps
    normalized_time = (np.arange(n_steps, dtype=float) + 0.5) / n_steps

    rabi = rabi_lower + 0.5 * (rabi_upper - rabi_lower) * np.sin(np.pi * normalized_time)
    detuning_center = 0.5 * (detuning_upper + detuning_lower)
    detuning_scale = 0.5 * (detuning_upper - detuning_lower)
    det_ref = detuning_center + 0.7 * detuning_scale + 0.3 * detuning_scale * np.cos(
        2.0 * np.pi * detuning_cycles * normalized_time
    )
    det_ref = np.clip(det_ref, detuning_lower, detuning_upper)

    amplitudes = np.zeros((n_steps, N_CONTROLS), dtype=float)
    for column in RABI_COLUMNS:
        amplitudes[:, column] = rabi
    for column, label in enumerate(CONTROL_LABELS):
        if label.startswith("det_ref"):
            amplitudes[:, column] = det_ref
    return PiecewiseConstantPulse(amplitudes=amplitudes, dt=dt)


def two_ion_raman_parameterization(
    n_steps=200,
    rabi_khz_bounds=DEFAULT_RABI_KHZ_BOUNDS,
    detuning_khz_bounds=DEFAULT_DETUNING_KHZ_BOUNDS,
):
    if n_steps < 1:
        raise ValueError("n_steps must be at least 1.")
    lower, upper = _control_bounds_rad_s(rabi_khz_bounds, detuning_khz_bounds)
    return BoundedAmplitudeParameterization(lower=lower, upper=upper)


def _control_bounds_rad_s(rabi_khz_bounds, detuning_khz_bounds):
    """Per-channel rad/s bounds arrays, role-mapped over ``CONTROL_LABELS``."""
    bounds_by_role = {
        "rabi": khz_bounds_to_rad_s(rabi_khz_bounds),
        "det": khz_bounds_to_rad_s(detuning_khz_bounds),
    }
    per_channel = [bounds_by_role[CONTROL_ROLES[i % 2]] for i in range(N_CONTROLS)]
    lower, upper = zip(*per_channel)
    return np.array(lower, dtype=float), np.array(upper, dtype=float)


# ---------------------------------------------------------------------------
# YAML config schema
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TwoIonRamanParams:
    """Physical parameters; fields define the ``system.params`` YAML schema.

    Attributes:
        n_levels: Fock-space truncation, shared by all four modes. The full
            Hilbert-space dimension is ``4 * n_levels**4``.
        phi_s: Fixed spin phase of both Raman pairs (set by the beam phases at
            their nominal values; ``phi_s = 0`` gives the XX-type MS drive).
        mode_frequencies_khz: Bare transverse mode frequencies ``omega_k`` in
            kHz, referenced to the two-photon carrier with no drive detuning
            baked in; order (x-COM, x-rocking, y-COM, y-rocking).
        mu_khz: Nominal Raman beat (sideband) detuning ``mu`` of the
            bichromatic pairs, kHz. The drift carries ``delta_k = omega_k -
            mu`` per mode; the detuning control channels modulate about it.
        etas: Per-mode Lamb-Dicke couplings of the nominal ``Delta k``.
        mode_signs: Per-mode relative sign of ion 2's participation
            (``+1`` COM-type, ``-1`` rocking-type modes).
        stark_shifts_khz: Constant residual differential AC-Stark offset per
            ion (kHz), added to the drift on ``sigma_z^(j)/2`` (e.g. an
            uncompensated shift from the fixed reference beams); default zero.
        stark_rabi_ratio: Drive-proportional differential AC-Stark shift per
            unit effective pair Rabi (dimensionless, ~1e-4 for 355 nm Yb+ per
            the paper Sec. III.B.5). Enters the *nominal* Hamiltonian through
            the rabi channel operators (``2*ratio*Z_j`` on ``rabi_ref``,
            ``ratio*Z_j`` on ``rabi_blue``/``rabi_red``), so the shift follows
            the drive amplitude; its shot-to-shot fluctuation is covered by
            ``sigma_control_rabi_relative`` scaling the whole channel.
        target_gate: Key into the shared spin_boson ``TARGET_GATES`` registry.
        motion_target_phonon_cutoff: Total-phonon cutoff of the motional
            target resolution in ``state_pairs``. The default 2 keeps the pair
            count tractable and yields a strict lower bound on the
            motion-traced spin fidelity; ``>= 4 * (n_levels - 1)`` restores
            the exact full-basis average.
        rabi_khz_bounds: ``(lower, upper)`` bounds shared by all six rabi
            channels, in effective pair-Rabi kHz.
        detuning_khz_bounds: Bounds shared by all six detuning channels (kHz).
        initial_pulse_shape: ``"random"`` (default) or ``"cosine"``. Random
            starts the ``det_ref`` columns near their offset midpoint with
            per-step Gaussian noise and the rabi columns flat at the bounds
            midpoint; cosine is the deterministic
            ``two_ion_raman_initial_pulse`` shape. Both leave the remaining
            detuning columns at zero.
        detuning_cycles: ``det_ref`` cosine cycles across the pulse (cosine
            shape only).
        detuning_offset_fraction: Offset of the initial ``det_ref`` guess from
            the bounds midpoint, as a fraction of the half-width (random
            shape only).
        detuning_noise_fraction: Standard deviation of the Gaussian noise on
            the initial ``det_ref`` guess, in half-widths (random shape only).
        rabi_endpoint_zero: When true (default), the parameterization pins all
            six rabi columns to zero at the first and last time step
            (``RabiEndpointZeroParameterization``), so the drive ramps up from
            and returns to zero.
    """

    n_levels: int = 3
    phi_s: float = 0.0
    mode_frequencies_khz: tuple[float, float, float, float] = DEFAULT_MODE_FREQUENCIES_KHZ
    mu_khz: float = DEFAULT_MU_KHZ
    etas: tuple[float, float, float, float] = DEFAULT_ETAS
    mode_signs: tuple[float, float, float, float] = DEFAULT_MODE_SIGNS
    stark_shifts_khz: tuple[float, float] = (0.0, 0.0)
    stark_rabi_ratio: float = DEFAULT_STARK_RABI_RATIO
    target_gate: str = "ms_xx_pi_over_2"
    motion_target_phonon_cutoff: int = 2
    rabi_khz_bounds: tuple[float, float] = DEFAULT_RABI_KHZ_BOUNDS
    detuning_khz_bounds: tuple[float, float] = DEFAULT_DETUNING_KHZ_BOUNDS
    initial_pulse_shape: str = "random"
    detuning_cycles: float = 1.0
    detuning_offset_fraction: float = 0.7
    detuning_noise_fraction: float = 0.3
    rabi_endpoint_zero: bool = True


@dataclass(frozen=True)
class TwoIonRamanDecoherence(DecoherenceConfigBase):
    """Lindblad decoherence rates (``system.noise.decoherence`` YAML schema).

    All rates are in 1/s and feed ``TwoIonRamanDefinition.decoherence_channels``;
    the ``enabled`` switch and ``any_rate_positive`` gating come from
    ``DecoherenceConfigBase``. Heating gets one rate per mode (uniform-field
    noise heats COM-type modes orders of magnitude faster than
    rocking-type ones); the spontaneous-emission rates model the
    scattering ``~ gamma_e Omega^2 / Delta^2`` from the adiabatically
    eliminated excited state as constant per-ion rates.

    Attributes:
        gamma_heating_mode1: Heating rate of mode 1 (x-COM).
        gamma_heating_mode2: Heating rate of mode 2 (x-rocking).
        gamma_heating_mode3: Heating rate of mode 3 (y-COM).
        gamma_heating_mode4: Heating rate of mode 4 (y-rocking).
        gamma_motional_dephasing: Common per-mode dephasing rate (number-
            operator jump on each mode).
        gamma_spin_dephasing: Collective spin dephasing rate.
        gamma_raman_ion1: Spontaneous Raman scattering rate of ion 1
            (spin-flip jumps ``sigma_+``, ``sigma_-`` at half this rate each).
        gamma_raman_ion2: Same for ion 2.
        gamma_rayleigh_ion1: Rayleigh scattering (elastic) dephasing rate of
            ion 1 (``sigma_z^(1)/2`` jump).
        gamma_rayleigh_ion2: Same for ion 2.
    """

    gamma_heating_mode1: float = 0.0
    gamma_heating_mode2: float = 0.0
    gamma_heating_mode3: float = 0.0
    gamma_heating_mode4: float = 0.0
    gamma_motional_dephasing: float = 0.0
    gamma_spin_dephasing: float = 0.0
    gamma_raman_ion1: float = 0.0
    gamma_raman_ion2: float = 0.0
    gamma_rayleigh_ion1: float = 0.0
    gamma_rayleigh_ion2: float = 0.0


@dataclass(frozen=True)
class TwoIonRamanFluctuations(FluctuationsConfigBase):
    """Coherent fluctuation strengths (``system.noise.fluctuations`` schema).

    Each sigma is the standard deviation of a quasi-static error term; see
    ``TwoIonRamanDefinition.fluctuation_terms`` for the operator each one
    multiplies. The relative-control sigmas are keyed by channel *role*: one
    value feeds all six channels of that role, giving 12 control-kind terms
    (plus 2 static ones). Note the exact ``--faithful`` evaluation draws an
    independent variable per term (``hermite_points**14`` nodes at full
    noise) — zero out sigmas to run faithful checks on subsets.

    Attributes:
        sigma_static_spin_dephasing: Collective spin-dephasing offset, rad/s.
        sigma_static_motional_frequency: Common motional-frequency (mu-drift)
            offset multiplying ``sum_k n_k``, rad/s.
        sigma_control_rabi_relative: Relative (dimensionless) amplitude noise
            on each rabi channel.
        sigma_control_detuning_relative: Same for the detuning channels.
    """

    enabled: bool = True  # overrides the base default
    sigma_static_spin_dephasing: float = 314.159
    sigma_static_motional_frequency: float = 300.0
    sigma_control_rabi_relative: float = 0.0001
    sigma_control_detuning_relative: float = 0.0001


@dataclass(frozen=True)
class TwoIonRamanNoise:
    """Container splitting ``system.noise`` into its two YAML subsections."""

    decoherence: TwoIonRamanDecoherence = field(default_factory=TwoIonRamanDecoherence)
    fluctuations: TwoIonRamanFluctuations = field(default_factory=TwoIonRamanFluctuations)


# ---------------------------------------------------------------------------
# Parameterization override
# ---------------------------------------------------------------------------

class RabiEndpointZeroParameterization:
    """Parameterization wrapper pinning the rabi columns to zero at both ends.

    Generalizes spin_boson's ``Alpha2EndpointZeroParameterization`` from one
    hard-coded column to a tuple of columns: the drive must ramp up from zero
    and return to zero so the gate starts and ends with the qubits decoupled
    from the motion, and the constraint is enforced structurally in every
    representation the optimizer touches — physical amplitudes, normalized
    parameters, gradients, and bounds.

    ``base`` is the affine normalized-parameter map returned by
    ``two_ion_raman_parameterization``.
    """

    def __init__(self, base, columns=RABI_COLUMNS):
        self.base = base
        self.columns = tuple(columns)

    def to_physical(self, normalized):
        """Map normalized parameters to amplitudes, zeroing pinned endpoints."""
        amplitudes = self.base.to_physical(normalized)
        for column in self.columns:
            amplitudes[[0, -1], column] = 0.0
        return amplitudes

    def to_parameters(self, amplitudes):
        """Map amplitudes to normalized parameters.

        The endpoint entries are overwritten with the normalized value that
        maps back to physical 0, regardless of what the input amplitudes held,
        so round-tripping always lands on the constraint.
        """
        parameters = self.base.to_parameters(amplitudes)
        lower, upper = self.base.bounds_for(amplitudes.shape)
        for column in self.columns:
            parameters[[0, -1], column] = self._normalized_zero(
                lower[[0, -1], column], upper[[0, -1], column]
            )
        return parameters

    def pullback_gradient(self, physical_gradient):
        """Pull the physical gradient back, zeroing the frozen entries."""
        gradient = self.base.pullback_gradient(physical_gradient)
        for column in self.columns:
            gradient[[0, -1], column] = 0.0
        return gradient

    def parameter_bounds(self, shape):
        """Return flat parameter bounds with the endpoint entries collapsed.

        The frozen parameters get ``(value, value)`` bounds so a bounded
        optimizer treats them as fixed; bounds are indexed in the flattened
        (row-major, ``np.ravel_multi_index``) parameter order.
        """
        bounds = self.base.parameter_bounds(shape)
        lower, upper = self.base.bounds_for(shape)
        for column in self.columns:
            endpoint_value = self._normalized_zero(
                lower[[0, -1], column], upper[[0, -1], column]
            )
            for row, value in zip((0, shape[0] - 1), endpoint_value):
                bounds[np.ravel_multi_index((row, column), shape)] = (
                    float(value),
                    float(value),
                )
        return bounds

    def bounds_for(self, shape):
        """Physical amplitude bounds, broadcast to ``shape`` (delegates)."""
        return self.base.bounds_for(shape)

    @staticmethod
    def _normalized_zero(lower, upper):
        """Invert the affine map: the normalized value whose physical image is 0."""
        return (0.0 - 0.5 * (upper + lower)) / (0.5 * (upper - lower))


# ---------------------------------------------------------------------------
# Driver-facing definition
# ---------------------------------------------------------------------------

class TwoIonRamanDefinition(SystemDefinitionBase):
    """System definition registered under ``system.type: two_ion_raman``.

    Supplies the physics hooks of ``SystemDefinitionBase``; the generic
    ``build_systems`` assembly comes from the base class, while the initial
    pulse and parameterization are overridden for the system-specific start
    shape and rabi endpoint constraint.
    """

    name = "two_ion_raman"

    @override
    def default_params(self):
        return TwoIonRamanParams()

    @override
    def default_noise(self):
        return TwoIonRamanNoise()

    def _physics_arguments(self, params):
        """The kHz->rad/s-converted arguments of ``_drift_and_controls``."""
        return {
            "n_levels": params.n_levels,
            "phi_s": params.phi_s,
            "mode_frequencies_rad_s": tuple(
                RAD_S_PER_KHZ * value for value in params.mode_frequencies_khz
            ),
            "mu_rad_s": RAD_S_PER_KHZ * params.mu_khz,
            "etas": params.etas,
            "mode_signs": params.mode_signs,
            "stark_shifts_rad_s": tuple(
                RAD_S_PER_KHZ * value for value in params.stark_shifts_khz
            ),
            "stark_rabi_ratio": params.stark_rabi_ratio,
        }

    @override
    def build_closed_system(self, params):
        return two_ion_raman_control_system(**self._physics_arguments(params))

    @override
    def fluctuation_terms(self, params, fluctuations):
        """The 2 static + 12 control-kind coherent fluctuation terms.

        - ``spin-shift`` (static): collective spin dephasing,
          ``kron(0.5 * (sz x I + I x sz), I_motion)``.
        - ``motion-shift`` (static): common motional-frequency (mu) drift,
          ``kron(I_spin, sum_k n_k)``.
        - ``<label>-rel`` (control, one per channel in channel order): relative
          amplitude noise, sigma keyed by the channel's role.
        """
        n_levels = params.n_levels
        spin_identity = np.eye(4, dtype=complex)
        motion_identity = np.eye(n_levels ** N_MODES, dtype=complex)

        sigma_by_role = {
            "rabi": fluctuations.sigma_control_rabi_relative,
            "det": fluctuations.sigma_control_detuning_relative,
        }
        # The control-kind operators must match the corresponding control
        # Hamiltonian terms exactly (same operator, same eta/mode conventions):
        # the propagator applies them as amplitude * sigma * operator, so any
        # mismatch would silently change the meaning of the relative sigmas.
        _, controls = _drift_and_controls(**self._physics_arguments(params))
        terms = [
            FluctuationTerm(
                kind="static",
                name="spin-shift",
                coefficient=fluctuations.sigma_static_spin_dephasing,
                operator=np.kron(collective_sz(), motion_identity),
                definition="kron(0.5 * (sz x I + I x sz), I_motion)",
                usage="added directly to H_fluctuation",
            ),
            FluctuationTerm(
                kind="static",
                name="motion-shift",
                coefficient=fluctuations.sigma_static_motional_frequency,
                operator=np.kron(spin_identity, total_number_operator(n_levels)),
                definition="kron(I_spin, sum_k number_operator_k)",
                usage="added directly to H_fluctuation",
            ),
        ]
        terms += [
            FluctuationTerm(
                kind="control",
                name=f"{label}-rel",
                coefficient=sigma_by_role[CONTROL_ROLES[index % 2]],
                operator=controls[index],
                definition=f"control Hamiltonian of channel {label!r}",
                usage=f"{label}(t) * control[{index}]",
            )
            for index, label in enumerate(CONTROL_LABELS)
        ]
        return terms

    @override
    def decoherence_channels(self, params, decoherence):
        """The 15 Lindblad channels of the two-ion model.

        Same operators and ordering as the ``two_ion_raman_collapse_operators``
        convenience builder above; the ``sqrt(gamma)`` scaling lives on
        ``DecoherenceChannel.matrix`` and the base class drops zero-rate
        channels.
        """
        n_levels = params.n_levels
        spin_identity = np.eye(4, dtype=complex)
        motion_identity = np.eye(n_levels ** N_MODES, dtype=complex)
        sz = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
        sigma_plus = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=complex)
        sigma_minus = sigma_plus.conj().T
        adag = creation_operator(n_levels)

        heating_rates = (
            decoherence.gamma_heating_mode1,
            decoherence.gamma_heating_mode2,
            decoherence.gamma_heating_mode3,
            decoherence.gamma_heating_mode4,
        )
        channels = [
            DecoherenceChannel(
                name=f"heating-m{k + 1}",
                rate=heating_rates[k],
                operator=np.kron(spin_identity, mode_operator(adag, k, n_levels)),
                definition=f"kron(I_spin, adag_m{k + 1})",
            )
            for k in range(N_MODES)
        ]
        channels += [
            DecoherenceChannel(
                name=f"motion-dephasing-m{k + 1}",
                rate=decoherence.gamma_motional_dephasing,
                operator=np.kron(spin_identity, mode_number_operator(k, n_levels)),
                definition=f"kron(I_spin, number_operator_m{k + 1})",
            )
            for k in range(N_MODES)
        ]
        channels.append(
            DecoherenceChannel(
                name="spin-dephasing",
                rate=decoherence.gamma_spin_dephasing,
                operator=np.kron(collective_sz(), motion_identity),
                definition="kron(0.5 * (sz x I + I x sz), I_motion)",
            )
        )
        raman_rates = (decoherence.gamma_raman_ion1, decoherence.gamma_raman_ion2)
        rayleigh_rates = (decoherence.gamma_rayleigh_ion1, decoherence.gamma_rayleigh_ion2)
        for ion_index in range(N_IONS):
            channels += [
                DecoherenceChannel(
                    name=f"raman-up-ion{ion_index + 1}",
                    rate=0.5 * raman_rates[ion_index],
                    operator=np.kron(
                        single_ion_operator(sigma_plus, ion_index), motion_identity
                    ),
                    definition=f"kron(sigma_plus^({ion_index + 1}), I_motion), rate gamma/2",
                ),
                DecoherenceChannel(
                    name=f"raman-down-ion{ion_index + 1}",
                    rate=0.5 * raman_rates[ion_index],
                    operator=np.kron(
                        single_ion_operator(sigma_minus, ion_index), motion_identity
                    ),
                    definition=f"kron(sigma_minus^({ion_index + 1}), I_motion), rate gamma/2",
                ),
                DecoherenceChannel(
                    name=f"rayleigh-ion{ion_index + 1}",
                    rate=rayleigh_rates[ion_index],
                    operator=0.5 * np.kron(
                        single_ion_operator(sz, ion_index), motion_identity
                    ),
                    definition=f"kron(sigma_z^({ion_index + 1})/2, I_motion)",
                ),
            ]
        return channels

    @override
    def control_bounds(self, params):
        return _control_bounds_rad_s(
            params.rabi_khz_bounds,
            params.detuning_khz_bounds,
        )

    @override
    def build_initial_pulse(self, params, pulse_config):
        """Construct the initial pulse guess from the ``pulse`` config section.

        ``params.initial_pulse_shape`` selects the guess:

        - ``"random"``: the two ``det_ref`` columns start at the bounds
          midpoint shifted by ``detuning_offset_fraction`` half-widths plus
          per-step Gaussian noise of ``detuning_noise_fraction`` half-widths
          (reproducible when ``pulse_config.random_seed`` is set), clipped
          back into bounds; the six rabi columns start flat at their bounds
          midpoint; everything else starts at zero.
        - ``"cosine"``: the deterministic ``two_ion_raman_initial_pulse``
          shape.

        Either way the rabi endpoint-zero constraint is applied later by the
        parameterization, not here.
        """
        n_steps, dt = validate_pulse_config(pulse_config)
        if params.initial_pulse_shape == "cosine":
            return two_ion_raman_initial_pulse(
                n_steps=n_steps,
                total_time_us=float(pulse_config.total_time_us),
                rabi_khz_bounds=params.rabi_khz_bounds,
                detuning_khz_bounds=params.detuning_khz_bounds,
                detuning_cycles=params.detuning_cycles,
            )
        if params.initial_pulse_shape != "random":
            raise ValueError(
                f"initial_pulse_shape must be 'random' or 'cosine', "
                f"got {params.initial_pulse_shape!r}."
            )
        lower, upper = self.control_bounds(params)
        detuning_lower, detuning_upper = khz_bounds_to_rad_s(params.detuning_khz_bounds)
        rabi_lower, rabi_upper = khz_bounds_to_rad_s(params.rabi_khz_bounds)

        detuning_center = 0.5 * (detuning_upper + detuning_lower)
        detuning_scale = 0.5 * (detuning_upper - detuning_lower)
        # Seedless runs draw from numpy's global RNG (varies run to run);
        # with a seed, a dedicated generator gives reproducible pulses
        # without touching global state.
        detuning_noise = (
            np.random.randn(n_steps, 2)
            if pulse_config.random_seed is None
            else np.random.default_rng(pulse_config.random_seed).standard_normal(
                (n_steps, 2)
            )
        )
        det_ref = (
            detuning_center
            + params.detuning_offset_fraction * detuning_scale
            + params.detuning_noise_fraction * detuning_scale * detuning_noise
        )
        det_ref = np.clip(det_ref, detuning_lower, detuning_upper)

        amplitudes = np.zeros((n_steps, N_CONTROLS), dtype=float)
        # Flat at the bounds midpoint; the endpoint zeros come from the
        # parameterization, not from the initial guess.
        for column in RABI_COLUMNS:
            amplitudes[:, column] = 0.5 * (rabi_lower + rabi_upper)
        det_ref_columns = [
            column
            for column, label in enumerate(CONTROL_LABELS)
            if label.startswith("det_ref")
        ]
        for slot, column in enumerate(det_ref_columns):
            amplitudes[:, column] = det_ref[:, slot]
        return PiecewiseConstantPulse(amplitudes=amplitudes, dt=dt)

    @override
    def build_parameterization(self, params, pulse):
        """Return the bounded parameterization, rabi endpoints frozen by default.

        ``params.rabi_endpoint_zero`` (default true) applies the
        ``RabiEndpointZeroParameterization`` wrapper; when false the bare
        bounded parameterization is returned.
        """
        base = two_ion_raman_parameterization(
            pulse.n_steps,
            rabi_khz_bounds=params.rabi_khz_bounds,
            detuning_khz_bounds=params.detuning_khz_bounds,
        )
        if not params.rabi_endpoint_zero:
            return base
        return RabiEndpointZeroParameterization(base)

    @override
    def target_gate(self, params):
        return resolve_target_gate(params.target_gate)

    @override
    def state_pairs(self, params):
        """Expand the spin gate into motion-resolved (initial, target) pairs.

        The 4x4 target acts on the spins only; targets are resolved over the
        four-mode motional basis up to ``params.motion_target_phonon_cutoff``
        total phonons (see ``motion_resolved_gate_state_pairs_4modes``).
        """
        return motion_resolved_gate_state_pairs_4modes(
            self.target_gate(params),
            params.n_levels,
            params.motion_target_phonon_cutoff,
        )

    # ---- presentation hooks --------------------------------------------------

    @override
    def control_channels(self, params):
        return [
            ControlChannel(label=label, display_scale=1.0 / RAD_S_PER_KHZ, display_unit="kHz")
            for label in CONTROL_LABELS
        ]

    @override
    def population_structure(self, params):
        n_levels = params.n_levels
        return PopulationStructure(
            dims=(4, n_levels ** N_MODES),
            names=("two-qubit spin", "motion"),
            labels=(
                ("|00>", "|01>", "|10>", "|11>"),
                tuple(
                    "|" + "".join(str(index) for index in indices) + ">"
                    for indices in itertools.product(range(n_levels), repeat=N_MODES)
                ),
            ),
        )

    @override
    def probe_state_pair(self, params):
        return StateProbe(
            initial_state=basis_state(0, 4 * params.n_levels ** N_MODES),
            target_state=ms_bell_target_motion_ground(params.n_levels),
            description="|00>|0000> -> MS Bell state, motion ground",
        )
