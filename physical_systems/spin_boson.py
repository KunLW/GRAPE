"""Spin-boson (trapped-ion MS gate) physics and system definition.

Everything spin-boson lives in this one file: the operators and Hamiltonian
builders, the noise-term declarations, the fidelity definition
(``state_pairs``), and the ``SpinBosonDefinition`` adapter registered under
``system.type: spin_boson``. It is the reference implementation of the
system-definition interface documented in ``physical_systems/__init__.py``
(generic plumbing in ``physical_systems/common.py``).

Physical model
--------------
Two qubits coupled to one shared motional mode of the ion crystal, so the
Hilbert space is ``(2 x 2) tensor Fock(n_levels)`` and every operator below is
built as ``kron(spin_part, motion_part)``. Two piecewise-constant controls
drive the system:

- ``alpha1(t)``: motional-frequency control, coupling through the number
  operator ``kron(I_spin, n)``.
- ``alpha2(t)``: bichromatic spin-motion drive, coupling through
  ``eta * kron(S_phi, X1)`` with ``X1 = (a + adag)/2`` and ``S_phi`` the
  two-qubit spin operator at phase ``phi_s`` weighted by ``mode_vector``.

Amplitudes are specified in kHz in the YAML config and converted to angular
frequency (rad/s) internally; the default target is the maximally entangling
Molmer-Sorensen XX(pi/2) gate.

Noise enters in two independent ways, mirrored by the two ``system.noise``
subsections:

- ``fluctuations``: coherent quasi-static errors added to the Hamiltonian
  (shot-to-shot spin dephasing, motional-frequency drift, relative amplitude
  noise on each control). These feed the robustness (fluctuation-sensitivity)
  part of the objective.
- ``decoherence``: incoherent Lindblad channels (heating, motional dephasing,
  spin dephasing) handled by the first-order decoherence correction via
  collapse operators.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from quantum_control import (
    RAD_S_PER_KHZ,
    khz_bounds_to_rad_s,
    ms_xx_pi_over_2_gate,
    two_qubit_logical_test_states,
)
from quantum_control.pulses.parameterization import BoundedAmplitudeParameterization
from quantum_control.pulses.pulse import PiecewiseConstantPulse
from quantum_control.state_average import StatePair
from quantum_control.systems import (
    ClosedSystem,
    DecoherenceChannel,
    FluctuationTerm,
    OpenSystem,
)

from physical_systems.common import (
    ControlChannel,
    DecoherenceConfigBase,
    PopulationStructure,
    StateProbe,
    SystemDefinitionBase,
    basis_state,
    validate_pulse_config,
)

DEFAULT_LAMB_DICKE_ETA = 0.075
DEFAULT_ALPHA1_KHZ_BOUNDS = (1.0, 60.0)
DEFAULT_ALPHA2_KHZ_BOUNDS = (0.0, 200.0)


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------

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


def _collective_sz():
    """``0.5 * (sz ⊗ I + I ⊗ sz)`` on the two-qubit spin space."""
    single_identity = np.eye(2, dtype=complex)
    sz = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    return 0.5 * (np.kron(sz, single_identity) + np.kron(single_identity, sz))


# ---------------------------------------------------------------------------
# Fidelity definition
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# System builders (also used directly by the legacy experiments/ scripts)
# ---------------------------------------------------------------------------

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
    motion_identity = np.eye(n_levels, dtype=complex)

    channels = [
        (gamma_heating, np.kron(spin_identity, creation_operator(n_levels))),
        (gamma_motional_dephasing, np.kron(spin_identity, number_operator(n_levels))),
        (gamma_spin_dephasing, np.kron(_collective_sz(), motion_identity)),
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
    """Build the two-channel spin-boson system with optional noise.

    The nominal Hamiltonian is
    ``alpha_1 I_spin ⊗ a†a + alpha_2 eta S_phi ⊗ X1`` with
    ``X1 = (a + a†) / 2`` and two-qubit
    ``S_phi = b_1 sigma_phi ⊗ I + b_2 I ⊗ sigma_phi``. The default
    ``mode_vector`` is the stretch-mode vector ``(1, -1) / 2``; use
    ``(1, 1) / 2`` for the COM mode. The default Lamb-Dicke factor is
    ``eta = 0.075``.

    With no noise arguments this returns a plain ``ClosedSystem``. The noise
    arguments take *already-scaled* matrices (``sigma_xi H_xi`` for static,
    ``sigma_chi_i H_chi_i`` per control channel, ``sqrt(gamma) L`` jump
    operators) and are wrapped into unit-strength ``NoiseTerm``s on an
    ``OpenSystem``, so the numerics match the historical convention exactly.
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


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------

# Registry of target gates selectable via ``system.params.target_gate``.
# Values are zero-argument builders returning the unitary as an array-like;
# adding a new gate is a one-line entry here.
TARGET_GATES = {
    "ms_xx_pi_over_2": ms_xx_pi_over_2_gate,
}


def resolve_target_gate(name):
    """Look up ``name`` in ``TARGET_GATES`` and return the gate as an array.

    Raises ``ValueError`` if the name is unknown (listing the registered
    gates) or if the built matrix is not unitary, so a config typo or a broken
    gate builder fails loudly before any optimization starts.
    """
    try:
        builder = TARGET_GATES[name]
    except KeyError:
        registered = ", ".join(sorted(TARGET_GATES))
        raise ValueError(
            f"unknown target_gate {name!r}; registered gates: {registered}."
        ) from None
    gate = np.asarray(builder(), dtype=complex)
    # Frobenius norm of (U^dag U - I); zero iff U is unitary.
    deviation = np.linalg.norm(gate.conj().T @ gate - np.eye(gate.shape[0]))
    if deviation > 1e-10:
        raise ValueError(f"target_gate {name!r} is not unitary (deviation {deviation:.3g}).")
    return gate


def ms_bell_target_motion_ground(n_levels):
    """Bell state ``(|00> - i|11>)/sqrt(2)`` with the motion in its ground state.

    This is where MS_XX(pi/2) sends ``|00>|n=0>``; used as the driver's
    single-state fidelity probe.
    """
    dimension = 4 * n_levels
    target = np.zeros(dimension, dtype=complex)
    target[0] = 1.0 / np.sqrt(2.0)
    target[3 * n_levels] = -1j / np.sqrt(2.0)
    return target


# ---------------------------------------------------------------------------
# YAML config schema
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SpinBosonParams:
    """Physical parameters; fields define the ``system.params`` YAML schema.

    Attributes:
        n_levels: Fock-space truncation of the motional mode. The full
            Hilbert-space dimension is ``4 * n_levels``.
        phi_s: Phase of the two-qubit spin operator ``S_phi`` in the
            spin-motion coupling (``phi_s = 0`` gives the XX-type MS drive).
        eta: Lamb-Dicke parameter scaling the spin-motion coupling strength.
        mode_vector: Per-ion participation amplitudes of the driven motional
            mode; ``(0.5, -0.5)`` selects the out-of-phase (stretch) mode.
        target_gate: Key into ``TARGET_GATES`` selecting the gate to optimize
            toward.
        alpha1_khz_bounds: ``(lower, upper)`` amplitude bounds for the
            ``alpha1`` control, in kHz (converted to rad/s internally).
        alpha2_khz_bounds: Same for the ``alpha2`` control.
        alpha1_offset_fraction: Offset of the initial ``alpha1`` guess from
            the bounds midpoint, as a fraction of the bounds half-width
            (0.7 biases the start toward the upper bound).
        alpha1_noise_fraction: Standard deviation of the Gaussian noise added
            to the initial ``alpha1`` guess, as a fraction of the bounds
            half-width.
    """

    n_levels: int = 6
    phi_s: float = 0.0
    eta: float = DEFAULT_LAMB_DICKE_ETA
    mode_vector: tuple[float, float] = (0.5, -0.5)
    target_gate: str = "ms_xx_pi_over_2"
    alpha1_khz_bounds: tuple[float, float] = DEFAULT_ALPHA1_KHZ_BOUNDS
    alpha2_khz_bounds: tuple[float, float] = DEFAULT_ALPHA2_KHZ_BOUNDS
    alpha1_offset_fraction: float = 0.7
    alpha1_noise_fraction: float = 0.3


@dataclass(frozen=True)
class SpinBosonDecoherence(DecoherenceConfigBase):
    """Lindblad decoherence rates (``system.noise.decoherence`` YAML schema).

    All rates are in 1/s and feed ``SpinBosonDefinition.decoherence_channels``;
    the ``enabled`` switch and ``any_rate_positive`` gating come from
    ``DecoherenceConfigBase``.

    Attributes:
        gamma_heating: Motional heating rate (creation-operator jump).
        gamma_motional_dephasing: Motional dephasing rate (number-operator
            jump).
        gamma_spin_dephasing: Collective spin dephasing rate.
    """

    gamma_heating: float = 0.0
    gamma_motional_dephasing: float = 0.0
    gamma_spin_dephasing: float = 0.0


@dataclass(frozen=True)
class SpinBosonFluctuations:
    """Coherent fluctuation strengths (``system.noise.fluctuations`` schema).

    Each sigma is the standard deviation of a quasi-static (constant over one
    gate) error term; see ``SpinBosonDefinition.fluctuation_terms`` for the
    operator each one multiplies.

    Attributes:
        enabled: When false, ``build_systems`` puts no fluctuation terms on
            the open system.
        sigma_static_spin_dephasing: Collective spin-dephasing offset, in
            rad/s (default 31.4159 = 2*pi*5).
        sigma_static_motional_frequency: Motional-frequency offset multiplying
            the number operator, in rad/s.
        sigma_control_alpha1_relative: Relative (dimensionless) amplitude
            noise on ``alpha1``; the error term scales with ``alpha1(t)``.
        sigma_control_alpha2_relative: Same for ``alpha2``.
    """

    enabled: bool = True
    sigma_static_spin_dephasing: float = 31.4159
    sigma_static_motional_frequency: float = 30.0
    sigma_control_alpha1_relative: float = 0.0001
    sigma_control_alpha2_relative: float = 0.0001


@dataclass(frozen=True)
class SpinBosonNoise:
    """Container splitting ``system.noise`` into its two YAML subsections."""

    decoherence: SpinBosonDecoherence = field(default_factory=SpinBosonDecoherence)
    fluctuations: SpinBosonFluctuations = field(default_factory=SpinBosonFluctuations)


# ---------------------------------------------------------------------------
# Parameterization override
# ---------------------------------------------------------------------------

class Alpha2EndpointZeroParameterization:
    """Parameterization wrapper pinning ``alpha2`` to zero at both endpoints.

    The spin-motion drive must ramp up from zero and return to zero so the
    gate starts and ends with the qubits decoupled from the motion. Rather
    than trusting the optimizer to find this, the constraint is enforced
    structurally: the first and last time steps of control column 1
    (``alpha2``) are clamped to physical amplitude 0 in every representation
    the optimizer touches — physical amplitudes, normalized parameters,
    gradients, and bounds — so those two parameters are simply frozen.

    ``base`` is the affine normalized-parameter map returned by
    ``spin_boson_parameterization`` (normalized value ``p`` in ``[-1, 1]``
    maps to ``center + p * half_width`` per bound pair).
    """

    def __init__(self, base):
        self.base = base

    def to_physical(self, normalized):
        """Map normalized parameters to amplitudes, zeroing alpha2 endpoints."""
        amplitudes = self.base.to_physical(normalized)
        # Index [[0, -1], 1] = first and last time step of the alpha2 column;
        # the same pattern recurs in every method below.
        amplitudes[[0, -1], 1] = 0.0
        return amplitudes

    def to_parameters(self, amplitudes):
        """Map amplitudes to normalized parameters.

        The endpoint entries are overwritten with the normalized value that
        maps back to physical 0, regardless of what the input amplitudes held,
        so round-tripping always lands on the constraint.
        """
        parameters = self.base.to_parameters(amplitudes)
        lower, upper = self.base.bounds_for(amplitudes.shape)
        parameters[[0, -1], 1] = self._normalized_zero(lower[[0, -1], 1], upper[[0, -1], 1])
        return parameters

    def pullback_gradient(self, physical_gradient):
        """Pull the physical gradient back, zeroing the frozen entries.

        With zero gradient the optimizer never proposes moving the endpoint
        parameters, keeping them exactly at the constrained value.
        """
        gradient = self.base.pullback_gradient(physical_gradient)
        gradient[[0, -1], 1] = 0.0
        return gradient

    def parameter_bounds(self, shape):
        """Return flat parameter bounds with the endpoint entries collapsed.

        The two frozen parameters get ``(value, value)`` bounds so a bounded
        optimizer (e.g. L-BFGS-B) treats them as fixed even if numerical
        noise reaches them. Bounds are indexed in the flattened (row-major,
        ``np.ravel_multi_index``) parameter order.
        """
        bounds = self.base.parameter_bounds(shape)
        lower, upper = self.base.bounds_for(shape)
        endpoint_value = self._normalized_zero(lower[[0, -1], 1], upper[[0, -1], 1])
        for row, value in zip((0, shape[0] - 1), endpoint_value):
            bounds[np.ravel_multi_index((row, 1), shape)] = (float(value), float(value))
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

class SpinBosonDefinition(SystemDefinitionBase):
    """System definition registered under ``system.type: spin_boson``.

    Supplies the physics hooks of ``SystemDefinitionBase``; the generic
    ``build_systems`` assembly comes from the base class, while the initial
    pulse and parameterization are overridden for the spin-boson-specific
    start shape and endpoint constraint.
    """

    name = "spin_boson"

    def default_params(self):
        return SpinBosonParams()

    def default_noise(self):
        return SpinBosonNoise()

    def build_closed_system(self, params):
        return spin_boson_control_system(
            n_levels=params.n_levels,
            phi_s=params.phi_s,
            mode_vector=params.mode_vector,
            eta=params.eta,
        )

    def fluctuation_terms(self, params, fluctuations):
        """The four coherent fluctuation terms.

        - ``spin-shift`` (static): collective spin dephasing,
          ``kron(0.5 * (sz x I + I x sz), I_motion)`` — a shot-to-shot qubit
          frequency offset common to both ions.
        - ``motion-shift`` (static): motional-frequency drift,
          ``kron(I_spin, n)``.
        - ``alpha1-rel`` (control): relative amplitude noise on ``alpha1``.
        - ``alpha2-rel`` (control): relative amplitude noise on ``alpha2``.
        """
        n_levels = params.n_levels
        spin_identity = np.eye(4, dtype=complex)
        motion_identity = np.eye(n_levels, dtype=complex)
        number = number_operator(n_levels)
        x1 = 0.5 * (annihilation_operator(n_levels) + creation_operator(n_levels))
        s_phi = two_qubit_spin_phase_mode(params.phi_s, params.mode_vector)

        # The control-kind operators must match the corresponding control
        # Hamiltonian terms exactly (same operator, same eta/mode conventions):
        # the propagator applies them as amplitude * sigma * operator, so any
        # mismatch would silently change the meaning of the relative sigmas.
        return [
            FluctuationTerm(
                kind="static",
                name="spin-shift",
                coefficient=fluctuations.sigma_static_spin_dephasing,
                operator=np.kron(_collective_sz(), motion_identity),
                definition="kron(0.5 * (sz ⊗ I + I ⊗ sz), I_motion)",
                usage="added directly to H_fluctuation",
            ),
            FluctuationTerm(
                kind="static",
                name="motion-shift",
                coefficient=fluctuations.sigma_static_motional_frequency,
                operator=np.kron(spin_identity, number),
                definition="kron(I_spin, number_operator)",
                usage="added directly to H_fluctuation",
            ),
            FluctuationTerm(
                kind="control",
                name="alpha1-rel",
                coefficient=fluctuations.sigma_control_alpha1_relative,
                operator=np.kron(spin_identity, number),
                definition="kron(I_spin, number_operator)",
                usage="alpha1(t) * control[0]",
            ),
            FluctuationTerm(
                kind="control",
                name="alpha2-rel",
                coefficient=fluctuations.sigma_control_alpha2_relative,
                operator=params.eta * np.kron(s_phi, x1),
                definition=(
                    f"eta * kron(S_phi(mode={tuple(params.mode_vector)}), X1), "
                    f"X1=(a + adag)/2, eta={params.eta:.12g}"
                ),
                usage="alpha2(t) * control[1]",
            ),
        ]

    def decoherence_channels(self, params, decoherence):
        """The three Lindblad channels of the trapped-ion model.

        Same operators and ordering as ``spin_boson_collapse_operators``
        (kept above for the legacy scripts); the ``sqrt(gamma)`` scaling
        lives on ``DecoherenceChannel.matrix`` and the base class drops
        zero-rate channels.
        """
        n_levels = params.n_levels
        spin_identity = np.eye(4, dtype=complex)
        motion_identity = np.eye(n_levels, dtype=complex)
        return [
            DecoherenceChannel(
                name="heating",
                rate=decoherence.gamma_heating,
                operator=np.kron(spin_identity, creation_operator(n_levels)),
                definition="kron(I_spin, adag)",
            ),
            DecoherenceChannel(
                name="motion-dephasing",
                rate=decoherence.gamma_motional_dephasing,
                operator=np.kron(spin_identity, number_operator(n_levels)),
                definition="kron(I_spin, number_operator)",
            ),
            DecoherenceChannel(
                name="spin-dephasing",
                rate=decoherence.gamma_spin_dephasing,
                operator=np.kron(_collective_sz(), motion_identity),
                definition="kron(0.5 * (sz ⊗ I + I ⊗ sz), I_motion)",
            ),
        ]

    def control_bounds(self, params):
        alpha1_lower, alpha1_upper = khz_bounds_to_rad_s(params.alpha1_khz_bounds)
        alpha2_lower, alpha2_upper = khz_bounds_to_rad_s(params.alpha2_khz_bounds)
        return (
            np.array([alpha1_lower, alpha2_lower], dtype=float),
            np.array([alpha1_upper, alpha2_upper], dtype=float),
        )

    def build_initial_pulse(self, params, pulse_config):
        """Construct the initial pulse guess from the ``pulse`` config section.

        ``alpha1`` starts at the bounds midpoint shifted by
        ``alpha1_offset_fraction`` of the half-width, plus per-step Gaussian
        noise of ``alpha1_noise_fraction`` half-widths (reproducible when
        ``pulse_config.random_seed`` is set), clipped back into bounds.
        ``alpha2`` starts flat at its upper bound; the endpoint-zero
        constraint is applied later by the parameterization, not here.
        """
        n_steps, dt = validate_pulse_config(pulse_config)
        (alpha1_lower, _), (alpha1_upper, alpha2_upper) = self.control_bounds(params)

        alpha1_center = 0.5 * (alpha1_upper + alpha1_lower)
        alpha1_scale = 0.5 * (alpha1_upper - alpha1_lower)
        # Seedless runs draw from numpy's global RNG (varies run to run);
        # with a seed, a dedicated generator gives reproducible pulses
        # without touching global state.
        alpha1_noise = (
            np.random.randn(n_steps)
            if pulse_config.random_seed is None
            else np.random.default_rng(pulse_config.random_seed).standard_normal(n_steps)
        )
        alpha1 = (
            alpha1_center
            + params.alpha1_offset_fraction * alpha1_scale
            + params.alpha1_noise_fraction * alpha1_scale * alpha1_noise
        )
        alpha1 = np.clip(alpha1, alpha1_lower, alpha1_upper)
        # Flat at the upper bound; the endpoint zeros come from the
        # parameterization, not from the initial guess.
        alpha2 = alpha2_upper * np.ones(n_steps, dtype=float)

        return PiecewiseConstantPulse(
            amplitudes=np.column_stack([alpha1, alpha2]),
            dt=dt,
        )

    def build_parameterization(self, params, pulse):
        """Return the bounded parameterization with alpha2 endpoints frozen."""
        return Alpha2EndpointZeroParameterization(
            spin_boson_parameterization(
                pulse.n_steps,
                alpha1_khz_bounds=params.alpha1_khz_bounds,
                alpha2_khz_bounds=params.alpha2_khz_bounds,
            )
        )

    def target_gate(self, params):
        return resolve_target_gate(params.target_gate)

    def state_pairs(self, params):
        """Expand the spin gate into motion-resolved (initial, target) pairs.

        The 4x4 target acts on the spins only; the fidelity is averaged over
        state pairs resolved across the ``n_levels`` motional levels.
        """
        return motion_resolved_gate_state_pairs(
            self.target_gate(params), params.n_levels
        )

    # ---- presentation hooks --------------------------------------------------

    def control_channels(self, params):
        return [
            ControlChannel(label="alpha1", display_scale=1.0 / RAD_S_PER_KHZ, display_unit="kHz"),
            ControlChannel(label="alpha2", display_scale=1.0 / RAD_S_PER_KHZ, display_unit="kHz"),
        ]

    def population_structure(self, params):
        return PopulationStructure(
            dims=(4, params.n_levels),
            names=("two-qubit spin", "motion"),
            labels=(
                ("|00>", "|01>", "|10>", "|11>"),
                tuple(f"n={level}" for level in range(params.n_levels)),
            ),
        )

    def probe_state_pair(self, params):
        return StateProbe(
            initial_state=basis_state(0, 4 * params.n_levels),
            target_state=ms_bell_target_motion_ground(params.n_levels),
            description="|00>|n=0> -> MS Bell state, motion ground",
        )
