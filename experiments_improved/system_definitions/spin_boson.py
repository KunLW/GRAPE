"""Spin-boson (trapped-ion MS gate) system definition for the experiment driver.

Reference implementation of the system-definition interface documented in
``system_definitions/__init__.py``: the physics hooks of ``SystemDefinitionBase`` (see
``system_definitions/common.py``) are filled in here, while the generic plumbing —
noise-spec bookkeeping, decoherence gating, presentation defaults — lives in
the base class. The ``SpinBosonParams`` / ``SpinBosonNoise`` dataclasses
define the ``system.params`` / ``system.noise`` YAML schema.

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
    DEFAULT_ALPHA1_KHZ_BOUNDS,
    DEFAULT_ALPHA2_KHZ_BOUNDS,
    DEFAULT_LAMB_DICKE_ETA,
    RAD_S_PER_KHZ,
    annihilation_operator,
    creation_operator,
    khz_bounds_to_rad_s,
    motion_resolved_gate_state_pairs,
    ms_xx_pi_over_2_gate,
    number_operator,
    spin_boson_control_system,
    spin_boson_parameterization,
    two_qubit_spin_phase_mode,
)
from quantum_control.pulses.pulse import PiecewiseConstantPulse

from experiments_improved.system_definitions.common import (
    ControlChannel,
    DecoherenceChannel,
    DecoherenceConfigBase,
    NoiseTerm,
    PopulationStructure,
    StateProbe,
    SystemDefinitionBase,
    basis_state,
    validate_pulse_config,
)

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
    gate) error term; see ``SpinBosonDefinition.noise_terms`` for the
    operator each one multiplies.

    Attributes:
        enabled: When false, ``build_systems`` returns the ideal system for
            both the nominal and the noisy slot and no noise specs.
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


class SpinBosonDefinition(SystemDefinitionBase):
    """System definition registered under ``system.type: spin_boson``.

    Supplies the physics hooks of ``SystemDefinitionBase``; the generic
    ``build_systems`` / ``build_collapse_operators`` come from the base
    class, while the initial pulse and parameterization are overridden for
    the spin-boson-specific start shape and endpoint constraint.
    """

    name = "spin_boson"

    def default_params(self):
        return SpinBosonParams()

    def default_noise(self):
        return SpinBosonNoise()

    def build_nominal_system(self, params, static_fluctuations=(), control_fluctuations=()):
        return spin_boson_control_system(
            n_levels=params.n_levels,
            phi_s=params.phi_s,
            mode_vector=params.mode_vector,
            eta=params.eta,
            static_fluctuations=static_fluctuations,
            control_fluctuations=control_fluctuations,
        )

    def noise_terms(self, params, fluctuations):
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
        single_identity = np.eye(2, dtype=complex)
        sz = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
        sz1_plus_sz2 = np.kron(sz, single_identity) + np.kron(single_identity, sz)
        number = number_operator(n_levels)
        x1 = 0.5 * (annihilation_operator(n_levels) + creation_operator(n_levels))
        s_phi = two_qubit_spin_phase_mode(params.phi_s, params.mode_vector)

        # The control-kind operators must match the corresponding control
        # Hamiltonian terms exactly (same operator, same eta/mode conventions):
        # the propagator applies them as amplitude * sigma * operator, so any
        # mismatch would silently change the meaning of the relative sigmas.
        return [
            NoiseTerm(
                kind="static",
                name="spin-shift",
                coefficient=fluctuations.sigma_static_spin_dephasing,
                operator=np.kron(0.5 * sz1_plus_sz2, motion_identity),
                definition="kron(0.5 * (sz ⊗ I + I ⊗ sz), I_motion)",
                usage="added directly to H_fluctuation",
            ),
            NoiseTerm(
                kind="static",
                name="motion-shift",
                coefficient=fluctuations.sigma_static_motional_frequency,
                operator=np.kron(spin_identity, number),
                definition="kron(I_spin, number_operator)",
                usage="added directly to H_fluctuation",
            ),
            NoiseTerm(
                kind="control",
                name="alpha1-rel",
                coefficient=fluctuations.sigma_control_alpha1_relative,
                operator=np.kron(spin_identity, number),
                definition="kron(I_spin, number_operator)",
                usage="alpha1(t) * control[0]",
            ),
            NoiseTerm(
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

        Same operators and ordering as ``spin_boson_collapse_operators`` in
        quantum_control (kept there for legacy callers); the base class
        applies the ``sqrt(gamma)`` scaling and drops zero-rate channels.
        """
        n_levels = params.n_levels
        spin_identity = np.eye(4, dtype=complex)
        single_identity = np.eye(2, dtype=complex)
        motion_identity = np.eye(n_levels, dtype=complex)
        sz = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
        sz_collective = 0.5 * (
            np.kron(sz, single_identity) + np.kron(single_identity, sz)
        )
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
                operator=np.kron(sz_collective, motion_identity),
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
