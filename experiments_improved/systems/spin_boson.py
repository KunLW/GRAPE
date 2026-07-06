"""Spin-boson (trapped-ion MS gate) system definition for the experiment driver.

Reference implementation of the system-definition interface documented in
``systems/__init__.py``. The ``SpinBosonParams`` / ``SpinBosonNoise``
dataclasses define the ``system.params`` / ``system.noise`` YAML schema; the
builder methods produce the quantum_control objects the driver consumes.

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
    annihilation_operator,
    creation_operator,
    motion_resolved_gate_state_pairs,
    ms_xx_pi_over_2_gate,
    number_operator,
    spin_boson_collapse_operators,
    spin_boson_control_system,
    spin_boson_parameterization,
    two_qubit_spin_phase_mode,
)
from quantum_control.pulses.pulse import PiecewiseConstantPulse

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
class SpinBosonDecoherence:
    """Lindblad decoherence rates (``system.noise.decoherence`` YAML schema).

    All rates are in 1/s and feed ``spin_boson_collapse_operators``.

    Attributes:
        enabled: Master switch; when false the rates are ignored entirely.
        gamma_heating: Motional heating rate (creation-operator jump).
        gamma_motional_dephasing: Motional dephasing rate (number-operator
            jump).
        gamma_spin_dephasing: Collective spin dephasing rate.
    """

    enabled: bool = False
    gamma_heating: float = 0.0
    gamma_motional_dephasing: float = 0.0
    gamma_spin_dephasing: float = 0.0

    @property
    def any_rate_positive(self):
        """True if at least one rate is positive.

        Lets the driver skip building collapse operators when decoherence is
        nominally enabled but all rates are zero, so the more expensive
        corrected propagation only runs when it changes the result.
        """
        return (
            self.gamma_heating > 0.0
            or self.gamma_motional_dephasing > 0.0
            or self.gamma_spin_dephasing > 0.0
        )


@dataclass(frozen=True)
class SpinBosonFluctuations:
    """Coherent fluctuation strengths (``system.noise.fluctuations`` schema).

    Each sigma is the standard deviation of a quasi-static (constant over one
    gate) error term; see ``spin_boson_noise_term_specs`` for the operator
    each one multiplies.

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


def spin_boson_noise_term_specs(params, fluctuations):
    """Build the four coherent fluctuation terms as self-describing specs.

    Returns a list of dicts (see ``_noise_term_spec`` for the keys), one per
    error channel:

    - ``spin-shift`` (static): collective spin dephasing,
      ``kron(0.5 * (sz x I + I x sz), I_motion)`` — a shot-to-shot qubit
      frequency offset common to both ions.
    - ``motion-shift`` (static): motional-frequency drift,
      ``kron(I_spin, n)``.
    - ``alpha1-rel`` (control): relative amplitude noise on ``alpha1``; same
      operator as the ``alpha1`` control term, scaled by ``alpha1(t)`` at
      propagation time.
    - ``alpha2-rel`` (control): relative amplitude noise on ``alpha2``; same
      operator as the ``alpha2`` drive term ``eta * kron(S_phi, X1)``.

    Static terms are added to the Hamiltonian as-is; control terms are
    multiplied by the instantaneous control amplitude, which is what makes
    their sigmas *relative* errors. The ``definition``/``usage`` strings are
    carried through to the optimization report so it can document exactly
    what noise model was applied.
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
        _noise_term_spec(
            kind="static",
            name="spin-shift",
            coefficient=fluctuations.sigma_static_spin_dephasing,
            operator=np.kron(0.5 * sz1_plus_sz2, motion_identity),
            definition="kron(0.5 * (sz ⊗ I + I ⊗ sz), I_motion)",
            usage="added directly to H_fluctuation",
        ),
        _noise_term_spec(
            kind="static",
            name="motion-shift",
            coefficient=fluctuations.sigma_static_motional_frequency,
            operator=np.kron(spin_identity, number),
            definition="kron(I_spin, number_operator)",
            usage="added directly to H_fluctuation",
        ),
        _noise_term_spec(
            kind="control",
            name="alpha1-rel",
            coefficient=fluctuations.sigma_control_alpha1_relative,
            operator=np.kron(spin_identity, number),
            definition="kron(I_spin, number_operator)",
            usage="alpha1(t) * control[0]",
        ),
        _noise_term_spec(
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


def _noise_term_spec(kind, name, coefficient, operator, definition, usage):
    """Normalize one noise term into the spec dict consumed by the driver.

    ``matrix`` is the pre-multiplied ``coefficient * operator`` actually
    handed to ``spin_boson_control_system``; ``operator`` and ``coefficient``
    are kept separately (with the human-readable ``definition``/``usage``)
    for reporting.
    """
    coefficient = float(coefficient)
    operator = np.asarray(operator, dtype=complex)
    return {
        "kind": kind,
        "name": name,
        "coefficient": coefficient,
        "operator": operator,
        "definition": definition,
        "usage": usage,
        "matrix": coefficient * operator,
    }


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
    maps to ``center + p * half_width`` per bound pair). Note this wrapper
    reaches into ``base._bounds_for`` to recover the physical bounds arrays,
    so it is coupled to that implementation.
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
        lower, upper = self.base._bounds_for(amplitudes.shape)
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
        lower, upper = self.base._bounds_for(shape)
        endpoint_value = self._normalized_zero(lower[[0, -1], 1], upper[[0, -1], 1])
        for row, value in zip((0, shape[0] - 1), endpoint_value):
            bounds[np.ravel_multi_index((row, 1), shape)] = (float(value), float(value))
        return bounds

    @staticmethod
    def _normalized_zero(lower, upper):
        """Invert the affine map: the normalized value whose physical image is 0."""
        return (0.0 - 0.5 * (upper + lower)) / (0.5 * (upper - lower))


def _khz_bounds_to_rad_s(bounds):
    """Convert a ``(lower, upper)`` kHz bound pair to rad/s, validating order."""
    lower, upper = np.asarray(bounds, dtype=float)
    if upper <= lower:
        raise ValueError("upper bounds must be greater than lower bounds.")
    return 2.0 * np.pi * 1000.0 * lower, 2.0 * np.pi * 1000.0 * upper


class SpinBosonDefinition:
    """System definition registered under ``system.type: spin_boson``.

    Implements the driver interface documented in ``systems/__init__.py``;
    each ``build_*`` method receives the (possibly YAML-overridden) dataclass
    instances from ``default_params()`` / ``default_noise()``.
    """

    name = "spin_boson"

    def default_params(self):
        return SpinBosonParams()

    def default_noise(self):
        return SpinBosonNoise()

    def build_systems(self, params, noise):
        """Return ``(system, noisy_system, noise_specs)``.

        ``system`` is the ideal control system used for the nominal fidelity;
        ``noisy_system`` additionally carries the static/control fluctuation
        matrices for the robustness terms. With fluctuations disabled the
        same ideal system is returned in both slots and ``noise_specs`` is
        empty.
        """
        system = spin_boson_control_system(
            n_levels=params.n_levels,
            phi_s=params.phi_s,
            mode_vector=params.mode_vector,
            eta=params.eta,
        )
        if not noise.fluctuations.enabled:
            return system, system, []

        noise_specs = spin_boson_noise_term_specs(params, noise.fluctuations)
        noisy_system = spin_boson_control_system(
            n_levels=params.n_levels,
            phi_s=params.phi_s,
            mode_vector=params.mode_vector,
            eta=params.eta,
            static_fluctuations=[
                spec["matrix"] for spec in noise_specs if spec["kind"] == "static"
            ],
            control_fluctuations=[
                spec["matrix"] for spec in noise_specs if spec["kind"] == "control"
            ],
        )
        return system, noisy_system, noise_specs

    def build_collapse_operators(self, params, noise):
        """Return Lindblad jump operators for the decoherence correction.

        Empty when decoherence is disabled or all rates are zero, which the
        driver takes as "no decoherence correction".
        """
        decoherence = noise.decoherence
        if not (decoherence.enabled and decoherence.any_rate_positive):
            return []
        return spin_boson_collapse_operators(
            params.n_levels,
            gamma_heating=decoherence.gamma_heating,
            gamma_motional_dephasing=decoherence.gamma_motional_dephasing,
            gamma_spin_dephasing=decoherence.gamma_spin_dephasing,
        )

    def build_initial_pulse(self, params, pulse_config):
        """Construct the initial pulse guess from the ``pulse`` config section.

        ``alpha1`` starts at the bounds midpoint shifted by
        ``alpha1_offset_fraction`` of the half-width, plus per-step Gaussian
        noise of ``alpha1_noise_fraction`` half-widths (reproducible when
        ``pulse_config.random_seed`` is set), clipped back into bounds.
        ``alpha2`` starts flat at its upper bound; the endpoint-zero
        constraint is applied later by the parameterization, not here.
        Durations arrive in microseconds and are converted to seconds to
        match the rad/s amplitudes.
        """
        n_steps = pulse_config.n_steps
        if n_steps < 1:
            raise ValueError("n_steps must be at least 1.")
        total_time = float(pulse_config.total_time_us) * 1e-6
        if total_time <= 0.0:
            raise ValueError("total_time_us must be positive.")

        alpha1_lower, alpha1_upper = _khz_bounds_to_rad_s(params.alpha1_khz_bounds)
        alpha2_lower, alpha2_upper = _khz_bounds_to_rad_s(params.alpha2_khz_bounds)
        dt = total_time / n_steps

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
