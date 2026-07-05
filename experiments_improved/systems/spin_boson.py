"""Spin-boson (trapped-ion MS gate) system definition for the experiment driver.

Reference implementation of the system-definition interface documented in
``systems/__init__.py``. The ``SpinBosonParams`` / ``SpinBosonNoise``
dataclasses define the ``system.params`` / ``system.noise`` YAML schema; the
builder methods produce the quantum_control objects the driver consumes.
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

TARGET_GATES = {
    "ms_xx_pi_over_2": ms_xx_pi_over_2_gate,
}


def resolve_target_gate(name):
    try:
        builder = TARGET_GATES[name]
    except KeyError:
        registered = ", ".join(sorted(TARGET_GATES))
        raise ValueError(
            f"unknown target_gate {name!r}; registered gates: {registered}."
        ) from None
    gate = np.asarray(builder(), dtype=complex)
    deviation = np.linalg.norm(gate.conj().T @ gate - np.eye(gate.shape[0]))
    if deviation > 1e-10:
        raise ValueError(f"target_gate {name!r} is not unitary (deviation {deviation:.3g}).")
    return gate


@dataclass(frozen=True)
class SpinBosonParams:
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
    enabled: bool = False
    gamma_heating: float = 0.0
    gamma_motional_dephasing: float = 0.0
    gamma_spin_dephasing: float = 0.0

    @property
    def any_rate_positive(self):
        return (
            self.gamma_heating > 0.0
            or self.gamma_motional_dephasing > 0.0
            or self.gamma_spin_dephasing > 0.0
        )


@dataclass(frozen=True)
class SpinBosonFluctuations:
    enabled: bool = True
    sigma_static_spin_dephasing: float = 31.4159
    sigma_static_motional_frequency: float = 30.0
    sigma_control_alpha1_relative: float = 0.0001
    sigma_control_alpha2_relative: float = 0.0001


@dataclass(frozen=True)
class SpinBosonNoise:
    decoherence: SpinBosonDecoherence = field(default_factory=SpinBosonDecoherence)
    fluctuations: SpinBosonFluctuations = field(default_factory=SpinBosonFluctuations)


def spin_boson_noise_term_specs(params, fluctuations):
    n_levels = params.n_levels
    spin_identity = np.eye(4, dtype=complex)
    motion_identity = np.eye(n_levels, dtype=complex)
    single_identity = np.eye(2, dtype=complex)
    sz = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    sz1_plus_sz2 = np.kron(sz, single_identity) + np.kron(single_identity, sz)
    number = number_operator(n_levels)
    x1 = 0.5 * (annihilation_operator(n_levels) + creation_operator(n_levels))
    s_phi = two_qubit_spin_phase_mode(params.phi_s, params.mode_vector)

    return [
        _noise_term_spec(
            kind="static",
            name="static[0]",
            coefficient=fluctuations.sigma_static_spin_dephasing,
            operator=np.kron(0.5 * sz1_plus_sz2, motion_identity),
            definition="kron(0.5 * (sz ⊗ I + I ⊗ sz), I_motion)",
            usage="added directly to H_fluctuation",
        ),
        _noise_term_spec(
            kind="static",
            name="static[1]",
            coefficient=fluctuations.sigma_static_motional_frequency,
            operator=np.kron(spin_identity, number),
            definition="kron(I_spin, number_operator)",
            usage="added directly to H_fluctuation",
        ),
        _noise_term_spec(
            kind="control",
            name="control[0]",
            coefficient=fluctuations.sigma_control_alpha1_relative,
            operator=np.kron(spin_identity, number),
            definition="kron(I_spin, number_operator)",
            usage="alpha1(t) * control[0]",
        ),
        _noise_term_spec(
            kind="control",
            name="control[1]",
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
    def __init__(self, base):
        self.base = base

    def to_physical(self, normalized):
        amplitudes = self.base.to_physical(normalized)
        amplitudes[[0, -1], 1] = 0.0
        return amplitudes

    def to_parameters(self, amplitudes):
        parameters = self.base.to_parameters(amplitudes)
        lower, upper = self.base._bounds_for(amplitudes.shape)
        parameters[[0, -1], 1] = self._normalized_zero(lower[[0, -1], 1], upper[[0, -1], 1])
        return parameters

    def pullback_gradient(self, physical_gradient):
        gradient = self.base.pullback_gradient(physical_gradient)
        gradient[[0, -1], 1] = 0.0
        return gradient

    def parameter_bounds(self, shape):
        bounds = self.base.parameter_bounds(shape)
        lower, upper = self.base._bounds_for(shape)
        endpoint_value = self._normalized_zero(lower[[0, -1], 1], upper[[0, -1], 1])
        for row, value in zip((0, shape[0] - 1), endpoint_value):
            bounds[np.ravel_multi_index((row, 1), shape)] = (float(value), float(value))
        return bounds

    @staticmethod
    def _normalized_zero(lower, upper):
        return (0.0 - 0.5 * (upper + lower)) / (0.5 * (upper - lower))


def _khz_bounds_to_rad_s(bounds):
    lower, upper = np.asarray(bounds, dtype=float)
    if upper <= lower:
        raise ValueError("upper bounds must be greater than lower bounds.")
    return 2.0 * np.pi * 1000.0 * lower, 2.0 * np.pi * 1000.0 * upper


class SpinBosonDefinition:
    name = "spin_boson"

    def default_params(self):
        return SpinBosonParams()

    def default_noise(self):
        return SpinBosonNoise()

    def build_systems(self, params, noise):
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
        alpha2 = alpha2_upper * np.ones(n_steps, dtype=float)

        return PiecewiseConstantPulse(
            amplitudes=np.column_stack([alpha1, alpha2]),
            dt=dt,
        )

    def build_parameterization(self, params, pulse):
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
        return motion_resolved_gate_state_pairs(
            self.target_gate(params), params.n_levels
        )
