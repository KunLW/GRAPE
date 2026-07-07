"""Shared plumbing for system definitions.

``SystemDefinitionBase`` implements the generic half of the system-definition
interface documented in ``system_definitions/__init__.py`` — noise-spec bookkeeping,
decoherence gating, default parameterization/initial pulse, and presentation
hooks the driver uses for plots and reports — so a concrete system module
only supplies physics: operators, Hamiltonians, noise terms, and targets.
See ``system_definitions/spin_boson.py`` for the reference subclass and
``system_definitions/README.md`` for the walkthrough.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np

from quantum_control import BoundedAmplitudeParameterization
from quantum_control.pulses.pulse import PiecewiseConstantPulse


@dataclass(frozen=True)
class NoiseTerm:
    """One coherent fluctuation term of the noisy Hamiltonian.

    ``kind`` is ``"static"`` (added to H as-is) or ``"control"`` (scaled by
    the instantaneous control amplitude at propagation time, which makes the
    coefficient a *relative* error). ``coefficient`` is the standard
    deviation sigma; ``definition``/``usage`` are human-readable strings
    carried into the optimization report.
    """

    kind: str
    name: str
    coefficient: float
    operator: np.ndarray
    definition: str
    usage: str

    def __post_init__(self):
        if self.kind not in ("static", "control"):
            raise ValueError(f"noise term kind must be 'static' or 'control', got {self.kind!r}.")
        object.__setattr__(self, "coefficient", float(self.coefficient))
        object.__setattr__(self, "operator", np.asarray(self.operator, dtype=complex))

    @property
    def matrix(self):
        """The already-scaled ``sigma * operator`` handed to the system builder."""
        return self.coefficient * self.operator

    def as_spec(self):
        """Dict form consumed by the driver's Noise Terms report table."""
        return {
            "kind": self.kind,
            "name": self.name,
            "coefficient": self.coefficient,
            "operator": self.operator,
            "definition": self.definition,
            "usage": self.usage,
            "matrix": self.matrix,
        }


@dataclass(frozen=True)
class DecoherenceChannel:
    """One Lindblad channel, declared unscaled (parallel to ``NoiseTerm``).

    ``rate`` is gamma in 1/s; the jump operator handed to the decoherence
    correction is ``matrix = sqrt(rate) * operator`` (unlike ``NoiseTerm``,
    whose coefficient multiplies linearly). ``definition`` is the
    human-readable operator description for the report.
    """

    name: str
    rate: float
    operator: np.ndarray
    definition: str

    def __post_init__(self):
        rate = float(self.rate)
        if rate < 0.0:
            raise ValueError("decoherence rates must be non-negative.")
        object.__setattr__(self, "rate", rate)
        object.__setattr__(self, "operator", np.asarray(self.operator, dtype=complex))

    @property
    def matrix(self):
        """The scaled jump operator ``L = sqrt(rate) * operator``."""
        return np.sqrt(self.rate) * self.operator


@dataclass(frozen=True)
class DecoherenceConfigBase:
    """Base for ``system.noise.decoherence`` dataclasses.

    Subclasses add their rate fields (floats, 1/s); ``enabled`` plus the
    generic ``any_rate_positive`` gate lets the driver skip the more
    expensive decoherence-corrected propagation whenever it cannot change
    the result.
    """

    enabled: bool = False

    @property
    def rates(self):
        """All float fields (i.e. everything except ``enabled``) by name."""
        return {
            field.name: float(getattr(self, field.name))
            for field in dataclasses.fields(self)
            if field.name != "enabled"
        }

    @property
    def any_rate_positive(self):
        return any(rate > 0.0 for rate in self.rates.values())


@dataclass(frozen=True)
class ControlChannel:
    """Presentation metadata for one control channel (plots, CSV export)."""

    label: str
    display_scale: float = 1.0
    display_unit: str = "rad/s"


@dataclass(frozen=True)
class PopulationStructure:
    """Bipartite structure for the driver's population-marginal plot.

    The state vector is reshaped to ``(n_times, *dims)`` and summed over one
    axis at a time; ``labels`` are the fully formatted legend entries per
    subsystem (``len(labels[i]) == dims[i]``).
    """

    dims: tuple[int, int]
    names: tuple[str, str]
    labels: tuple[tuple[str, ...], tuple[str, ...]]

    def __post_init__(self):
        for dim, subsystem_labels in zip(self.dims, self.labels):
            if len(subsystem_labels) != dim:
                raise ValueError("each subsystem needs exactly one label per level.")


@dataclass(frozen=True)
class StateProbe:
    """A single (initial, target) state pair for the driver's extra fidelity
    diagnostic and propagation plot."""

    initial_state: np.ndarray
    target_state: np.ndarray
    description: str = ""


class SystemDefinitionBase:
    """Generic half of the system-definition interface.

    Subclasses must provide::

        name                                        registry key
        default_params() / default_noise()          frozen config dataclasses
        build_nominal_system(params, static_fluctuations=(),
                             control_fluctuations=())
        noise_terms(params, fluctuations)           list of NoiseTerm
        control_bounds(params)                      (lower, upper) rad/s arrays
        target_gate(params)                         target unitary
        state_pairs(params)                         weighted StatePair tuple

    and may override::

        decoherence_channels(params, decoherence)   list of DecoherenceChannel
                                                    (default: none)
        build_parameterization / build_initial_pulse (defaults derived from
                                                     control_bounds)
        control_channels / population_structure / probe_state_pair
                                                    presentation hooks
    """

    name = None

    # ---- physics hooks (subclass responsibilities) -------------------------

    def build_nominal_system(self, params, static_fluctuations=(), control_fluctuations=()):
        raise NotImplementedError

    def noise_terms(self, params, fluctuations):
        raise NotImplementedError

    def control_bounds(self, params):
        raise NotImplementedError

    def decoherence_channels(self, params, decoherence):
        """Declarative Lindblad channels, parallel to ``noise_terms``."""
        return []

    # ---- generic driver interface ------------------------------------------

    def build_systems(self, params, noise):
        """Return ``(system, noisy_system, noise_specs)``.

        With fluctuations disabled the ideal system fills both slots and the
        spec list is empty; otherwise the noisy system carries the
        already-scaled static/control matrices from ``noise_terms``.
        """
        system = self.build_nominal_system(params)
        if not noise.fluctuations.enabled:
            return system, system, []
        terms = list(self.noise_terms(params, noise.fluctuations))
        noisy_system = self.build_nominal_system(
            params,
            static_fluctuations=[term.matrix for term in terms if term.kind == "static"],
            control_fluctuations=[term.matrix for term in terms if term.kind == "control"],
        )
        return system, noisy_system, [term.as_spec() for term in terms]

    def build_decoherence_channels(self, params, noise):
        """Validated, active Lindblad channels; empty when gated off.

        Applies the ``enabled``/``any_rate_positive`` gate and drops
        zero-rate channels, so the returned list contains exactly the
        channels that contribute to the decoherence correction.
        """
        decoherence = noise.decoherence
        if not (decoherence.enabled and decoherence.any_rate_positive):
            return []
        return [
            channel
            for channel in self.decoherence_channels(params, decoherence)
            if channel.rate > 0.0
        ]

    def build_collapse_operators(self, params, noise):
        """Scaled jump operators for the Lindblad correction; empty means "skip"."""
        return [
            channel.matrix
            for channel in self.build_decoherence_channels(params, noise)
        ]

    def build_parameterization(self, params, pulse):
        lower, upper = self.control_bounds(params)
        return BoundedAmplitudeParameterization(
            lower=np.asarray(lower, dtype=float),
            upper=np.asarray(upper, dtype=float),
        )

    def build_initial_pulse(self, params, pulse_config):
        """Default guess: every control flat at the midpoint of its bounds.

        ``pulse_config.random_seed`` is ignored here; systems wanting a
        randomized start override this method.
        """
        n_steps, dt = validate_pulse_config(pulse_config)
        lower, upper = self.control_bounds(params)
        center = 0.5 * (np.asarray(upper, dtype=float) + np.asarray(lower, dtype=float))
        return PiecewiseConstantPulse(
            amplitudes=np.tile(center, (n_steps, 1)),
            dt=dt,
        )

    # ---- presentation hooks (safe defaults) ---------------------------------

    def control_channels(self, params):
        lower, _upper = self.control_bounds(params)
        n_controls = np.asarray(lower, dtype=float).size
        return [ControlChannel(label=f"control[{i}]") for i in range(n_controls)]

    def population_structure(self, params):
        """Bipartite population-plot structure, or ``None`` to skip the plot."""
        return None

    def probe_state_pair(self, params):
        """Extra single-state fidelity probe, or ``None`` to skip it."""
        return None


def validate_pulse_config(pulse_config):
    """Validate the generic pulse section; return ``(n_steps, dt_seconds)``."""
    n_steps = int(pulse_config.n_steps)
    if n_steps < 1:
        raise ValueError("n_steps must be at least 1.")
    total_time = float(pulse_config.total_time_us) * 1e-6
    if total_time <= 0.0:
        raise ValueError("total_time_us must be positive.")
    return n_steps, total_time / n_steps


def basis_state(index, dimension):
    state = np.zeros(dimension, dtype=complex)
    state[index] = 1.0
    return state
