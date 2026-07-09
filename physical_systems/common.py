"""Shared plumbing for system definitions.

``SystemDefinitionBase`` implements the generic half of the system-definition
interface documented in ``physical_systems/__init__.py`` — assembling the
``OpenSystem`` from declared noise terms, YAML noise gating, default
parameterization/initial pulse, and presentation hooks the driver uses for
plots and reports — so a concrete system module only supplies physics: a
closed system (nominal + control Hamiltonians), optional noise terms, and the
fidelity definition. See ``physical_systems/spin_boson.py`` for the reference
subclass and ``physical_systems/README.md`` for the walkthrough.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import numpy as np

from quantum_control import BoundedAmplitudeParameterization
from quantum_control.pulses.pulse import PiecewiseConstantPulse
from quantum_control.systems import OpenSystem


def _float_fields_by_name(config):
    """All float fields of ``config`` (everything except ``enabled``) by name."""
    return {
        field.name: float(getattr(config, field.name))
        for field in dataclasses.fields(config)
        if field.name != "enabled"
    }


@dataclass(frozen=True)
class FluctuationsConfigBase:
    """Base for ``system.noise.fluctuations`` dataclasses.

    Subclasses add their sigma fields; a closed-only system can use this base
    directly (nothing enabled, no fields). ``enabled`` plus the generic
    ``any_sigma_positive`` gate let the driver skip the more expensive
    fluctuation-expansion propagation whenever it cannot change the result,
    mirroring ``DecoherenceConfigBase``.
    """

    enabled: bool = False

    @property
    def sigmas(self):
        """All float fields (i.e. everything except ``enabled``) by name."""
        return _float_fields_by_name(self)

    @property
    def any_sigma_positive(self):
        return any(sigma > 0.0 for sigma in self.sigmas.values())


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
        return _float_fields_by_name(self)

    @property
    def any_rate_positive(self):
        return any(rate > 0.0 for rate in self.rates.values())


@dataclass(frozen=True)
class NoiseConfigBase:
    """Default ``system.noise`` container: both noise types, both disabled.

    Systems with their own sigma/rate fields define their own container with
    the same two-section shape (see ``SpinBosonNoise``); a closed-only system
    needs no noise dataclasses at all.
    """

    decoherence: DecoherenceConfigBase = field(default_factory=DecoherenceConfigBase)
    fluctuations: FluctuationsConfigBase = field(default_factory=FluctuationsConfigBase)


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

        name                                registry key
        default_params()                    frozen params dataclass
        build_closed_system(params)         ClosedSystem (nominal + controls)
        control_bounds(params)              (lower, upper) rad/s arrays
        state_pairs(params)                 weighted StatePair tuple (the
                                            fidelity definition)

    and may override::

        default_noise()                     noise config (default: everything
                                            disabled via NoiseConfigBase)
        fluctuation_terms(params, fluctuations)
                                            list of FluctuationTerm (default:
                                            none)
        decoherence_channels(params, decoherence)
                                            list of DecoherenceChannel
                                            (default: none)
        target_gate(params)                 target unitary (default: None)
        build_parameterization / build_initial_pulse
                                            (defaults derived from
                                            control_bounds)
        control_channels / population_structure / probe_state_pair
                                            presentation hooks
    """

    name = None

    # ---- physics hooks (subclass responsibilities) -------------------------

    def build_closed_system(self, params):
        """The noiseless system: nominal + control Hamiltonians only."""
        raise NotImplementedError

    def control_bounds(self, params):
        raise NotImplementedError

    def state_pairs(self, params):
        """Weighted ``StatePair`` average defining the gate fidelity."""
        raise NotImplementedError

    def fluctuation_terms(self, params, fluctuations):
        """Declarative quasi-static fluctuation terms (default: none)."""
        return []

    def decoherence_channels(self, params, decoherence):
        """Declarative Lindblad channels (default: none)."""
        return []

    def default_noise(self):
        """Noise config defaults; closed-only systems keep everything off."""
        return NoiseConfigBase()

    def target_gate(self, params):
        """Target unitary, if the system has one (informational)."""
        return None

    # ---- generic driver interface ------------------------------------------

    def build_systems(self, params, noise):
        """Return ``(closed_system, open_system)``.

        The open system is the closed system plus the noise terms selected by
        the ``noise`` config: fluctuation terms when
        ``noise.fluctuations.enabled`` and at least one sigma is positive,
        positive-rate decoherence channels when ``noise.decoherence.enabled``.
        With everything disabled the open system carries no terms and evolves
        identically to the closed one.

        The closed system must expose ``drift`` and ``controls`` (i.e. be a
        ``ClosedSystem``-shaped dataclass) so the terms can be attached
        generically.
        """
        closed_system = self.build_closed_system(params)
        noise_terms = []
        fluctuations = noise.fluctuations
        if fluctuations.enabled and fluctuations.any_sigma_positive:
            # All-or-nothing, unlike the per-channel decoherence filter below:
            # control-kind terms align with control channels positionally, so
            # dropping an individual zero-sigma term would misalign the rest.
            noise_terms.extend(self.fluctuation_terms(params, fluctuations))
        decoherence = noise.decoherence
        if decoherence.enabled and decoherence.any_rate_positive:
            noise_terms.extend(
                channel
                for channel in self.decoherence_channels(params, decoherence)
                if channel.rate > 0.0
            )
        open_system = OpenSystem(
            drift=closed_system.drift,
            controls=closed_system.controls,
            noise_terms=tuple(noise_terms),
        )
        return closed_system, open_system

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
