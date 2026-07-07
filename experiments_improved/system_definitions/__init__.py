"""Registry of physical systems available to the experiment driver.

The experiment config selects a system by name (``system.type`` in the YAML
file); the system's own module defines what its ``system.params`` and
``system.noise`` sections look like and how to build the quantum_control
objects from them. A system definition is any object providing:

- ``name``: registry key (``str``)
- ``default_params()`` / ``default_noise()``: frozen dataclasses whose fields
  define the ``system.params`` / ``system.noise`` YAML schema
- ``build_systems(params, noise)`` -> ``(system, noisy_system, noise_specs)``
- ``build_collapse_operators(params, noise)`` -> list of jump operators
- ``build_initial_pulse(params, pulse_config)`` -> ``PiecewiseConstantPulse``
- ``build_parameterization(params, pulse)`` -> pulse parameterization
- ``target_gate(params)`` / ``state_pairs(params)`` -> target unitary and the
  weighted ``StatePair`` tuple for the gate average
- presentation hooks ``control_channels`` / ``population_structure`` /
  ``probe_state_pair`` (optional; the driver skips what is absent)

In practice, subclass ``SystemDefinitionBase`` from ``system_definitions/common.py``,
which implements everything generic from a handful of physics hooks.

Invariant: the driver and quantum_control core never import concrete system
modules; all system specifics flow through this registry.

Not to be confused with ``quantum_control/systems/``: that package holds the
reusable physics library (System classes, Hamiltonian/operator builders such
as ``spin_boson_control_system``) consumed by both the legacy ``experiments/``
scripts and the definitions here. This package holds the driver-facing
adapters that assemble those library pieces per the YAML config.

See ``system_definitions/README.md`` for a walkthrough of adding a new system
(e.g. an NV center or a Rydberg array) and ``system_definitions/spin_boson.py`` for the
reference implementation.
"""

from __future__ import annotations

SYSTEM_REGISTRY = {}


def register_system(definition):
    name = definition.name
    if name in SYSTEM_REGISTRY:
        raise ValueError(f"system {name!r} is already registered.")
    SYSTEM_REGISTRY[name] = definition
    return definition


def get_system(name):
    try:
        return SYSTEM_REGISTRY[name]
    except KeyError:
        registered = ", ".join(sorted(SYSTEM_REGISTRY)) or "<none>"
        raise ValueError(
            f"unknown system type {name!r}; registered systems: {registered}."
        ) from None


from experiments_improved.system_definitions import spin_boson as _spin_boson  # noqa: E402

register_system(_spin_boson.SpinBosonDefinition())
