"""Auto-discovering registry of physical systems.

Every non-underscore module in this package is imported at package-import
time and each concrete ``SystemDefinitionBase`` subclass with a ``name`` is
registered, so **adding a system = adding one module file here** (plus a YAML
config that selects it via ``system.type``). No registry edit is needed.

The experiment config selects a system by name (``system.type`` in the YAML
file); the system's own module defines what its ``system.params`` and
``system.noise`` sections look like and how to build the quantum_control
objects from them. See ``physical_systems/README.md`` for the two-file
walkthrough (e.g. an NV center or a Rydberg array) and
``physical_systems/spin_boson.py`` for the reference implementation; the
generic plumbing lives in ``SystemDefinitionBase``
(``physical_systems/common.py``).

Invariant: the driver and quantum_control core never import concrete system
modules; all system specifics flow through this registry.

Not to be confused with ``quantum_control/systems/``: that package holds the
engine's generic system model (``ClosedSystem``, ``OpenSystem``, and the
``NoiseTerm`` vocabulary). This package holds the concrete physics plus the
driver-facing adapters that assemble those pieces per the YAML config.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil

from physical_systems.common import SystemDefinitionBase

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


def _discover_systems():
    """Import every sibling module and register its definition classes.

    A module participates by defining a ``SystemDefinitionBase`` subclass
    with a non-None ``name``; underscore-prefixed modules are skipped.
    """
    for module_info in pkgutil.iter_modules(__path__):
        if module_info.name.startswith("_") or module_info.name == "common":
            continue
        module = importlib.import_module(f"{__name__}.{module_info.name}")
        for _, candidate in inspect.getmembers(module, inspect.isclass):
            if (
                issubclass(candidate, SystemDefinitionBase)
                and candidate is not SystemDefinitionBase
                and candidate.__module__ == module.__name__
                and candidate.name is not None
            ):
                register_system(candidate())


_discover_systems()
