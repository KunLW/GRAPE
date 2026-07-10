# Adding a physical system: two files

The experiment driver (`experiments/driver/run_experiment.py`) is fully
decoupled from the physical system, and the registry in
`physical_systems/__init__.py` auto-discovers every module in this folder.
Adding a system (an NV center, a Rydberg array, ...) therefore means creating
exactly **two files** — no driver, registry, or engine edits:

1. `physical_systems/my_system.py` — the physics + definition
2. a YAML config with `system.type: my_system`

## How this differs from `quantum_control/systems/`

- `quantum_control/systems/` is the engine's **generic system model**:
  `ClosedSystem` (nominal + control Hamiltonians only), `OpenSystem`
  (a closed system combined with `NoiseTerm`s), and the noise vocabulary
  (`NoiseTerm` umbrella base with `FluctuationTerm` and `DecoherenceChannel`
  subtypes). It contains no concrete physics.
- `physical_systems/` (this folder) holds the **concrete physics**: one file
  per system with its operators, Hamiltonian, noise-term declarations,
  fidelity definition, and the `SystemDefinitionBase` adapter the driver
  consumes.

## Minimal closed-only system (~35 lines)

For closed-GRAPE optimization you implement only the closed system and the
fidelity; every noise hook defaults to "none" and the noise config defaults
to everything-disabled:

```python
# physical_systems/my_qubit.py
from dataclasses import dataclass

import numpy as np

from quantum_control import ClosedSystem, StatePair

from physical_systems.common import SystemDefinitionBase


@dataclass(frozen=True)
class MyQubitParams:                      # fields = system.params YAML schema
    rabi_max_rad_s: float = 1.0e5


class MyQubitDefinition(SystemDefinitionBase):
    name = "my_qubit"                     # = system.type in the YAML

    def default_params(self):
        return MyQubitParams()

    def build_closed_system(self, params):
        sx = np.array([[0, 1], [1, 0]], dtype=complex)
        return ClosedSystem(drift=np.zeros((2, 2), dtype=complex), controls=[sx])

    def control_bounds(self, params):     # (lower, upper) rad/s per channel
        return (np.array([0.0]), np.array([params.rabi_max_rad_s]))

    def state_pairs(self, params):        # the fidelity definition
        zero = np.array([1.0, 0.0], dtype=complex)
        one = np.array([0.0, 1.0], dtype=complex)
        return (StatePair(zero, one, 0.5), StatePair(one, zero, 0.5))
```

```yaml
# my_qubit.yaml
system:
  type: my_qubit
pulse:
  n_steps: 10
  total_time_us: 50.0
optimizer:
  maxiter: 10
```

Put the config in `experiments/my_qubit/configs/` (see `experiments/README.md`
for the per-system folder layout) and run with
`python -m experiments.driver.run_experiment --config experiments/my_qubit/configs/my_qubit.yaml`.

Required hooks: `name`, `default_params()`, `build_closed_system(params)`,
`control_bounds(params)`, `state_pairs(params)`. The default parameterization
is a `BoundedAmplitudeParameterization` from `control_bounds` and the default
initial pulse is flat at the bounds midpoint.

## Adding noise later

Noise is declared, not wired: override the term hooks and the base class
assembles the `OpenSystem` (closed system + selected terms) with the YAML
`enabled`/sigma/rate gating applied.

1. Config dataclasses (only for the noise types you use):
   - `MyFluctuations(FluctuationsConfigBase)` — sigma fields
     (``system.noise.fluctuations`` schema); `enabled` / `any_sigma_positive`
     inherited
   - `MyDecoherence(DecoherenceConfigBase)` — rate fields, 1/s
     (``system.noise.decoherence`` schema); `enabled` / `any_rate_positive`
     inherited
   - `MyNoise` container with `fluctuations` / `decoherence` fields, returned
     by `default_noise()`
2. Term hooks (each returns engine-level `NoiseTerm` subtypes):
   - `fluctuation_terms(params, fluctuations)` → list of `FluctuationTerm`
     (`kind="static"` added to H as-is; `kind="control"` scaled by the
     control amplitude, aligned with control channels positionally). Applied
     all-or-nothing: skipped entirely when every sigma is zero, but an
     individual zero-sigma term is never dropped, so control terms keep
     their positional alignment.
   - `decoherence_channels(params, decoherence)` → list of
     `DecoherenceChannel` (the `L = sqrt(gamma) * A` scaling lives on
     `.matrix`; zero-rate channels are dropped). Active channels appear in
     the report's Decoherence Channels table.

## Optional presentation & shaping hooks

- `target_gate(params)` → target unitary (informational; the fidelity itself
  comes from `state_pairs`)
- `build_initial_pulse` / `build_parameterization` — override for custom
  starts or structural constraints (see the alpha2 endpoint-zero wrapper in
  `spin_boson.py`)
- `control_channels(params)` → `ControlChannel(label, display_scale,
  display_unit)` per channel (plot labels/units, CSV columns)
- `population_structure(params)` → `PopulationStructure(dims, names, labels)`
  for the bipartite population-marginal plot, or `None` to skip
- `probe_state_pair(params)` → `StateProbe(initial_state, target_state,
  description)` for the single-state fidelity diagnostic and propagation
  plot, or `None` to skip

See `physical_systems/spin_boson.py` for the full reference implementation
(all hooks, plus standalone physics helpers such as the raw system builders).

## Invariant

**The driver and quantum_control core must never import concrete system
modules.** Everything system-specific flows through the definition object
resolved from `system.type`. (Grep-checkable: `run_experiment.py`,
`quantum_control/` must not mention a concrete system name; the only
exception is the driver's default `system.type`.)

## Naming convention

"Noise" is the umbrella term covering every noise type; the quasi-static
coherent kind is always called "fluctuation" and the Markovian kind
"decoherence". A name containing "noise" (e.g. `OpenSystem.noise_terms`,
`system.noise`, `noisy_gate_fidelity`) covers all types.

## Known debt

- `experiments/spin_boson/initial_pulses/make_initial_pulses.py` duplicates spin-boson
  constants (bounds, default total time, alpha2 endpoint convention)
  independently of the system definition; it is a standalone spin-boson
  utility.
- The CSV export writes converted-unit columns with a `_khz` suffix
  regardless of the channel's display unit (the conversion factor itself is
  taken from `control_channels`).
