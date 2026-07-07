# Adding a physical system

The experiment driver (`run_experiment.py`) is fully decoupled from the
physical system through the registry in `system_definitions/__init__.py`: metrics,
reports, plots, and CSV exports all go through the system definition, so a
new system needs **no driver changes**. The YAML config selects a system by
name and supplies its parameters:

```yaml
system:
  type: my_system        # registry key
  params:                # schema = your Params dataclass
    ...
  noise:                 # schema = your Noise dataclass
    decoherence:
      enabled: false
      ...
    fluctuations:
      enabled: true
      ...
```

## How this differs from `quantum_control/systems/`

- `quantum_control/systems/` is the **physics library**: System classes
  (`ClosedSystem`, `FluctuatingClosedSystem`, ...) and concrete
  Hamiltonian/operator builders (e.g. `spin_boson_control_system`). It is
  used by both the legacy `experiments/` scripts and the definitions here.
- `experiments_improved/system_definitions/` (this folder) is the **driver
  adapter layer**: registry + `SystemDefinitionBase` subclasses that assemble
  library pieces into everything `run_experiment.py` needs, per the YAML
  config.

## Invariant

**The driver and quantum_control core must never import concrete system
modules.** Everything system-specific flows through the definition object
resolved from `system.type`. (Grep-checkable: `run_experiment.py` and
`quantum_control/{problem,optimizers,evolution,objectives,steps,
differentiators}` must not mention a concrete system name; the only
exceptions are the driver's default `system.type` and the spin-boson-only
`--gamma-*` CLI sugar, which raises for other system types.)

## Steps (e.g. for an NV center or a Rydberg array)

Subclass `SystemDefinitionBase` from `system_definitions/common.py`; it provides the
generic plumbing (noise-spec bookkeeping, decoherence gating, default
parameterization/initial pulse, presentation defaults), so the subclass is
essentially pure physics.

1. Create `system_definitions/my_system.py` with the config dataclasses:
   - `MySystemParams` — every physics knob (dimensions, couplings, control
     bounds, target specification). Field names and defaults define the
     `system.params` YAML schema, and all fields are echoed generically into
     reports.
   - `MySystemDecoherence(DecoherenceConfigBase)` — just the rate fields
     (floats, 1/s); `enabled` and the `any_rate_positive` gating are
     inherited.
   - `MySystemFluctuations` — an `enabled: bool` flag plus the sigma fields.
   - `MySystemNoise` — container with `decoherence` / `fluctuations` fields.
2. Implement the physics hooks on `MySystemDefinition(SystemDefinitionBase)`:
   - `name = "my_system"`
   - `default_params()` / `default_noise()`
   - `build_nominal_system(params, static_fluctuations=(), control_fluctuations=())`
     → a `FluctuatingClosedSystem`-compatible system
   - `noise_terms(params, fluctuations)` → list of `NoiseTerm` (declarative;
     the base class builds the spec dicts and fluctuation matrices)
   - `control_bounds(params)` → `(lower, upper)` rad/s arrays, one entry per
     control channel (drives the default parameterization and initial pulse)
   - `target_gate(params)` / `state_pairs(params)` → target unitary and the
     weighted `StatePair` average defining the gate fidelity
3. Optionally override:
   - `decoherence_channels(params, decoherence)` → list of
     `DecoherenceChannel` (declarative like `noise_terms`; the base class
     gates on `enabled`/`any_rate_positive`, drops zero-rate channels, and
     applies the `L = sqrt(gamma) * A` scaling; default: none). Active
     channels are documented in the report's Decoherence Channels table.
   - `build_initial_pulse` / `build_parameterization` — the defaults are
     flat-at-midpoint amplitudes and a plain `BoundedAmplitudeParameterization`
     from `control_bounds`
   - presentation hooks (all optional, driver skips what is absent):
     - `control_channels(params)` → `ControlChannel(label, display_scale,
       display_unit)` per channel (plot labels/units, CSV columns)
     - `population_structure(params)` → `PopulationStructure(dims, names,
       labels)` for the bipartite population-marginal plot, or `None`
     - `probe_state_pair(params)` → `StateProbe(initial_state, target_state,
       description)` for the single-state fidelity diagnostic and the
       propagation plot, or `None`
4. Register it at the bottom of `system_definitions/__init__.py`:

   ```python
   from experiments_improved.system_definitions import my_system as _my_system
   register_system(_my_system.MySystemDefinition())
   ```

5. Run with `--config my_config.yaml` where `system.type: my_system`.

No changes to `config_io.py` or the driver are needed — the YAML sections are
validated generically against your dataclasses (unknown keys raise, absent
keys keep defaults), and every report/plot adapts through the hooks above.

See `system_definitions/spin_boson.py` for the reference subclass: it overrides the
initial pulse (randomized alpha1 start) and the parameterization (alpha2
pinned to zero at both endpoints) and provides all three presentation hooks.

## Known debt

- `make_initial_pulses.py` duplicates spin-boson constants (bounds, default
  total time, alpha2 endpoint convention) independently of the system
  definition; it is a standalone spin-boson utility.
- The CSV export writes converted-unit columns with a `_khz` suffix
  regardless of the channel's display unit (the conversion factor itself is
  taken from `control_channels`).
