# Adding a physical system

The experiment driver (`spin_boson_open.py`) is decoupled from the physical
system through the registry in `systems/__init__.py`. The YAML config selects
a system by name and supplies its parameters:

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

## Steps (e.g. for an NV center or a Rydberg array)

1. Create `systems/my_system.py` with two frozen dataclasses:
   - `MySystemParams` — every physics knob (dimensions, couplings, control
     bounds, target specification, initial-pulse shape parameters). Field
     names and defaults define the `system.params` YAML schema.
   - `MySystemNoise` — nested `decoherence` / `fluctuations` dataclasses,
     each with an `enabled: bool` flag plus its rates/strengths.
2. Implement a definition class (duck-typed, no base class needed):
   - `name = "my_system"`
   - `default_params()` / `default_noise()`
   - `build_systems(params, noise)` → `(system, noisy_system, noise_specs)`
     where the systems are `FluctuatingClosedSystem`-compatible and
     `noise_specs` is a list of dicts with keys
     `kind/name/coefficient/definition/usage/matrix` (used for the report's
     Noise Terms table). Return `(system, system, [])` when fluctuations are
     disabled.
   - `build_collapse_operators(params, noise)` → list of already-scaled jump
     operators `L = sqrt(gamma) A`; empty unless decoherence is enabled.
   - `build_initial_pulse(params, pulse_config)` → `PiecewiseConstantPulse`
     (`pulse_config` carries the generic `n_steps`, `total_time_us`,
     `random_seed`).
   - `build_parameterization(params, pulse)` → object with
     `to_physical` / `to_parameters` / `pullback_gradient` /
     `parameter_bounds`.
   - `target_gate(params)` / `state_pairs(params)` → target unitary and the
     weighted `StatePair` average defining the gate fidelity.
3. Register it at the bottom of `systems/__init__.py`:

   ```python
   from experiments_improved.systems import my_system as _my_system
   register_system(_my_system.MySystemDefinition())
   ```

4. Run with `--config my_config.yaml` where `system.type: my_system`.

No changes to `config_io.py` or the driver's config loading are needed — the
YAML sections are validated generically against your dataclasses (unknown
keys raise, absent keys keep defaults).

Note: some driver diagnostics (population-marginal plots, the Bell-state
single-state fidelity) still assume the spin ⊗ motion structure of the
spin-boson system; generalize or disable those when adding a structurally
different system.
