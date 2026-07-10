# Experiments

This directory is the driver layer: it runs YAML-configured optimizations and
evaluations on top of the system-agnostic `quantum_control/` engine and the
concrete systems in `physical_systems/`. The driver is sealed — it reaches every
system-specific fact (Hamiltonians, noise, bounds, targets, plot labels) through
the system definition resolved from the YAML `system.type`, and never imports
concrete system modules directly.

Always run from the repository root with the repo virtualenv (`.venv/bin/python`).

## Running an optimization

```bash
.venv/bin/python -m experiments.driver.run_experiment --config experiments/spin_boson/configs/example.yaml
```

The YAML config selects the physical system (`system.type`), its parameters and
noise terms, plus pulse, optimizer, penalty, and runtime settings. Explicit CLI
flags override the config; the most common ones:

| Flag | Effect |
| --- | --- |
| `--maxiter`, `--n-steps` | optimizer iterations / pulse steps |
| `--workers N` | worker processes for state-pair averaging |
| `--initial-pulse-npz FILE` | start from a custom pulse (`amplitudes` array) |
| `--close-grape` | disable all noise terms — fluctuations and decoherence (closed-system GRAPE) |
| `--output-root`, `--output-prefix` | where run directories go and how they are named (see below) |
| `--print-step`, `--print-fidelity-terms`, `--no-progress` | logging verbosity |

## Evaluating a pulse without optimizing

```bash
.venv/bin/python -m experiments.driver.run_experiment evaluate \
    --config <cfg.yaml> --pulse-npz <pulse.npz>
```

Reports the closed and noisy gate fidelities for the pulse under the configured
system. Add `--faithful` to also compute the exact check — full Lindblad
density-matrix propagation with Gauss–Hermite averaging over the fluctuations
(`--hermite-points` nodes per fluctuation dimension; cost grows as
`hermite_points ** n_fluctuation_terms`).

## Output folder structure

Every run creates a timestamped directory under `output.output_root`
(CLI: `--output-root`), named `<prefix>_<YYYYMMDD>_<HHMMSS>` where `<prefix>`
is `output.prefix` (CLI: `--output-prefix`) and defaults to the system name.
When `output_root` is unset, runs land in the system's scratch area,
`experiments/<system.type>/tmp/outputs/`; real experiments set it explicitly
to their own `<system_name>/<experiment_name>/outputs/`.
An optimization run contains:

```
<output_root>/
└── <prefix>_20260710_193000/
    ├── config.yaml                        # exact config snapshot (reloadable via --config)
    ├── report.md                          # summary: fidelities, parameters, file index
    ├── step_log.csv                       # per-step fidelities, penalties, cost, gradient norm
    ├── pulse.png                          # control-amplitude plot
    ├── population.png                     # population/fidelity plot
    ├── pulse/                             # all pulse exports; .npz carries the step count,
    │   ├── initial_pulse_s<n_steps>.npz   #   .csv is the same pulse in kHz for inspection
    │   ├── initial_pulse.csv
    │   ├── final_pulse_s<n_steps>.npz
    │   ├── final_pulse.csv
    │   ├── latest_pulse_s<n_steps>.npz    # checkpoints, overwritten during optimization;
    │   ├── latest_pulse.csv               #   deleted when the run completes, kept only
    │   └── latest_parameters.npz          #   after an interrupted run
    ├── fidelity_terms.csv                 # only with --print-fidelity-terms /
    └── fidelity_terms_by_pair.csv         #   runtime.save_fidelity_terms
```

An `evaluate` run writes a lighter `<prefix>_evaluation_<timestamp>/` directory
with `config.yaml`, `eva_report.md`, `population.png`, and
`pulse/evaluated_pulse_s<n_steps>.npz` / `pulse/evaluated_pulse.csv`.

## Files
- `driver/` — driver scripts: `run_experiment.py` (main entry point),
  `reporting.py` (report/step-log/plot plumbing), `config_io.py` (YAML ↔ config
  dataclasses), and `evaluate_error_budget.py` (standalone error-budget CLI).
- `<system_name>/` — one folder per physical system, named after the registry
  name (`system.type`), e.g. `spin_boson/`.

## Adding a new physical system configuration
Add a folder `<system_name>/configs/` and put a YAML config there, along with any auxiliary files (e.g. initial pulses, noise data). The config should point to the system module via `system.type` and set any parameters or noise terms. Add a system-specific `README.md` if needed.

## Adding a new experiment
If the system is not yet configured, first add a new physical system configuration as described above.
All experiment scripts should be placed under `<system_name>/<experiment_name>/`. Experiments may use a different config, so every experiment's output will have its own config file when the experiment is run.
- `<system_name>/tmp/` — for all temporary files. Not git-tracked.
- `<system_name>/<experiment_name>/outputs/` — for all outputs of the experiment. Not git-tracked.
- `<system_name>/reports/` — for all reports of that system. Git-tracked.

When running multiple experiments, using a git worktree per experiment is preferred.

## Conventions

- User-facing configs are in kHz and microseconds; everything internal is
  angular frequency in rad/s and seconds (`quantum_control/units.py`).
- Pulses are piecewise-constant arrays of shape `(n_steps, n_controls)`.
- YAML uses block style with `system: {type, params, noise: {fluctuations,
  decoherence}}` sections validated against the system's dataclasses.
