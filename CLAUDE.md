# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment and commands

Always use the repo's virtualenv, not system Python: `.venv/bin/python` (when working in a git worktree under `.claude/worktrees/`, the venv lives in the main checkout: `/Users/kun/Documents/GRAPE VERGE/.venv/bin/python`). Run everything from the repository root; `pytest.ini` sets `pythonpath = .`.

```bash
# Full test suite (one file holds all tests)
.venv/bin/python -m pytest tests/test_quantum_control.py

# Single test
.venv/bin/python -m pytest tests/test_quantum_control.py::test_name  # or -k pattern

# Modern YAML-driven optimization run
.venv/bin/python -m experiments_improved.run_experiment --config experiments_improved/configs/example.yaml

# Evaluate an exported pulse without optimizing
.venv/bin/python -m experiments_improved.run_experiment evaluate --config <cfg.yaml> --pulse-npz <pulse.npz>

# Legacy standalone spin-boson optimizer
.venv/bin/python experiments/spin_boson_perturbative_lbfgsb.py --maxiter 40 --n-steps 200
```

A couple of tests targeting the legacy `experiments/` scripts fail pre-existingly (stale sigma defaults in their expectations). When verifying a change, compare the pass/fail set against a baseline run rather than expecting zero failures — and for numerical changes, rerun a small fixed-seed experiment and check the metrics reproduce.

Runs write timestamped directories under the configured `output_root` (e.g. `experiments_improved/outputs/`) containing `report.md`, pulse `.npz`/`.csv` exports, `step_log.csv`, and checkpoint files that update during optimization.

## Architecture

Three layers, from generic to specific:

1. **`quantum_control/`** — the system-agnostic GRAPE engine (installable package). Data flow:
   `System + Pulse → StepBuilder → Evolution → Objective → Differentiator → Optimizer`.
   Two noise treatments, deliberately separate because they are leading-order expansions in different small parameters:
   - *Quasi-static (long-correlation) fluctuations*: `PerturbativeExpansionEvolution` + `ExpansionFidelity` propagate expansion components (`F`, `SF`, `DF`, …) so higher orders can be added without touching objectives or optimizers.
   - *Decoherence (short-correlation)*: `LindbladExpansionEvolution` + `LindbladCorrectedStateFidelity` compute an additive first-order correction; `CombinedStateAverageProblem` sums the two into the optimizer objective.
   `ExpansionStateAverageFidelity` averages any of these over weighted state pairs (with optional worker processes). Gate metrics: `closed_gate_fidelity(system, pulse, state_pairs)` and `noisy_gate_fidelity(system, pulse, state_pairs, collapse_operators=())` — the latter equals the optimizer's raw objective (expansion + Lindblad correction).

2. **`experiments_improved/`** — the modern driver. `run_experiment.py` is sealed: it reaches every system-specific fact (Hamiltonians, noise, bounds, targets, plot labels) through a system-definition object resolved from the YAML `system.type` via the registry in `system_definitions/__init__.py`. **Invariant: the driver and `quantum_control` core must never import concrete system modules.** Adding a new physical system = one subclass of `SystemDefinitionBase` (`system_definitions/common.py`) + one registry line, zero driver edits — follow `system_definitions/README.md`; `system_definitions/spin_boson.py` is the reference subclass. Don't confuse `experiments_improved/system_definitions/` (driver-facing adapters) with `quantum_control/systems/` (the reusable physics library — System classes and Hamiltonian builders — consumed by both the definitions and the legacy scripts).

3. **`experiments/`** — legacy standalone spin-boson scripts. Kept runnable (shared `StepLog`/reporting in `experiments/reporting.py`) but frozen in vocabulary; don't extend them, extend `experiments_improved/` instead.

### Conventions that span files

- **Units**: user-facing configs use kHz and microseconds; everything internal is angular frequency in rad/s and seconds. Convert via `quantum_control/units.py` (`RAD_S_PER_KHZ`, `khz_bounds_to_rad_s`); never hand-roll the factor.
- **Noise scaling**: coherent fluctuation matrices are already-scaled `sigma * H` (linear in sigma; declared as `NoiseTerm`). Lindblad jump operators are `L = sqrt(gamma) * A` (declared as `DecoherenceChannel`); the base class applies the scaling and the `enabled`/`any_rate_positive` gating.
- **Pulses**: piecewise-constant, shape `(n_steps, n_controls)`; parameterizations map bounded physical amplitudes to unbounded optimizer parameters, and the generic constraint check is the round trip `to_physical(to_parameters(A)) ≈ A`.
- **YAML configs**: block style; `system: {type, params, noise:{fluctuations, decoherence}}` sections are validated generically against the system's dataclasses (unknown keys raise, absent keys keep defaults). Keep example-config values trivially small (few levels/steps/iterations).

## Theory documents

`doc/report_opengrape_iontrap.tex` is the theory source for the perturbative and Lindblad limits. When reviewing it, never edit equations directly — flag suspected math errors with `% REVIEW` comments instead.
