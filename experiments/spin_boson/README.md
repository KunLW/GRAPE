# Spin-Boson Experiments

Everything specific to the `spin_boson` system (two-qubit Mølmer–Sørensen-type
gate with a shared motional mode) lives here, following the layout in
`experiments/README.md`:

```
spin_boson/
├── configs/          # YAML configs + auxiliary files (git-tracked)
│   ├── example.yaml                     # documents every configuration key
│   ├── example2.yaml                    # deterministic cosine/sine initial pulse
│   ├── spin_boson_high_static_fluc.yaml # pulse-search config, strong static fluctuations
│   └── spin_boson_low_static_fluc.yaml  # pulse-search config, weak static fluctuations
├── initial_pulses/   # experiment: generate + screen a family of starting pulses
├── time_collapsing/  # Experiment 1: gate time shrinking
├── pulse_search/     # Experiment 2: multi-initial-pulse search (local or Slurm)
├── slurm_time_shrink/# Experiment 1b: shrink every pulse-search winner from 3 start times (Slurm)
├── evaluation_check/ # sanity check: the four fidelity calculations along a run
├── reports/          # written reports for this system (git-tracked)
├── tmp/              # scratch; default output_root for ad-hoc runs (git-ignored)
└── outputs/          # legacy run data from before the folder restructure (git-ignored)
```

Ad-hoc run against this system:

```bash
.venv/bin/python -m experiments.driver.run_experiment --config experiments/spin_boson/configs/example.yaml
```

## Experiment plan

The three experiments build on each other: Experiment 2 finds good pulse
shapes, Experiment 1 shrinks the gate time for the best of them, and
Experiment 3 analyzes the noise robustness of the resulting optimal pulse.

### Experiment 1: gate time shrinking (`time_collapsing/`)

Finds how short a pulse can get before re-optimization stops paying off:
each round multiplies `pulse.total_time_us` by a shrink factor and re-runs
GRAPE warm-started from the previous round's optimized pulse (`n_steps`
fixed, so only `dt` shrinks). The loop stops when the noisy gate fidelity
drops by more than a tolerance, and reports the best round with a
`fidelity_vs_time.png` sweep plus per-round summaries.

Status: implemented in `time_collapsing/` (see its README; run via
`python -m experiments.spin_boson.time_collapsing.run_time_collapsing`).
A completed sweep (T = 300 µs shrunk to ~105 µs over 11 rounds) is in
`outputs/time_collapsing_20260709_195249/`.

### Experiment 2: pulse searching (`pulse_search/`)

GRAPE converges to a local optimum near its initial guess, so this experiment
searches over a gallery of ~10 named initial pulses, running one independent
optimization per pulse under the same config, then collects a best-first
summary to pick the best basin. Runs are idempotent per-pulse directories, so
they parallelize locally (`--parallel K`) or as a Slurm job array on USTC-SCC.

Status: implemented in `pulse_search/` (see its README and `scc_deploy.md`
for the USTC-SCC Slurm workflow). The
`configs/spin_boson_{high,low}_static_fluc.yaml` configs here define its two
noise scenarios; a completed gallery search is in `outputs/pulse_search_260709/`.

### Experiment 1b: time shrinking at scale (`slurm_time_shrink/`)

Combines the two: every final pulse from the pulse search is run through the
time-collapsing loop from three starting total times (the original 225.8 µs,
300 µs, and 170 µs), as a 30-task Slurm array on USTC-SCC. The search's final
pulses are copied into the git-tracked `slurm_time_shrink/pulses/` so they
reach the cluster.

Status: implemented in `slurm_time_shrink/` (its README is the full deploy,
monitoring, and result-retrieval guide).

### Experiment 3: noise robustness analysis (planned)

After Experiment 2 yields good pulse shapes and Experiment 1 shrinks their
gate time, analyze the noise robustness of the optimal pulse shape: sweep the
fluctuation sigmas / decoherence rates around their nominal values and check
the fidelity response (`evaluate --faithful` gives the exact Lindblad +
Gauss–Hermite reference; `driver/evaluate_error_budget.py` gives the
per-term perturbative budget). Not started — no code yet.

## Supporting experiment: `evaluation_check/`

Compares the four fidelity calculations (closed, perturbative open, Lindblad-
corrected, and the exact faithful check) along an optimization run to validate
that the optimizer's objective tracks the exact fidelity:

```bash
.venv/bin/python experiments/spin_boson/evaluation_check/run_evaluation_check.py \
    --config experiments/spin_boson/evaluation_check/spin-boson-origin.yaml
```

## Supporting experiment: `initial_pulses/`

Generates the named starting-pulse families used across the experiments
(`make_initial_pulses.py`, written to `initial_pulses/outputs/initial_pulses/`)
and batch-evaluates or batch-optimizes them (`run_initial_pulses.py`):

```bash
.venv/bin/python -m experiments.spin_boson.initial_pulses.make_initial_pulses
.venv/bin/python -m experiments.spin_boson.initial_pulses.run_initial_pulses --workers 4
```
