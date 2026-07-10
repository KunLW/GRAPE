# Time-collapsing pulse optimization

Finds how short a pulse can get before re-optimization stops paying off.
Starting from any experiment YAML, each round multiplies `pulse.total_time_us`
by a shrink factor and re-runs the GRAPE optimization warm-started from the
previous round's optimized pulse (`n_steps` stays fixed, so only `dt` shrinks
and the amplitudes carry over unchanged).

## Usage

```bash
.venv/bin/python -m experiments.spin_boson.time_collapsing.run_time_collapsing \
    --config experiments/spin_boson/time_collapsing/spin_boson_time_collapsing.yaml \
    --shrink-factor 0.9 \
    --fidelity-drop-tolerance 1e-6 \
    --max-rounds 40
```

`--maxiter N` overrides `optimizer.maxiter` from the base config (handy for
quick smoke runs). Any config accepted by `experiments.driver.run_experiment` works.

## Stop rule

The loop keeps shrinking while the open (noisy) gate fidelity holds or
improves. It stops at the first round whose `final_noisy_gate_fidelity` falls
more than `--fidelity-drop-tolerance` below the previous round's, when the
optimization is interrupted, or after `--max-rounds` rounds. The best round is
the one with the highest noisy gate fidelity.

## Outputs

One timestamped directory `<output_root>/time_collapsing_<timestamp>/`
containing:

- `round_NN_T<time>us/` — a full `run_experiment` run per round (report.md,
  pulse exports, step_log.csv, …)
- `summary.csv` — per-round total time, fidelities, objective, convergence
- `summary.md` — parameters, per-round table, best round, stop reason
- `fidelity_vs_time.png` — noisy and closed gate fidelity vs total pulse time
