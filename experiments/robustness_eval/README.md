# Robustness evaluation

System-agnostic pulse robustness check: load any valid experiment `config.yaml`
plus a pulse `.npz` (an `amplitudes` array matching the config's pulse grid,
e.g. an exported `final_pulse_s*.npz`) and sweep a global noise-scale factor
applied to the config's noise strengths. All fluctuation sigmas and
decoherence rates are multiplied by the scale — relative to the config's
values — while `enabled` flags are untouched, so noise types disabled in the
config stay disabled at every scale.

At each scale the pulse is evaluated with `noisy_gate_fidelity` (perturbative
expansion + first-order Lindblad correction). `--faithful` adds the exact
`faithful_gate_fidelity` curve (full Lindblad propagation, Gauss-Hermite
averaged; cost `hermite_points ** n_fluctuation_terms` nodes per scale). The
`closed_gate_fidelity` with all noise disabled is the scale -> 0 reference,
drawn as a horizontal dashed line on the log-scaled scale axis.

Run from the repository root:

```bash
.venv/bin/python -m experiments.robustness_eval.run_robustness_eval \
    --config <config.yaml> --pulse-npz <pulse.npz>
```

The default grid is 13 log-spaced scales from 0.01 to 100; customize with
`--scale-min/--scale-max/--n-scales` or an explicit `--scales 0.01,0.1,1,10,100`.
The plot's y-axis defaults to linear fidelity; `--y-scale infidelity` plots
`1 - F` on a log axis instead (log-log, so power-law noise scaling shows as a
straight line; points with `F >= 1` cannot be drawn and are dropped).
`--close-grape-pulse-npz <pulse.npz>` evaluates a second pulse (e.g. one
optimized with closed-system GRAPE) on the same scales and draws its curves
in the same figure, labeled "close-grape" — color encodes the pulse (blue vs
orange), marker/linestyle the metric. The close-grape pulse keeps its own
time grid from the npz (step count and dt; the config's dt is only a
fallback when the npz has none), so a pulse with a different duration is
evaluated as designed — both pulses face the identical scaled noise model,
each over its own gate time.
Results land in a timestamped directory next to the pulse `.npz` (override
with `--output-root`): `robustness.csv`, `robustness.png`, `report.md`, and a
resolved config snapshot.

Note the perturbative curve is a second-order expansion in the noise
strength; at large scales it can leave its validity range (values may even go
negative) — use `--faithful` where that matters.
