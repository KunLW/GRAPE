# Constant-pulse MS-gate baseline

Flat-amplitude Molmer-Sorensen reference pulse on the flattop_122us grid
(400 steps, T = 122.0133 us, eta = 0.075):

- `alpha1 = 2*pi / T` — detuning closing one phase-space loop over the gate,
- `alpha2 = 2*pi / (eta * T)` — drive strength giving the entangling phase
  `Theta = pi/2` at loop closure.

With the repo Hamiltonian `H = alpha1 (I x n) + alpha2 * eta * S_phi x
(a + adag)/2`, the effective drive is `g = alpha2 * eta / 2` and the closed
loop accumulates `Theta = 2*pi * (g / alpha1)^2`; `Theta = pi/2` requires
`g = alpha1 / 2`, i.e. `alpha2 = alpha1 / eta`. (An `alpha2 = pi / (2*eta*T)`
convention — 4x smaller — gives only `Theta = pi/32` here, closed fidelity
~0.577.) The first/last `alpha2` steps are pinned to zero by the shared
parameterization constraint; the remaining closed-gate infidelity
(~1.6e-4) is dominated by that plus the `n_levels = 6` Fock truncation at
peak displacement |alpha| ~ 1.

`config.yaml` copies the noise model of
`robustness_eval/flattop_122us/config.yaml` (static sigmas 60 rad/s,
relative control sigmas 1e-4, decoherence disabled) so results are directly
comparable to the shaped 122 us pulses. Run from the repository root:

```bash
# Generate outputs/constant_pulse_s400.npz (+ csv, png)
.venv/bin/python -m experiments.spin_boson.constant_pulse.make_constant_pulse

# Robustness sweep (results in a timestamped dir under outputs/)
.venv/bin/python -m experiments.spin_boson.robustness_eval.run_robustness_eval \
    --config experiments/spin_boson/constant_pulse/config.yaml \
    --pulse-npz experiments/spin_boson/constant_pulse/outputs/constant_pulse_s400.npz \
    --output-root experiments/spin_boson/constant_pulse/outputs \
    --y-scale infidelity
```
