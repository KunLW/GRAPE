# Perturbative Open-Gate Optimization (spin_boson)

Generated at: 2026-07-14T06:45:10

## Run Summary

| Parameter | Value |
| --- | --- |
| experiment_dir | round_21_T76.90us |
| objective | noisy_gate_fidelity_expansion |
| system_type | spin_boson |
| n_levels | 6 |
| phi_s | 0 |
| eta | 0.075 |
| mode_vector | (0.5, -0.5) |
| target_gate | ms_xx_pi_over_2 |
| alpha1_khz_bounds | (1.0, 60.0) |
| alpha2_khz_bounds | (0.0, 200.0) |
| initial_pulse_shape | random |
| alpha1_cycles | 1 |
| alpha1_offset_fraction | 0.7 |
| alpha1_noise_fraction | 0.3 |
| n_steps | 400 |
| total_time_us | 76.8988152159 |
| include_fluctuations | True |
| alpha1_bounds_kHz | 1 to 60 |
| alpha2_bounds_kHz | 0 to 200 |
| max_order | 2 |
| state_pair_count | 96 |
| l1_smooth_weight | 0.0005 |
| l2_smooth_weight | 0.0001 |
| initial_pulse | final_pulse_s400.npz |

## Validity (Perturbative Expansion)

| Metric | Value | Definition |
| --- | --- | --- |
| kappa_1 | 0.367798285446 | dt * max_alpha \|\|H_nominal\|\|_2 over bounds (nominal step size) |
| kappa_2 | 0.0284892034428 | T * max_alpha \|\|H_fluctuation\|\|_2 over bounds (expansion small parameter) |

## Fluctuation Terms

| Term | Coefficient | Definition | Usage | Spectral Norm |
| --- | --- | --- | --- | --- |
| spin-shift | 31.4159 | kron(0.5 * (sz ⊗ I + I ⊗ sz), I_motion) | added directly to H_fluctuation | 31.4159 |
| motion-shift | 30 | kron(I_spin, number_operator) | added directly to H_fluctuation | 150 |
| alpha1-rel | 0.0001 | kron(I_spin, number_operator) | alpha1(t) * control[0] | 0.0005 |
| alpha2-rel | 0.0001 | eta * kron(S_phi(mode=(0.5, -0.5)), X1), X1=(a + adag)/2, eta=0.075 | alpha2(t) * control[1] | 1.24659653758e-05 |

## Results

| Metric | Initial | Final | Delta |
| --- | --- | --- | --- |
| single_state_fidelity | 0.998706421902 | 0.999998892267 | 0.00129247036493 |
| close_gate_fidelity | 0.998910075831 | 0.999999409318 | 0.00108933348657 |
| noisy_gate_fidelity | 0.998908704087 | 0.999997984297 | 0.00108928020964 |
| decoherence_correction | 0 | 0 | 0 |
| l1_penalty | 0.00260842388245 | 0.00257533304647 | -3.30908359846e-05 |
| l2_penalty | 1.89979020095e-06 | 1.57127485051e-06 | -3.28515350442e-07 |
| penalized_objective | 0.996298380415 | 0.997421079976 | 0.00112269956097 |

## Optimizer

| Parameter | Value |
| --- | --- |
| method | L-BFGS-B |
| options | {'maxiter': 250, 'gtol': 1e-10, 'ftol': 1e-15} |
| success | False |
| message | STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT |
| nit | 250 |
| nfev | 256 |
| interrupted | False |

## Key Files

| Output | Path |
| --- | --- |
| final_pulse_npz | final_pulse_s400.npz |
| final_pulse_csv | final_pulse.csv |
| initial_pulse_npz | initial_pulse_s400.npz |
| step_log | step_log.csv |

## Figures

### Pulse parameters

![Pulse parameters](pulse.png)

### State propagation

![State propagation](population.png)
