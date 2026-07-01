# Spin-Boson Perturbative Open-Gate Optimization

Generated at: 2026-06-22T16:17:41

## Configuration

| Parameter | Value |
| --- | --- |
| objective | open_gate_fidelity_expansion |
| target_state | (\|00,0>-i\|11,0>)/sqrt(2) |
| target_gate | MS_XX(pi/2) |
| n_levels | 6 |
| n_steps | 20 |
| dt_s | 1.129e-05 |
| total_time_us | 225.8 |
| phi_s | 0 |
| alpha1_cycles | 1 |
| alpha1_bounds_khz | 1 to 600 |
| alpha2_bounds_khz | 0 to 20 |
| alpha2_endpoint_constraint | initial and final alpha2 fixed to 0 |
| static_fluctuation_count | 2 |
| control_fluctuation_count | 2 |
| max_order | 2 |
| drop_odd_average | True |
| workers | 1 |
| normalize_weights | False |
| no_progress | True |
| print_step | False |
| state_pair_count | 96 |
| l1_smooth_weight | 0 |
| l2_smooth_weight | 0.00015 |
| sweep_mode | random |
| sweep_seed | 12346 |
| noise_scale | NA |
| step_log | step_log.csv |
| initial_pulse_npz | initial_pulse.npz |
| initial_pulse_csv | initial_pulse.csv |
| final_pulse_npz | final_pulse.npz |
| final_pulse_csv | final_pulse.csv |
| optimizer_method | L-BFGS-B |
| optimizer_maximize | True |
| optimizer_options | {'maxiter': 1, 'gtol': 1e-12, 'ftol': 1e-15} |

## Results

| Metric | Initial | Final | Delta |
| --- | --- | --- | --- |
| single_state_fidelity | 0.213905350858 | 0.242821444379 | 0.0289160935216 |
| close_gate_fidelity | 0.412345707769 | 0.473816516487 | 0.0614708087177 |
| open_gate_fidelity | 1.0343733763 | 1.23085474705 | 0.196481370745 |
| l1_penalty | 0 | 0 | 0 |
| l2_penalty | 0.0123674416252 | 0.0125618414372 | 0.000194399811972 |
| penalized_objective | 1.02200593468 | 1.21829290561 | 0.196286970933 |

## Optimizer

| Parameter | Value |
| --- | --- |
| success | False |
| message | STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT |
| nit | 1 |
| nfev | 6 |

## Figures

### Pulse parameters

![Pulse parameters](spin_boson_perturbative_pulses.png)

### State propagation

![State propagation](spin_boson_perturbative_state_propagation.png)
