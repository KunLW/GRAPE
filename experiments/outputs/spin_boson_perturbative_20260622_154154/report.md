# Spin-Boson Perturbative Open-Gate Optimization

Generated at: 2026-06-22T15:41:54

## Configuration

| Parameter | Value |
| --- | --- |
| objective | open_gate_fidelity_expansion |
| target_state | (\|00,0>-i\|11,0>)/sqrt(2) |
| target_gate | MS_XX(pi/2) |
| n_levels | 6 |
| n_steps | 200 |
| dt_s | 1.129e-06 |
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
| workers | 10 |
| normalize_weights | False |
| no_progress | False |
| print_step | True |
| state_pair_count | 96 |
| l1_smooth_weight | 0 |
| l2_smooth_weight | 0.0015 |
| step_log | step_log.csv |
| initial_pulse_npz | initial_pulse.npz |
| initial_pulse_csv | initial_pulse.csv |
| final_pulse_npz | final_pulse.npz |
| final_pulse_csv | final_pulse.csv |
| optimizer_method | L-BFGS-B |
| optimizer_maximize | True |
| optimizer_options | {'maxiter': 100, 'gtol': 1e-12, 'ftol': 1e-15} |

## Results

| Metric | Initial | Final | Delta |
| --- | --- | --- | --- |
| single_state_fidelity | 0.302428010701 | 0.999633886271 | 0.697205875569 |
| close_gate_fidelity | 0.416886183732 | 0.999724915365 | 0.582838731633 |
| open_gate_fidelity | 0.426179051826 | 0.999159593272 | 0.572980541446 |
| l1_penalty | 0 | 0 | 0 |
| l2_penalty | 9.20996221446e-07 | 0.000237692600057 | 0.000236771603835 |
| penalized_objective | 0.42617813083 | 0.998921900672 | 0.572743769842 |

## Optimizer

| Parameter | Value |
| --- | --- |
| success | False |
| message | STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT |
| nit | 100 |
| nfev | 817 |

## Figures

### Pulse parameters

![Pulse parameters](spin_boson_perturbative_pulses.png)

### State propagation

![State propagation](spin_boson_perturbative_state_propagation.png)
