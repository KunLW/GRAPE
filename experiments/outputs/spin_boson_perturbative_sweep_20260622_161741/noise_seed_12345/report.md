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
| sweep_mode | noise |
| sweep_seed | 12345 |
| noise_scale | 0.3 |
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
| single_state_fidelity | 0.272046679695 | 0.221039475292 | -0.0510072044039 |
| close_gate_fidelity | 0.448497584317 | 0.380238591147 | -0.0682589931701 |
| open_gate_fidelity | 1.12550846906 | 1.96801026467 | 0.842501795611 |
| l1_penalty | 0 | 0 | 0 |
| l2_penalty | 0.00320878727482 | 0.0137827495227 | 0.0105739622479 |
| penalized_objective | 1.12229968178 | 1.95422751515 | 0.831927833363 |

## Optimizer

| Parameter | Value |
| --- | --- |
| success | False |
| message | STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT |
| nit | 1 |
| nfev | 2 |

## Figures

### Pulse parameters

![Pulse parameters](spin_boson_perturbative_pulses.png)

### State propagation

![State propagation](spin_boson_perturbative_state_propagation.png)
