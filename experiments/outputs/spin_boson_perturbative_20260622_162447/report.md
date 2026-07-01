# Spin-Boson Perturbative Open-Gate Optimization

Generated at: 2026-06-22T16:24:47

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
| l2_smooth_weight | 0 |
| step_log | step_log.csv |
| initial_pulse_npz | initial_pulse.npz |
| initial_pulse_csv | initial_pulse.csv |
| final_pulse_npz | final_pulse.npz |
| final_pulse_csv | final_pulse.csv |
| optimizer_method | L-BFGS-B |
| optimizer_maximize | True |
| optimizer_options | {'maxiter': 4, 'gtol': 1e-12, 'ftol': 1e-15} |

## Results

| Metric | Initial | Final | Delta |
| --- | --- | --- | --- |
| single_state_fidelity | 0.953715161752 | 0.999932444788 | 0.0462172830364 |
| close_gate_fidelity | 0.956608276192 | 0.999948853875 | 0.0433405776828 |
| open_gate_fidelity | 0.9552236795 | 0.99834879667 | 0.0431251171704 |
| l1_penalty | 0 | 0 | 0 |
| l2_penalty | 0 | 0 | 0 |
| penalized_objective | 0.9552236795 | 0.99834879667 | 0.0431251171704 |

## Optimizer

| Parameter | Value |
| --- | --- |
| success | False |
| message | STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT |
| nit | 4 |
| nfev | 6 |

## Figures

### Pulse parameters

![Pulse parameters](spin_boson_perturbative_pulses.png)

### State propagation

![State propagation](spin_boson_perturbative_state_propagation.png)
