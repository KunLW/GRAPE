# Spin-Boson Perturbative Open-Gate Optimization

Generated at: 2026-06-22T16:30:49

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
| no_progress | True |
| print_step | True |
| state_pair_count | 96 |
| l1_smooth_weight | 0 |
| l2_smooth_weight | 0.00015 |
| sweep_mode | custom |
| source_npz | experiments/outputs/spin_boson_perturbative_20260622_155259/final_pulse.npz |
| source_dt | 1.129e-06 |
| dt_missing | False |
| dt_mismatch | False |
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
| single_state_fidelity | 0.99870750166 | 0.999020280062 | 0.000312778402572 |
| close_gate_fidelity | 0.998999157145 | 0.999262868071 | 0.000263710925623 |
| open_gate_fidelity | 0.9991793122 | 0.999861898633 | 0.000682586432558 |
| l1_penalty | 0 | 0 | 0 |
| l2_penalty | 5.17194161259e-05 | 3.15110868044e-05 | -2.02083293215e-05 |
| penalized_objective | 0.999127592784 | 0.999830387546 | 0.000702794761879 |

## Optimizer

| Parameter | Value |
| --- | --- |
| success | False |
| message | ABNORMAL:  |
| nit | 8 |
| nfev | 118 |

## Figures

### Pulse parameters

![Pulse parameters](spin_boson_perturbative_pulses.png)

### State propagation

![State propagation](spin_boson_perturbative_state_propagation.png)
