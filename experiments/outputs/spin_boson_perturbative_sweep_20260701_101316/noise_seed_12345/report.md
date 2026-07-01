# Spin-Boson Perturbative Open-Gate Optimization

Generated at: 2026-07-01T10:13:16

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
| print_fidelity_terms | True |
| save_fidelity_terms | True |
| interrupted | False |
| reported_final_step | 21 |
| state_pair_count | 96 |
| l1_smooth_weight | 0.001 |
| l2_smooth_weight | 0.00015 |
| sweep_mode | noise |
| sweep_seed | 12345 |
| noise_scale | 0.3 |
| step_log | step_log.csv |
| fidelity_terms | fidelity_terms.csv |
| fidelity_terms_by_pair | fidelity_terms_by_pair.csv |
| latest_pulse_npz | latest_pulse.npz |
| latest_pulse_csv | latest_pulse.csv |
| latest_parameters | latest_parameters.npz |
| initial_pulse_npz | initial_pulse.npz |
| initial_pulse_csv | initial_pulse.csv |
| final_pulse_npz | final_pulse.npz |
| final_pulse_csv | final_pulse.csv |
| optimizer_method | L-BFGS-B |
| optimizer_maximize | True |
| optimizer_options | {'maxiter': 40, 'gtol': 1e-12, 'ftol': 1e-15} |

## Results

| Metric | Initial | Final | Delta |
| --- | --- | --- | --- |
| single_state_fidelity | 0.216013135082 | 0.994293508789 | 0.778280373707 |
| close_gate_fidelity | 0.403504173498 | 0.996455387184 | 0.592951213686 |
| open_gate_fidelity | 0.403448604061 | 0.990248178117 | 0.586799574056 |
| l1_penalty | 0.107113289399 | 0.102400110243 | -0.0047131791566 |
| l2_penalty | 0.0221941460113 | 0.0202617250339 | -0.00193242097745 |
| penalized_objective | 0.27414116865 | 0.867586342841 | 0.59344517419 |

## Optimizer

| Parameter | Value |
| --- | --- |
| success | True |
| message | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |
| nit | 21 |
| nfev | 85 |

## Figures

### Pulse parameters

![Pulse parameters](spin_boson_perturbative_pulses.png)

### State propagation

![State propagation](spin_boson_perturbative_state_propagation.png)
