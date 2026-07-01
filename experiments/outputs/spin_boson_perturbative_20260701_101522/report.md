# Spin-Boson Perturbative Open-Gate Optimization

Generated at: 2026-07-01T10:15:22

## Configuration

| Parameter | Value |
| --- | --- |
| objective | open_gate_fidelity_expansion |
| target_state | (\|00,0>-i\|11,0>)/sqrt(2) |
| target_gate | MS_XX(pi/2) |
| n_levels | 6 |
| n_steps | 5 |
| dt_s | 4.516e-05 |
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
| print_step | True |
| print_fidelity_terms | True |
| save_fidelity_terms | True |
| interrupted | False |
| reported_final_step | 1 |
| state_pair_count | 96 |
| l1_smooth_weight | 0.001 |
| l2_smooth_weight | 0.00015 |
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
| optimizer_options | {'maxiter': 1, 'gtol': 1e-12, 'ftol': 1e-15} |

## Results

| Metric | Initial | Final | Delta |
| --- | --- | --- | --- |
| single_state_fidelity | 0.96127582588 | 0.99597591771 | 0.0347000918299 |
| close_gate_fidelity | 0.964102991754 | 0.996631284571 | 0.0325282928173 |
| open_gate_fidelity | 0.963828042063 | 0.995903188185 | 0.0320751461223 |
| l1_penalty | 0.00653262379212 | 0.00655201550282 | 1.93917106909e-05 |
| l2_penalty | 0.000713079306586 | 0.000720344626082 | 7.26531949541e-06 |
| penalized_objective | 0.956582338964 | 0.988630828056 | 0.0320484890921 |

## Optimizer

| Parameter | Value |
| --- | --- |
| success | False |
| message | STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT |
| nit | 1 |
| nfev | 4 |

## Figures

### Pulse parameters

![Pulse parameters](spin_boson_perturbative_pulses.png)

### State propagation

![State propagation](spin_boson_perturbative_state_propagation.png)
