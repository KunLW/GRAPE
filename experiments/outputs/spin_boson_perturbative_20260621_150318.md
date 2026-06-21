# Spin-Boson Perturbative Open-Gate Optimization

Generated at: 2026-06-21T15:03:18

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
| state_pair_count | 96 |
| l1_smooth_weight | 0.1 |
| l2_smooth_weight | 1 |
| optimizer_method | L-BFGS-B |
| optimizer_maximize | True |
| optimizer_options | {'maxiter': 10, 'gtol': 1e-12, 'ftol': 1e-15} |

## Results

| Metric | Initial | Final | Delta |
| --- | --- | --- | --- |
| single_state_fidelity | 0.302428010701 | 0.999964623327 | 0.697536612626 |
| close_gate_fidelity | 0.416886183732 | 0.99997543851 | 0.583089254778 |
| open_gate_fidelity | 0.416989188177 | 0.999359771727 | 0.58237058355 |
| l1_penalty | 0.799938316051 | 0.771296209135 | -0.028642106916 |
| l2_penalty | 0.000613997480964 | 0.000939878538596 | 0.000325881057632 |
| penalized_objective | -0.383563125355 | 0.227123684053 | 0.610686809408 |

## Optimizer

| Parameter | Value |
| --- | --- |
| success | False |
| message | STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT |
| nit | 10 |
| nfev | 16 |

## Figures

### Pulse parameters

![Pulse parameters](spin_boson_perturbative_pulses.png)

### State propagation

![State propagation](spin_boson_perturbative_state_propagation.png)
