# Spin-Boson Perturbative Open-Gate Optimization

Generated at: 2026-06-21T16:43:55

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
| l1_smooth_weight | 0.01 |
| l2_smooth_weight | 0.4 |
| optimizer_method | L-BFGS-B |
| optimizer_maximize | True |
| optimizer_options | {'maxiter': 100, 'gtol': 1e-12, 'ftol': 1e-15} |

## Results

| Metric | Initial | Final | Delta |
| --- | --- | --- | --- |
| single_state_fidelity | 0.302428010701 | 0.999841879407 | 0.697413868706 |
| close_gate_fidelity | 0.416886183732 | 0.999909379453 | 0.583023195721 |
| open_gate_fidelity | 0.426179051826 | 0.996667816262 | 0.570488764436 |
| l1_penalty | 0.0799938316051 | 0.0789985161894 | -0.000995315415703 |
| l2_penalty | 0.000245598992386 | 0.000499780883546 | 0.000254181891161 |
| penalized_objective | 0.345939621229 | 0.917169519189 | 0.57122989796 |

## Optimizer

| Parameter | Value |
| --- | --- |
| success | False |
| message | STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT |
| nit | 100 |
| nfev | 115 |

## Figures

### Pulse parameters

![Pulse parameters](spin_boson_perturbative_pulses.png)

### State propagation

![State propagation](spin_boson_perturbative_state_propagation.png)
