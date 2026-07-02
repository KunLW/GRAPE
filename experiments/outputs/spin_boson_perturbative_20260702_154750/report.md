# Spin-Boson Perturbative Open-Gate Optimization

Generated at: 2026-07-02T15:47:50

## Preview

### Configuration

| Parameter | Value |
| --- | --- |
| experiment_dir | spin_boson_perturbative_20260702_154750 |
| objective | open_gate_fidelity_expansion |
| target_state | (\|00,0>-i\|11,0>)/sqrt(2) |
| target_gate | MS_XX(pi/2) |
| n_levels | 6 |
| n_steps | 200 |
| dt_s | 1.129e-06 |
| total_time_us | 225.8 |
| phi_s | 0 |
| eta | 0.075 |
| alpha1_cycles | 1 |
| alpha1_bounds_khz | 1 to 10 |
| alpha2_bounds_khz | 0 to 20 |
| alpha2_endpoint_constraint | initial and final alpha2 fixed to 0 |
| static_fluctuation_count | 2 |
| control_fluctuation_count | 2 |
| max_order | 2 |
| drop_odd_average | True |
| workers | 10 |
| normalize_weights | False |
| no_progress | False |
| print_step | False |
| print_fidelity_terms | False |
| save_fidelity_terms | False |
| state_pair_count | 96 |
| l1_smooth_weight | 0.001 |
| l2_smooth_weight | 0.00015 |

### Optimizer

| Parameter | Value |
| --- | --- |
| optimizer_method | L-BFGS-B |
| optimizer_maximize | True |
| optimizer_options | {'maxiter': 40, 'gtol': 1e-12, 'ftol': 1e-15} |

### Initial Metrics

| Metric | Value |
| --- | --- |
| initial_penalized_objective | 0.0609191430344 |
| initial_raw_fidelity | 0.571901813037 |
| initial_close_gate_fidelity | 0.572600102729 |
| initial_open_gate_fidelity | 0.571901813037 |
| initial_l1_penalty | 0.259936730582 |
| initial_l2_penalty | 0.251045939421 |

### Kappa Diagnostics

| Metric | Value | Definition |
| --- | --- | --- |
| kappa_1 | 0.356648455605 | max_boundary_corner dt * \|\|H_nominal(alpha)\|\|_2 over alpha bounds |
| kappa_2 | 1.51096426278 | max_boundary_corner T * \|\|H_fluctuation(alpha)\|\|_2 over alpha bounds; fluctuation terms are already scaled |
| kappa_1_corner | 3 | boundary corner attaining max \|\|H_nominal\|\|_2 |
| kappa_2_corner | 3 | boundary corner attaining max \|\|H_fluctuation\|\|_2 |
| kappa_1_alpha | (62831.85307179586, 125663.70614359171) | alpha values at kappa_1 corner in rad/s |
| kappa_2_alpha | (62831.85307179586, 125663.70614359171) | alpha values at kappa_2 corner in rad/s |
| kappa_1_h_norm | 315897.657755 | max \|\|H_nominal\|\|_2 |
| kappa_2_h_fluc_norm | 6691.60435243 | max \|\|H_fluctuation\|\|_2 |
| kappa_dt_s | 1.129e-06 | pulse time step |
| kappa_total_time_s | 0.0002258 | pulse total duration |
| kappa_boundary_corner_count | 4 | number of alpha-boundary corners evaluated |

### Output Manifest

| Output | Path |
| --- | --- |
| pulse_plot | spin_boson_perturbative_pulses.png |
| propagation_plot | spin_boson_perturbative_state_propagation.png |
| step_log | step_log.csv |
| fidelity_terms | disabled |
| fidelity_terms_by_pair | disabled |
| latest_pulse_npz | latest_pulse.npz |
| latest_pulse_csv | latest_pulse.csv |
| latest_parameters | latest_parameters.npz |
| initial_pulse_npz | initial_pulse.npz |
| initial_pulse_csv | initial_pulse.csv |
| final_pulse_npz | final_pulse.npz |
| final_pulse_csv | final_pulse.csv |

## Noise Terms

| Term | Coefficient | Definition | Usage | Shape | Frobenius Norm | Spectral Norm | Zero |
| --- | --- | --- | --- | --- | --- | --- | --- |
| static[0] | 314.159 | kron(0.5 * (sz ⊗ I + I ⊗ sz), I_motion) | added directly to H_fluctuation | 24x24 | 1088.27869931 | 314.159 | False |
| static[1] | 1256.637 | kron(I_spin, number_operator) | added directly to H_fluctuation | 24x24 | 18638.9388365 | 6283.185 | False |
| control[0] | 0.0003 | kron(I_spin, number_operator) | alpha1(t) * control[0] | 24x24 | 0.00444971909226 | 0.0015 | False |
| control[1] | 0.0006 | eta * kron(S_phi(mode=(0.5, -0.5)), X1), X1=(a + adag)/2, eta=0.075 | alpha2(t) * control[1] | 24x24 | 0.000174284250579 | 7.47957922549e-05 | False |

## System Construction Script

```python
phi_s = 0.0
eta = 0.075
system = spin_boson_control_system(n_levels=6, phi_s=phi_s, eta=eta)
noisy_system = spin_boson_noisy_control_system(n_levels=6, phi_s=phi_s)
initial_pulse = _customized_initial_pulse(
    n_steps=200,
    alpha1_cycles=1.0,
)
parameterization = Alpha2EndpointZeroParameterization(
    spin_boson_parameterization(initial_pulse.n_steps)
)
target_gate = ms_xx_pi_over_2_gate()
state_pairs = motion_resolved_gate_state_pairs(target_gate, 6)
step_builder = PerturbativeStepBuilder()
expansion_objective = ExpansionFidelity(max_order=2, drop_odd_average=True)
optimization_problem = ExpansionStateAverageFidelity(
    system=noisy_system,
    pulse=initial_pulse,
    evolution=PerturbativeExpansionEvolution(step_builder, max_order=2),
    objective=expansion_objective,
    differentiator=PerturbativeExpansionDifferentiator(step_builder, expansion_objective),
    state_pairs=state_pairs,
    normalize_weights=False,
    n_workers=10,
)
parameterized_problem = ParameterizedControlProblem(optimization_problem, parameterization)
penalty = ParameterSmoothPenalty(
    l1_weight=0.001,
    l2_weight=0.00015,
)
penalized_problem = PenalizedParameterizedProblem(parameterized_problem, penalty)
optimizer = ScipyOptimizer(
    method='L-BFGS-B',
    maximize=True,
    options={'maxiter': 40, 'gtol': 1e-12, 'ftol': 1e-15},
)
```

## Results

_Optimization has not completed yet. Final results will be appended here._
