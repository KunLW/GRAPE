# Error Budget

## Metadata

- `n_steps`: 200
- `n_controls`: 2
- `dt`: 1.1290000000000001e-06
- `n_state_pairs`: 96
- `gradient_samples`: 2
- `mc_samples`: 4
- `fluctuation_scales`: (0.5, 1.0)
- `random_seed`: 12345
- `pulse_npz`: experiments/outputs/spin_boson_perturbative_20260701_110802/latest_pulse.npz
- `label`: quick_latest_spin_boson

## Metrics

| category | metric | scale | value | available | notes |
|---|---:|---:|---:|---|---|
| dW | sample_count |  | 2 | True | sampled gradient coordinates |
| dW | norm_first_minus_fd |  | 1.86746679589e-08 | True |  |
| dW | norm_frechet_minus_fd |  | 2.14938671816e-10 | True |  |
| dW | norm_first_minus_frechet |  | 1.87627060561e-08 | True |  |
| dW | relative_first_minus_fd |  | 1.86746679589e-08 | True |  |
| dW | relative_frechet_minus_fd |  | 2.14938671816e-10 | True |  |
| V | relative_fro_mean |  | 0.0270858664094 | True |  |
| V | relative_fro_median |  | 0.025813285541 | True |  |
| V | relative_fro_max |  | 0.0707978820956 | True |  |
| V | relative_spectral_mean |  | 0.0321395693317 | True |  |
| V | relative_spectral_max |  | 0.075947206086 | True |  |
| V | fidelity_leading |  | 0.295644696366 | True |  |
| V | fidelity_frechet |  | 0.295743335724 | True |  |
| V | fidelity_leading_minus_frechet |  | -9.86393577534e-05 | True |  |
| truncation | perturbative_fidelity | 0.5 | 0.224584330408 | True |  |
| truncation | mc_exact_fidelity | 0.5 | 0.211561271958 | True |  |
| truncation | perturbative_minus_mc | 0.5 | 0.0130230584492 | True |  |
| truncation | abs_perturbative_minus_mc | 0.5 | 0.0130230584492 | True |  |
| truncation | perturbative_fidelity | 1.0 | 0.295644696366 | True |  |
| truncation | mc_exact_fidelity | 1.0 | 0.240906879637 | True |  |
| truncation | perturbative_minus_mc | 1.0 | 0.0547378167291 | True |  |
| truncation | abs_perturbative_minus_mc | 1.0 | 0.0547378167291 | True |  |
