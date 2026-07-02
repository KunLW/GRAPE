# Error Budget

## Metadata

- `n_steps`: 200
- `n_controls`: 2
- `dt`: 1.1290000000000001e-06
- `n_state_pairs`: 96
- `gradient_samples`: 4
- `fluctuation_scales`: (0.25, 0.5, 1.0)
- `random_seed`: 12345
- `pulse_npz`: experiments/outputs/spin_boson_perturbative_20260702_104808/final_pulse.npz
- `system_factory`: experiments.spin_boson_error_factory:build

## Metrics

| category | metric | scale | value | available | notes |
|---|---:|---:|---:|---|---|
| W | unitarity_fro_mean |  | 6.81645915754e-15 | True | nominal expm W^dagger W - I |
| W | unitarity_fro_max |  | 1.76574912445e-14 | True | nominal expm W^dagger W - I |
| dW | sample_count |  | 4 | True | sampled gradient coordinates |
| dW | norm_first_minus_fd |  | 1.53021966477e-09 | True |  |
| dW | norm_frechet_minus_fd |  | 1.48735449822e-09 | True |  |
| dW | norm_first_minus_frechet |  | 5.10295766314e-10 | True |  |
| dW | relative_first_minus_fd |  | 1.53021966477e-09 | True |  |
| dW | relative_frechet_minus_fd |  | 1.48735449822e-09 | True |  |
| V | relative_fro_mean |  | 0.0440905638268 | True |  |
| V | relative_fro_median |  | 0.047328379165 | True |  |
| V | relative_fro_max |  | 0.0748327500762 | True |  |
| V | relative_spectral_mean |  | 0.0527864049549 | True |  |
| V | relative_spectral_max |  | 0.0896938549772 | True |  |
| V | fidelity_leading |  | 0.999301939689 | True |  |
| V | fidelity_frechet |  | 0.999303714438 | True |  |
| V | fidelity_leading_minus_frechet |  | -1.77474929963e-06 | True |  |
| truncation | perturbative_fidelity | 0.25 | 0.999955819242 | True |  |
| truncation | sigma_rms_spectral | 0.25 | 1572.75851654 | True |  |
| truncation | sigma_mean_spectral | 0.25 | 1572.75851654 | True |  |
| truncation | sigma_max_spectral | 0.25 | 1572.75851654 | True |  |
| truncation | total_time | 0.25 | 0.0002258 | True |  |
| truncation | sigmaT_squared_estimate | 0.25 | 0.126116516463 | True | (sigma_rms_spectral * total_time)^2 |
| truncation | perturbative_fidelity | 0.5 | 0.999825043331 | True |  |
| truncation | sigma_rms_spectral | 0.5 | 3145.51703308 | True |  |
| truncation | sigma_mean_spectral | 0.5 | 3145.51703308 | True |  |
| truncation | sigma_max_spectral | 0.5 | 3145.51703308 | True |  |
| truncation | total_time | 0.5 | 0.0002258 | True |  |
| truncation | sigmaT_squared_estimate | 0.5 | 0.504466065851 | True | (sigma_rms_spectral * total_time)^2 |
| truncation | perturbative_fidelity | 1.0 | 0.999301939689 | True |  |
| truncation | sigma_rms_spectral | 1.0 | 6291.03406615 | True |  |
| truncation | sigma_mean_spectral | 1.0 | 6291.03406615 | True |  |
| truncation | sigma_max_spectral | 1.0 | 6291.03406615 | True |  |
| truncation | total_time | 1.0 | 0.0002258 | True |  |
| truncation | sigmaT_squared_estimate | 1.0 | 2.0178642634 | True | (sigma_rms_spectral * total_time)^2 |

## Summary

| item | value | source metric |
|---|---:|---|
| W error | 1.76574912445e-14 | `W/unitarity_fro_max` |
| dW error | 1.53021966477e-09 | `dW/relative_first_minus_fd` |
| V fidelity error | -1.77474929963e-06 | `V/fidelity_leading_minus_frechet` |
| truncation fidelity error | 2.0178642634 | `truncation/sigmaT_squared_estimate[scale=1.0]` |
| sigmaT squared estimate | 2.0178642634 | `truncation/sigmaT_squared_estimate[scale=1.0]` |
| optimization perturbative fidelity | 0.999301939689 | `truncation/perturbative_fidelity[scale=1.0]` |
