# Error Budget

## Metadata

- `n_steps`: 400
- `n_controls`: 2
- `dt`: 3.0503326948555833e-07
- `n_state_pairs`: 96
- `gradient_samples`: 16
- `fluctuation_scales`: (0.25, 0.5, 1.0)
- `random_seed`: 12345

## Metrics

| category | metric | scale | value | available | notes |
|---|---:|---:|---:|---|---|
| W | unitarity_fro_mean |  | 5.54140088839e-16 | True | nominal expm W^dagger W - I |
| W | unitarity_fro_max |  | 1.06577079883e-15 | True | nominal expm W^dagger W - I |
| dW | sample_count |  | 16 | True | sampled gradient coordinates |
| dW | norm_first_minus_fd |  | 2.28997061083e-09 | True |  |
| dW | norm_frechet_minus_fd |  | 2.28819747564e-09 | True |  |
| dW | norm_first_minus_frechet |  | 3.06628755436e-12 | True |  |
| dW | relative_first_minus_fd |  | 2.28997061083e-09 | True |  |
| dW | relative_frechet_minus_fd |  | 2.28819747564e-09 | True |  |
| V | relative_fro_mean |  | 0.00318580916372 | True |  |
| V | relative_fro_median |  | 0.00349786837693 | True |  |
| V | relative_fro_max |  | 0.00376495714291 | True |  |
| V | relative_spectral_mean |  | 0.00325396759634 | True |  |
| V | relative_spectral_max |  | 0.00382074715202 | True |  |
| V | fidelity_leading |  | 0.999996030731 | True |  |
| V | fidelity_frechet |  | 0.999996035428 | True |  |
| V | fidelity_leading_minus_frechet |  | -4.69724081853e-09 | True |  |
| truncation | perturbative_fidelity | 0.25 | 0.999999501387 | True |  |
| truncation | sigma_rms_spectral | 0.25 | 43.456528585 | True |  |
| truncation | sigma_mean_spectral | 0.25 | 43.4361402987 | True |  |
| truncation | sigma_max_spectral | 0.25 | 45.5381001394 | True |  |
| truncation | total_time | 0.25 | 0.000122013307794 | True |  |
| truncation | sigmaT_squared_estimate | 0.25 | 2.81141180326e-05 | True | (sigma_rms_spectral * total_time)^2 |
| truncation | perturbative_fidelity | 0.5 | 0.999998807256 | True |  |
| truncation | sigma_rms_spectral | 0.5 | 86.91305717 | True |  |
| truncation | sigma_mean_spectral | 0.5 | 86.8722805973 | True |  |
| truncation | sigma_max_spectral | 0.5 | 91.0762002787 | True |  |
| truncation | total_time | 0.5 | 0.000122013307794 | True |  |
| truncation | sigmaT_squared_estimate | 0.5 | 0.00011245647213 | True | (sigma_rms_spectral * total_time)^2 |
| truncation | perturbative_fidelity | 1.0 | 0.999996030731 | True |  |
| truncation | sigma_rms_spectral | 1.0 | 173.82611434 | True |  |
| truncation | sigma_mean_spectral | 1.0 | 173.744561195 | True |  |
| truncation | sigma_max_spectral | 1.0 | 182.152400557 | True |  |
| truncation | total_time | 1.0 | 0.000122013307794 | True |  |
| truncation | sigmaT_squared_estimate | 1.0 | 0.000449825888521 | True | (sigma_rms_spectral * total_time)^2 |

## Summary

| item | value | source metric |
|---|---:|---|
| W error | 1.06577079883e-15 | `W/unitarity_fro_max` |
| dW error | 2.28997061083e-09 | `dW/relative_first_minus_fd` |
| V fidelity error | -4.69724081853e-09 | `V/fidelity_leading_minus_frechet` |
| truncation fidelity error | 0.000449825888521 | `truncation/sigmaT_squared_estimate[scale=1.0]` |
| sigmaT squared estimate | 0.000449825888521 | `truncation/sigmaT_squared_estimate[scale=1.0]` |
| optimization perturbative fidelity | 0.999996030731 | `truncation/perturbative_fidelity[scale=1.0]` |
