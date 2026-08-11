# Error Budget

## Metadata

- `n_steps`: 400
- `n_controls`: 2
- `dt`: 1.9224703803964076e-07
- `n_state_pairs`: 96
- `gradient_samples`: 16
- `fluctuation_scales`: (0.25, 0.5, 1.0)
- `random_seed`: 12345

## Metrics

| category | metric | scale | value | available | notes |
|---|---:|---:|---:|---|---|
| W | unitarity_fro_mean |  | 5.46310591457e-16 | True | nominal expm W^dagger W - I |
| W | unitarity_fro_max |  | 1.02378907537e-15 | True | nominal expm W^dagger W - I |
| dW | sample_count |  | 16 | True | sampled gradient coordinates |
| dW | norm_first_minus_fd |  | 1.17501638564e-09 | True |  |
| dW | norm_frechet_minus_fd |  | 1.17468554083e-09 | True |  |
| dW | norm_first_minus_frechet |  | 9.02611740988e-13 | True |  |
| dW | relative_first_minus_fd |  | 1.17501638564e-09 | True |  |
| dW | relative_frechet_minus_fd |  | 1.17468554083e-09 | True |  |
| V | relative_fro_mean |  | 0.00225541916729 | True |  |
| V | relative_fro_median |  | 0.00239197205743 | True |  |
| V | relative_fro_max |  | 0.00268915563479 | True |  |
| V | relative_spectral_mean |  | 0.00228166184702 | True |  |
| V | relative_spectral_max |  | 0.00269173167732 | True |  |
| V | fidelity_leading |  | 0.999997984297 | True |  |
| V | fidelity_frechet |  | 0.99999798593 | True |  |
| V | fidelity_leading_minus_frechet |  | -1.63292623867e-09 | True |  |
| truncation | perturbative_fidelity | 0.25 | 0.999999320254 | True |  |
| truncation | sigma_rms_spectral | 0.25 | 41.6030261872 | True |  |
| truncation | sigma_mean_spectral | 0.25 | 41.5726703293 | True |  |
| truncation | sigma_max_spectral | 0.25 | 44.3923152129 | True |  |
| truncation | total_time | 0.25 | 7.68988152159e-05 | True |  |
| truncation | sigmaT_squared_estimate | 0.25 | 1.02350305115e-05 | True | (sigma_rms_spectral * total_time)^2 |
| truncation | perturbative_fidelity | 0.5 | 0.999999053063 | True |  |
| truncation | sigma_rms_spectral | 0.5 | 83.2060523744 | True |  |
| truncation | sigma_mean_spectral | 0.5 | 83.1453406587 | True |  |
| truncation | sigma_max_spectral | 0.5 | 88.7846304259 | True |  |
| truncation | total_time | 0.5 | 7.68988152159e-05 | True |  |
| truncation | sigmaT_squared_estimate | 0.5 | 4.09401220459e-05 | True | (sigma_rms_spectral * total_time)^2 |
| truncation | perturbative_fidelity | 1.0 | 0.999997984297 | True |  |
| truncation | sigma_rms_spectral | 1.0 | 166.412104749 | True |  |
| truncation | sigma_mean_spectral | 1.0 | 166.290681317 | True |  |
| truncation | sigma_max_spectral | 1.0 | 177.569260852 | True |  |
| truncation | total_time | 1.0 | 7.68988152159e-05 | True |  |
| truncation | sigmaT_squared_estimate | 1.0 | 0.000163760488184 | True | (sigma_rms_spectral * total_time)^2 |

## Summary

| item | value | source metric |
|---|---:|---|
| W error | 1.02378907537e-15 | `W/unitarity_fro_max` |
| dW error | 1.17501638564e-09 | `dW/relative_first_minus_fd` |
| V fidelity error | -1.63292623867e-09 | `V/fidelity_leading_minus_frechet` |
| truncation fidelity error | 0.000163760488184 | `truncation/sigmaT_squared_estimate[scale=1.0]` |
| sigmaT squared estimate | 0.000163760488184 | `truncation/sigmaT_squared_estimate[scale=1.0]` |
| optimization perturbative fidelity | 0.999997984297 | `truncation/perturbative_fidelity[scale=1.0]` |
