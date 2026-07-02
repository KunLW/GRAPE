# Error Budget

## Metadata

- `n_steps`: 200
- `n_controls`: 2
- `dt`: 1.1290000000000001e-06
- `n_state_pairs`: 96
- `gradient_samples`: 4
- `fluctuation_scales`: (0.25, 0.5, 1.0)
- `random_seed`: 12345
- `pulse_npz`: experiments/outputs/spin_boson_perturbative_sweep_20260701_141220/noise_seed_12345/final_pulse.npz
- `label`: sweep_20260701_141220_no_mc

## Metrics

| category | metric | scale | value | available | notes |
|---|---:|---:|---:|---|---|
| W | unitarity_fro_mean |  | 8.21793161387e-15 | True | nominal expm W^dagger W - I |
| W | unitarity_fro_max |  | 2.20732186461e-14 | True | nominal expm W^dagger W - I |
| dW | sample_count |  | 4 | True | sampled gradient coordinates |
| dW | norm_first_minus_fd |  | 1.03777222402e-07 | True |  |
| dW | norm_frechet_minus_fd |  | 1.07937141639e-09 | True |  |
| dW | norm_first_minus_frechet |  | 1.03321971487e-07 | True |  |
| dW | relative_first_minus_fd |  | 1.03777222402e-07 | True |  |
| dW | relative_frechet_minus_fd |  | 1.07937141639e-09 | True |  |
| V | relative_fro_mean |  | 0.0294289996796 | True |  |
| V | relative_fro_median |  | 0.0284894834799 | True |  |
| V | relative_fro_max |  | 0.0703360577148 | True |  |
| V | relative_spectral_mean |  | 0.0354986593731 | True |  |
| V | relative_spectral_max |  | 0.0754167584536 | True |  |
| V | fidelity_leading |  | 0.990248178117 | True |  |
| V | fidelity_frechet |  | 0.990321960984 | True |  |
| V | fidelity_leading_minus_frechet |  | -7.37828671924e-05 | True |  |
| truncation | perturbative_fidelity | 0.25 | 0.996067436618 | True |  |
| truncation | sigma_rms_spectral | 0.25 | 3457.6435166 | True |  |
| truncation | sigma_mean_spectral | 0.25 | 3303.14836239 | True |  |
| truncation | sigma_max_spectral | 0.25 | 4964.43159899 | True |  |
| truncation | total_time | 0.25 | 0.0002258 | True |  |
| truncation | sigmaT_squared_estimate | 0.25 | 0.609548554992 | True | (sigma_rms_spectral * total_time)^2 |
| truncation | perturbative_fidelity | 0.5 | 0.994903584917 | True |  |
| truncation | sigma_rms_spectral | 0.5 | 6915.2870332 | True |  |
| truncation | sigma_mean_spectral | 0.5 | 6606.29672478 | True |  |
| truncation | sigma_max_spectral | 0.5 | 9928.86319797 | True |  |
| truncation | total_time | 0.5 | 0.0002258 | True |  |
| truncation | sigmaT_squared_estimate | 0.5 | 2.43819421997 | True | (sigma_rms_spectral * total_time)^2 |
| truncation | perturbative_fidelity | 1.0 | 0.990248178117 | True |  |
| truncation | sigma_rms_spectral | 1.0 | 13830.5740664 | True |  |
| truncation | sigma_mean_spectral | 1.0 | 13212.5934496 | True |  |
| truncation | sigma_max_spectral | 1.0 | 19857.7263959 | True |  |
| truncation | total_time | 1.0 | 0.0002258 | True |  |
| truncation | sigmaT_squared_estimate | 1.0 | 9.75277687988 | True | (sigma_rms_spectral * total_time)^2 |

## Summary

| item | value | source metric |
|---|---:|---|
| W error | 2.20732186461e-14 | `W/unitarity_fro_max` |
| dW error | 1.03777222402e-07 | `dW/relative_first_minus_fd` |
| V fidelity error | -7.37828671924e-05 | `V/fidelity_leading_minus_frechet` |
| truncation fidelity error | 9.75277687988 | `truncation/sigmaT_squared_estimate[scale=1.0]` |
| sigmaT squared estimate | 9.75277687988 | `truncation/sigmaT_squared_estimate[scale=1.0]` |
| optimization perturbative fidelity | 0.990248178117 | `truncation/perturbative_fidelity[scale=1.0]` |
