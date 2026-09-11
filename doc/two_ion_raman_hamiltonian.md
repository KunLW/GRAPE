# The two-ion Raman Hamiltonian as implemented

This document states exactly the Hamiltonian that `physical_systems/two_ion_raman.py`
builds (registry key `system.type: two_ion_raman`), with pointers to the code that
constructs each piece. Physical model: Wu, Wang & Duan, *PRA* **97**, 062325 (2018) —
two ions in a linear Paul trap, each addressed by its own phase-insensitive
three-beam set (reference + blue-detuned + red-detuned beam, 6 lasers total),
driving two bichromatic Raman transitions at beat detunings $\pm\mu$.

The model is what survives after: adiabatic elimination of the excited state,
first-order Lamb–Dicke expansion, dropping the carrier (it oscillates at $\pm\mu$,
paper Eq. 10), the Mølmer–Sørensen RWA, and moving to the frame rotating at $\mu$
for all four motional modes.

## Hilbert space and conventions

$$
\mathcal{H} = \underbrace{\mathbb{C}^2 \otimes \mathbb{C}^2}_{\text{spins, ions }1,2}
\;\otimes\; \underbrace{\mathrm{Fock}(n)^{\otimes 4}}_{\text{transverse modes}},
\qquad \dim = 4\,n^4 ,
$$

with $n$ = `n_levels` (default 3). Every operator is `kron(spin_part, motion_part)`;
the flat index of $|s\rangle|n_1 n_2 n_3 n_4\rangle$ is
$s\,n^4 + ((n_1 n + n_2)n + n_3)n + n_4$.

- Qubit basis: $\sigma_z|0\rangle = +|0\rangle$, $\sigma_+ = |1\rangle\langle 0|$.
- Modes $k = 1..4$: (x-COM, x-rocking, y-COM, y-rocking). All beams are transverse
  and $\Delta k$ lies in the x–y plane, so all four transverse modes couple.
- $\hbar = 1$; everything internal is angular frequency in rad/s
  (YAML values are kHz, converted via `RAD_S_PER_KHZ`).

Building blocks (code: `collective_lowering_operator`, `pair_coupling_operators`
in `physical_systems/two_ion_raman.py:194,223`):

$$
A_j = \sum_{k=1}^{4} \eta_k\, b_j^k\, a_k ,
\qquad b_1^k = \tfrac12,\quad b_2^k = \tfrac12 s_k,
\quad s = (+1,-1,+1,-1),
$$

$$
\sigma_j^{+,\phi_s} = e^{i\phi_s}\,\sigma_+^{(j)}, \qquad
\boxed{\;B_j = \sigma_j^{+,\phi_s} A_j^\dagger + \text{h.c.}\;}
\quad\text{(blue pair, anti-JC)},
\qquad
\boxed{\;R_j = \sigma_j^{+,\phi_s} A_j + \text{h.c.}\;}
\quad\text{(red pair, JC)},
$$

$$
Z_j = \tfrac12\sigma_z^{(j)} \otimes I_{\text{motion}},
\qquad
N = I_4 \otimes \sum_{k=1}^{4} \hat n_k .
$$

At equal blue/red amplitudes the pairs combine to the MS spin–phase coupling,
$B_j + R_j = \sigma_{\phi_s}^{(j)} \otimes \sum_k \eta_k\, 2 b_j^k\, (a_k + a_k^\dagger)$
with $\sigma_\phi = \cos\phi\,\sigma_x - \sin\phi\,\sigma_y$; the default
$\phi_s = 0$ gives the XX-type MS drive.

## Nominal Hamiltonian

Piecewise-constant controls $u_i(t)$, shape `(n_steps, 12)`; the propagator is
$U = \prod_m e^{-i H(t_m)\,\Delta t}$. Every control multiplies a **constant**
operator — no control parameter ever sits inside an exponential.

$$
H(t) = H_0 + \sum_{i=1}^{12} u_i(t)\, H_i .
$$

### Drift (`_drift_and_controls`, `physical_systems/two_ion_raman.py:318`)

$$
H_0 = \sum_{k=1}^{4} \delta_k \,\big(I_4 \otimes \hat n_k\big)
\;+\; \sum_{j=1}^{2} \epsilon_j\, Z_j ,
\qquad
\delta_k = \omega_k - \mu ,
$$

where $\epsilon_j$ = `stark_shifts_khz` (constant residual differential AC-Stark
offset per ion, default 0).

### The 12 control channels

2 per beam × 6 beams; channel order is, per ion, per beam (ref, blue, red), the
(rabi, det) pair. With $\kappa$ = `stark_rabi_ratio` (drive-proportional
differential AC-Stark shift per unit effective pair Rabi, default $10^{-4}$):

| # | label | operator $H_i$ | meaning |
|---|-------------|--------------------------------------|---------|
| 0 | `rabi_ref1` | $\tfrac12(B_1 + R_1) + 2\kappa Z_1$ | reference beam feeds both of ion 1's pairs |
| 1 | `det_ref1` | $N$ | reference-beam frequency offset = drive (μ) detuning |
| 2 | `rabi_blue1` | $\tfrac12 B_1 + \kappa Z_1$ | blue-beam amplitude |
| 3 | `det_blue1` | $-\tfrac12(Z_1 + N)$ | blue-beam frequency offset |
| 4 | `rabi_red1` | $\tfrac12 R_1 + \kappa Z_1$ | red-beam amplitude |
| 5 | `det_red1` | $\tfrac12(Z_1 - N)$ | red-beam frequency offset |
| 6–11 | same for ion 2 | $B_2, R_2, Z_2, N$ | |

Notes on the channel model (module docstring, `physical_systems/two_ion_raman.py:36-66`):

- **rabi channels** are in *effective pair-Rabi* units. A pair's coupling is
  bilinear in its two beams' fields ($\Omega_1\Omega_2/2\Delta$); the linear
  model uses the contribution convention — each pair's effective Rabi is the
  **sum** of its two beams' channel amplitudes
  ($u_i \sim \bar\Omega_{\text{partner}}\Omega_i / 4\Delta$), the symmetric
  first-order linearization of the product. The drive is off at zero controls.
- The **drive-proportional AC-Stark shift** rides on the rabi channels as
  $\kappa Z_j$ per pair fed ($2\kappa Z_j$ on `rabi_ref` since the reference
  beam feeds both pairs), so the shift follows the drive amplitude.
- **det channels**: a beam-frequency offset shifts its pairs' two-photon
  detunings (blue pair: $\partial/\partial\delta_{\text{ref}} = +1$,
  $\partial/\partial\delta_{\text{blue}} = -1$; red pair:
  $\partial/\partial\delta_{\text{red}} = +1$,
  $\partial/\partial\delta_{\text{ref}} = -1$); per ion this decomposes into a
  qubit-common part $\to Z_j$ and a drive-detuning part $\to N$. There is no
  separate mode-detuning channel — its role is played by `det_ref`. Note
  `det_ref1` and `det_ref2` share the *same* operator $N$: a per-ion drive
  detuning has no exact static per-ion generator; the global $N$ is the model's
  first-order approximation.

### Grouped form

Collecting terms, the Hamiltonian the optimizer actually evolves under is

$$
H(t) =
\sum_{j=1}^{2}\left[
\frac{u^{\text{rabi}}_{\text{ref},j} + u^{\text{rabi}}_{\text{blue},j}}{2}\, B_j
+ \frac{u^{\text{rabi}}_{\text{ref},j} + u^{\text{rabi}}_{\text{red},j}}{2}\, R_j
\right]
+ \sum_{k=1}^{4}\big[\delta_k + \Delta(t)\big]\,\hat n_k
+ \sum_{j=1}^{2} \zeta_j(t)\, \frac{\sigma_z^{(j)}}{2},
$$

with the time-dependent collective coefficients

$$
\Delta(t) = \sum_{j=1}^{2}\left[u^{\text{det}}_{\text{ref},j}
- \tfrac12\big(u^{\text{det}}_{\text{blue},j} + u^{\text{det}}_{\text{red},j}\big)\right]
\qquad\text{(common drive-detuning shift on all four modes)},
$$

$$
\zeta_j(t) = \epsilon_j
+ \kappa\left(2u^{\text{rabi}}_{\text{ref},j} + u^{\text{rabi}}_{\text{blue},j}
+ u^{\text{rabi}}_{\text{red},j}\right)
+ \tfrac12\big(u^{\text{det}}_{\text{red},j} - u^{\text{det}}_{\text{blue},j}\big)
\qquad\text{(per-ion qubit shift)} .
$$

## Default parameter values

Defaults in `physical_systems/two_ion_raman.py:133-143` (override in YAML;
illustrative trap: $\omega_x/2\pi = 3.0$ MHz, $\omega_y/2\pi = 2.9$ MHz,
$\omega_z/2\pi = 0.5$ MHz):

| mode $k$ | 1 (x-COM) | 2 (x-rock) | 3 (y-COM) | 4 (y-rock) |
|---|---|---|---|---|
| $\omega_k/2\pi$ (kHz) | 3000 | 2958 | 2900 | 2857 |
| $\delta_k/2\pi = (\omega_k-\mu)/2\pi$ (kHz) | −10 | −52 | −110 | −153 |
| $\eta_k$ | 0.075 | 0.076 | 0.053 | 0.054 |
| $s_k$ (ion-2 sign) | +1 | −1 | +1 | −1 |

$\mu/2\pi = 3010$ kHz, $\phi_s = 0$, $\kappa = 10^{-4}$, $\epsilon_j = 0$.
Control bounds: rabi channels $[0, 200]$ kHz (effective pair Rabi), detuning
channels $[-60, 60]$ kHz; the six rabi columns are pinned to zero at the first
and last time step (`rabi_endpoint_zero`, default true). Default gate time
225.8 µs; target gate `ms_xx_pi_over_2` (maximally entangling XX(π/2) MS gate).

## Noise terms

Noise never changes the nominal $H(t)$ above; it enters through the two
`system.noise` subsections of the `OpenSystem`.

### Quasi-static coherent fluctuations (`fluctuation_terms`, `physical_systems/two_ion_raman.py:854`)

Shot-to-shot Gaussian variables, treated by perturbative expansion
(`max_order` in the YAML). 2 static + 12 control-kind terms:

$$
H(t) \;\to\; H(t)
\;+\; \delta_s\, \big(Z_1 + Z_2\big)
\;+\; \delta_m\, N
\;+\; \sum_{i=1}^{12} \varepsilon_i\, u_i(t)\, H_i ,
$$

- $\delta_s \sim \mathcal N(0, \sigma_{\text{spin}}^2)$ — collective spin
  dephasing; default $\sigma = 314.159$ rad/s $= 2\pi\times 50$ Hz.
- $\delta_m \sim \mathcal N(0, \sigma_{\text{motion}}^2)$ — common
  motional-frequency / μ drift on $N$; default $\sigma = 300$ rad/s.
- $\varepsilon_i \sim \mathcal N(0, \sigma_{\text{role}}^2)$ — *relative*
  amplitude noise per control channel, sigma keyed by role (one value for all
  six rabi channels, one for all six detuning channels; default $10^{-4}$
  each). The control-kind operators are byte-identical to the $H_i$ above
  (both come from `_drift_and_controls`).

### Lindblad decoherence channels (`decoherence_channels`, `physical_systems/two_ion_raman.py:909`)

First-order Lindblad correction with jump operators $L = \sqrt{\gamma}\,A$
(rates in 1/s from the YAML; zero-rate channels are dropped):

| channel | jump operator $A$ | rate |
|---|---|---|
| heating, mode $k$ | $I_{\text{spin}} \otimes a_k^\dagger$ | $\gamma_{\text{heat},k}$ (one per mode) |
| motional dephasing, mode $k$ | $I_{\text{spin}} \otimes \hat n_k$ | $\gamma_{\text{md}}$ (common) |
| collective spin dephasing | $\big(Z_1 + Z_2\big)$ | $\gamma_{\text{sd}}$ |
| Raman scattering, ion $j$ | $\sigma_+^{(j)} \otimes I$ and $\sigma_-^{(j)} \otimes I$ | $\gamma_{\text{Raman},j}/2$ each |
| Rayleigh scattering, ion $j$ | $\tfrac12\sigma_z^{(j)} \otimes I$ | $\gamma_{\text{Rayleigh},j}$ |

The spontaneous-emission rates model scattering
$\sim \gamma_e \Omega^2/\Delta^2$ from the adiabatically eliminated excited
state as constant per-ion rates supplied via the config.
