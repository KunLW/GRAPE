# First-Order Lindblad Decoherence Correction for GRAPE

$$\newcommand{\dvbar}[1]{\overline{#1}}$$

Summary of the derivation in `report_opengrape_iontrap.tex` (section *Gradient with
Decoherence* and the perturbative-gradient lemma), with two corrections found during
verification (marked **[corrected]** below). This is the reference for the implementation in
`quantum_control/` (`LindbladExpansionEvolution`, `LindbladCorrectedStateFidelity`,
`LindbladExpansionDifferentiator`).

## 1. Setup

Markovian decoherence on top of the controlled Hamiltonian evolution:

$$
\dot\rho(t) = -i\,[H(t),\rho(t)] + \mathcal{L}[\rho(t)],
\qquad
\mathcal{L}[\rho] = \sum_\mu \left( L_\mu \rho L_\mu^\dagger
- \tfrac{1}{2} L_\mu^\dagger L_\mu \rho
- \tfrac{1}{2} \rho L_\mu^\dagger L_\mu \right),
$$

with rates absorbed into the jump operators, $L_\mu = \sqrt{\gamma_\mu}\, A_\mu$. Let
$U(t,0)$ solve $i\,\partial_t U = H(t)U$, and $\rho_0(t) = U(t,0)\rho(0)U^\dagger(t,0)$ be the
closed trajectory.

## 2. Interaction picture and Dyson expansion

With $\widetilde\rho(t) = U^\dagger(t,0)\rho(t)U(t,0)$ the Hamiltonian part is removed,
$\partial_t \widetilde\rho = \widetilde{\mathcal{L}}(t)[\widetilde\rho]$, where
$\widetilde{\mathcal{L}}(t)[\cdot] = U^\dagger(t,0)\,\mathcal{L}[U(t,0)\,\cdot\,U^\dagger(t,0)]\,U(t,0)$.
Iterating the integral equation and keeping first order in the decoherence strength, then
transforming back to the Schrödinger picture:
$$
\rho(T) \approx \rho_0(T) + \int_0^T U(T,s)\,\mathcal{L}[\rho_0(s)]\,U^\dagger(T,s)\,\dd s .
$$

The $U(T,s)$ factor matters: decoherence acts at the intermediate time $s$ and the remaining
control pulse propagates the correction to the final time.

## 3. GRAPE discretization

With $T = N\tau$, per-slice propagators $W_k = \exp\{-i\tau H(t_k)\}$, and

$$
\ket{F_k} = W_k \cdots W_1 \ket{\psi_0},
\qquad
\bra{B_k} = \bra{\psi_T} W_N \cdots W_{k+1},
$$

the correction to the state-transfer fidelity
$\mathcal{F} = \mel{\psi_T}{\rho(T)}{\psi_T}$ is localized on each time slice:

$$
\delta\mathcal{F}_{\mathrm{dec}}
= \tau \sum_{k=0}^{N} \mel{B_k}{\,\mathcal{L}\big[\ketbra{F_k}\big]\,}{B_k}.
$$

No density matrix or superoperator propagation is needed — only the state chains that GRAPE
already computes.

*Sampling convention:* the $k=0$ term samples decoherence at $t=0$ acting on the initial
state ($F_0 = \psi_0$, $\bra{B_0} = \bra{\psi_T}U(T,0)$). The report's discrete sum uses the
right endpoints $k=1\ldots N$; any endpoint choice for the Riemann sum differs by
$O(\tau\norm{\mathcal{L}})$, below the accuracy of the first-order correction. We include
$k=0$ here and seed the recursion accordingly in Section 6.

## 4. Scalar form

Define, per slice $k$ and channel $\mu$,

$$
a_{k\mu} = \mel{B_k}{L_\mu}{F_k},
\qquad
b_{k\mu} = \mel{B_k}{L_\mu^\dagger L_\mu}{F_k},
\qquad
s = \braket{B_k}{F_k} = \mel{\psi_T}{U(T,0)}{\psi_0},
$$

where $s$ is independent of $k$ (forward and backward pieces compose to the full propagator).
Substituting $\rho = \ketbra{F_k}$:

$$
\delta\mathcal{F}_{\mathrm{dec}}
= \tau \sum_{k=0}^{N} \sum_\mu
\Big[ \abs{a_{k\mu}}^2 - \Re\!\big( b_{k\mu}\, s^* \big) \Big],
\qquad
\mathcal{F} \approx \abs{s}^2 + \delta\mathcal{F}_{\mathrm{dec}}
+ O\!\big(\tau^2 \norm{\mathcal{L}}^2\big).
$$

Both terms are manifestly real; the correction is non-positive-leaning as expected for
decoherence acting on a near-optimal pulse.

## 5. Compact insertion operator **[corrected]**

The correction can be written as a single-insertion perturbative chain,
$\delta\mathcal{F}_{\mathrm{dec}} = \tau \sum_{k=0}^{N} \mel{B_k}{x_k}{F_k}$, with
$$
x_k = \sum_\mu \Big[
a_{k\mu}^*\, L_\mu
\;-\; \tfrac{1}{2}\, s^*\, L_\mu^\dagger L_\mu
\;-\; \tfrac{1}{2}\, b_{k\mu}^*\, \mathbb{1}
\Big].
$$

**Correction vs. the report:** the report writes the third scalar as
$\mel{B_k}{L_\mu^\dagger L_\mu}{F_k} = b_{k\mu}$ (in the lemma Example and in eq.
`multi_lindblad_decoherence_fidelity_2`); it must be the conjugate
$\mel{F_k}{L_\mu^\dagger L_\mu}{B_k} = b_{k\mu}^*$. With the report's version the last two
terms contract to $-b_{k\mu} \Re(s)$ instead of the correct $-\Re(b_{k\mu} s^*)$ — the
correction loses realness and any gradient built on it is wrong.

## 6. Gradient: frozen-coefficient recursion

Treat $V_k = x_k W_k$ as the insertion in the report's lemma and run the standard
single-insertion recursions,

$$
\ket{F_k^{(1)}} = W_k \ket{F_{k-1}^{(1)}} + x_k W_k \ket{F_{k-1}^{(0)}},
\qquad \ket{F_0^{(1)}} = x_0 \ket{F_0^{(0)}},
$$

$$
\bra{B_k^{(1)}} = \bra{B_{k+1}^{(1)}} W_{k+1} + \bra{B_{k+1}^{(0)}} x_{k+1} W_{k+1},
\qquad \bra{B_N^{(1)}} = 0,
$$

so that $\delta\mathcal{F}_{\mathrm{dec}} = \tau \braket{\psi_T}{F_N^{(1)}}$ (real up to
roundoff with the corrected $x_k$). The seed $\ket{F_0^{(1)}} = x_0\ket{F_0^{(0)}}$ carries
the $k=0$ (decoherence at $t=0$) sample of Section 3; the backward seed stays
$\bra{B_N^{(1)}} = 0$ because insertions at $j \le k$ are already carried by the forward
states in the contraction below. The derivative w.r.t. control $c_i(k)$, with
$\partial W_k \equiv \pdv{W_k}{c_i(k)}$ (exact Fréchet derivative, or $-i\tau H_i W_k$ to
first order in $\tau$), is

$$
\pdv{\mathcal{F}}{c_i(k)}
= 2\Re\Big\{ s^*\, \mel{B_k^{(0)}}{\partial W_k}{F_{k-1}^{(0)}} \Big\}
+ 2\tau\,\Re\Big\{
\mel{B_k^{(0)}}{\partial W_k}{F_{k-1}^{(1)}}
+ \mel{B_k^{(0)}}{x_k\, \partial W_k}{F_{k-1}^{(0)}}
+ \mel{B_k^{(1)}}{\partial W_k}{F_{k-1}^{(0)}}
\Big\}.
$$

**Why this is exact although $x_j$ depends on the controls** (the *frozen-coefficient*
argument): the scalars $a_{j\mu}^*, s^*, b_{j\mu}^*$ inside $x_j$ depend on all $c_i(k)$, and
the recursion above deliberately does **not** differentiate them. Term-by-term:

- $\abs{a_{j\mu}}^2$ term: freezing $a_{j\mu}^*$ and differentiating the operator chain gives
  $a_{j\mu}^*\,\partial a_{j\mu}$; the overall $2\Re\{\cdot\}$ supplies the conjugate half, and
  $2\Re\{a^*\partial a\} = \partial \abs{a}^2$ exactly.
- $-\Re(b_{j\mu} s^*)$ terms: the frozen $-\tfrac12 s^*$ piece contributes
  $-\tfrac12 s^* \partial b_{j\mu}$ and the frozen $-\tfrac12 b_{j\mu}^*$ piece contributes
  $-\tfrac12 b_{j\mu}^* \partial s$; after $2\Re\{\cdot\}$ these give
  $-\Re(s^*\partial b) - \Re(b^*\partial s) = -\partial\,\Re(b\, s^*)$ exactly.

So no $\partial x/\partial c$ terms are needed — but only with the corrected conjugations of
Section 5. **[corrected]** The report's final gradient line
$\braket*{\dvbar{B_k^{(1)}}}{F_k}$ should read
$\braket*{B_k^{(1)}}{\dvbar{F_k^{(0)}}}$, consistent with its own lemma
($\dvbar{\mathcal{F}^{(l)}} = \sum_{m+n=l} \braket*{B_k^{(m)}}{\dvbar{F_k^{(n)}}}$, here with
the additional $+\,\mathrm{c.c.}$ from the frozen coefficients).

## 7. Algorithm (two passes, $O(N d^2)$ per channel)

1. Build $W_k$ for all slices; propagate plain chains $F_k^{(0)}$, $B_k^{(0)}$.
2. Compute the scalars $s$, $a_{k\mu}$, $b_{k\mu}$ for $k = 0 \ldots N$; assemble $x_k$.
3. Run the single-insertion recursions for $F_k^{(1)}$, $B_k^{(1)}$, seeding
   $F_0^{(1)} = x_0 F_0^{(0)}$.
4. Value: $\mathcal{F} = \abs{s}^2 + \tau \Re \braket{\psi_T}{F_N^{(1)}}$.
5. Gradient: contract per slice/control with $\partial W_k$ as in Section 6.

## 8. Validity

- Trotter/first-order propagator: $\kappa_1 = \tau \norm{H} \ll 1$.
- First-order Dyson truncation: $\kappa_3 = T \norm{\mathcal{L}} < 1$, with the neglected
  term $O\big((T\norm{\mathcal{L}})^2\big)$; here
  $\norm{\mathcal{L}} \sim \sum_\mu \norm{L_\mu^\dagger L_\mu}_2$.
- For a gate fidelity, average the state-transfer correction over the logical test-state
  pairs (36 pairs for a two-qubit gate), as in the closed-fidelity construction.
