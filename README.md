# GRAPE VERGE

This repository contains a small modular quantum control engine. The important
design choice is that fluctuation approximation is represented as perturbative
expansion evolution, not as a single propagator.

The data flow is:

```text
System + Pulse -> StepBuilder -> Evolution -> Objective -> Differentiator -> Optimizer
```

The implemented perturbative path returns expansion components such as
`F`, `SF`, `DF`, and optional backward components, so higher-order extensions can
be added without changing objective or optimizer code.

This perturbative path implements the long-time-correlation fluctuation limit
from `doc/report_opengrape_iontrap.tex`. Static fluctuation matrices represent
`sigma_xi H_xi`; control fluctuation matrices represent `sigma_chi_i H_chi_i`;
the fluctuation Hamiltonian is
`sum static_fluctuations + sum control_i * control_fluctuation_i`. The
short-time-correlation decoherence limit belongs to the Lindblad/density-matrix
path and is intentionally left as a future module.

## Quick Example

```python
import numpy as np

from quantum_control import (
    ControlProblem,
    EvolutionContext,
    ExpansionFidelity,
    FluctuationTerm,
    OpenSystem,
    PerturbativeExpansionDifferentiator,
    PerturbativeExpansionEvolution,
    PerturbativeStepBuilder,
    PiecewiseConstantPulse,
)

sx = np.array([[0, 1], [1, 0]], dtype=complex)
sz = np.array([[1, 0], [0, -1]], dtype=complex)

system = OpenSystem(
    drift=np.zeros((2, 2), dtype=complex),
    controls=[sx],
    noise_terms=[
        FluctuationTerm(name="dephasing", operator=sz, definition="sigma_z", coefficient=0.01, kind="static"),
        FluctuationTerm(name="amplitude", operator=sx, definition="sigma_x", coefficient=0.02, kind="control"),
    ],
)
pulse = PiecewiseConstantPulse(np.full((20, 1), 0.1), dt=0.05)
context = EvolutionContext(
    initial_state=np.array([1, 0], dtype=complex),
    target_state=np.array([0, 1], dtype=complex),
)

step_builder = PerturbativeStepBuilder()
evolution = PerturbativeExpansionEvolution(step_builder, max_order=2)
objective = ExpansionFidelity(max_order=2)
differentiator = PerturbativeExpansionDifferentiator(step_builder, objective)

problem = ControlProblem(
    system=system,
    pulse=pulse,
    context=context,
    evolution=evolution,
    objective=objective,
    differentiator=differentiator,
)

value = problem.value()
gradient = problem.gradient()
```

## CLI Usage

Run the perturbative spin-boson open-gate optimizer from the repository root:

```bash
.venv/bin/python experiments/spin_boson_perturbative_lbfgsb.py \
  --maxiter 40 \
  --n-steps 200 \
  --workers 1
```

Each run writes a timestamped directory under `experiments/outputs/`. The
`report.md` file is created before optimization starts with a preview of the
configuration, output paths, system construction script, noise terms, and
`kappa_1`/`kappa_2` diagnostics. When optimization finishes or is interrupted,
the final results are appended to the same report. Checkpoint files
`latest_pulse.npz`, `latest_pulse.csv`, and `latest_parameters.npz` are updated
during optimization.

Useful options:

```text
--maxiter N                 L-BFGS-B iteration limit.
--n-steps N                 Number of piecewise-constant pulse slices.
--alpha1-cycles X           Initial alpha1 cosine cycles.
--l1-smooth-weight W        First-difference smoothness penalty.
--l2-smooth-weight W        Second-difference smoothness penalty.
--workers N                 Worker processes for state-pair averaging.
--print-step                Print per-step fidelity/objective diagnostics.
--print-fidelity-terms      Print and save perturbative fidelity terms.
--initial-pulse-npz PATH    Start from an exported pulse .npz.
--no-progress               Disable the progress bar.
```

Run an initial-condition sweep:

```bash
.venv/bin/python experiments/spin_boson_perturbative_initial_sweep.py \
  --initial-mode all \
  --n-runs 4 \
  --seed 12345 \
  --maxiter 40 \
  --sweep-workers 2
```

`--initial-mode` can be `noise`, `random`, `custom`, `both`, or `all`. Custom
initial pulses can be supplied by repeating `--initial-pulse-npz PATH`; the
loaded pulse must match the configured shape and keep the alpha2 endpoints at
zero.

## Averaging Multiple State Pairs

`ExpansionStateAverageFidelity` evaluates the same perturbative objective over
multiple state pairs and returns the weighted average. This matches the
state-pair averaging used by the open-gate fidelity notes while keeping each
individual evolution as a single-state propagation.

```python
from quantum_control import ExpansionStateAverageFidelity

averaged_problem = ExpansionStateAverageFidelity(
    system=system,
    pulse=pulse,
    evolution=evolution,
    objective=objective,
    differentiator=differentiator,
    state_pairs=[
        (initial_state_0, target_state_0, 1.0),
        (initial_state_1, target_state_1, 1.0),
    ],
)

value = averaged_problem.value()
gradient = averaged_problem.gradient()
```

## Spin-Boson Control Hamiltonian

The spin-boson helper builds the Hamiltonian
`H(t) = alpha_1(t) I_spin ⊗ a†a + alpha_2(t) eta S_phi ⊗ X1`,
where `X1 = (a† + a) / 2` and the default `eta = 0.075`. It is represented
as a two-channel system with two spin qubits (a `ClosedSystem`, or an
`OpenSystem` when noise terms are attached). The spin term is
`S_phi = b_1 sigma_phi ⊗ I + b_2 I ⊗ sigma_phi`; the default stretch-mode
vector is `b = (1, -1) / 2`, and the COM-mode vector is `b = (1, 1) / 2`.
The pulse array has shape `(n_steps, 2)`; column 0 is `alpha_1(t)` and column 1
is `alpha_2(t)`.
Optional `static_fluctuations` and `control_fluctuations` use the
already-scaled `sigma H` convention (wrapped into unit-strength
`FluctuationTerm`s on an `OpenSystem`). The pulse helper takes
user-facing bounds in kHz and total time in microseconds, then stores amplitudes
as angular frequencies in rad/s and `dt` in seconds.

```python
import numpy as np

from quantum_control import (
    ControlProblem,
    EvolutionContext,
    GrapeDifferentiator,
    NominalUnitaryEvolution,
    ParameterizedControlProblem,
    StateTransferFidelity,
    UnitaryStepBuilder,
)
from physical_systems.spin_boson import (
    spin_boson_control_system,
    spin_boson_initial_pulse,
    spin_boson_parameterization,
)

n_levels = 3
system = spin_boson_control_system(n_levels=n_levels, phi_s=0.0)
pulse = spin_boson_initial_pulse()
parameterization = spin_boson_parameterization(pulse.n_steps)
context = EvolutionContext(
    initial_state=np.eye(4 * n_levels, dtype=complex)[0],
    target_state=np.eye(4 * n_levels, dtype=complex)[3 * n_levels + 1],
)

step_builder = UnitaryStepBuilder()
evolution = NominalUnitaryEvolution(step_builder)
objective = StateTransferFidelity(context.target_state)
differentiator = GrapeDifferentiator(step_builder)

problem = ControlProblem(
    system=system,
    pulse=pulse,
    context=context,
    evolution=evolution,
    objective=objective,
    differentiator=differentiator,
)

value = problem.value()
gradient = problem.gradient()

parameterized_problem = ParameterizedControlProblem(problem, parameterization)
parameters = parameterized_problem.initial_parameters()
```

## Parameter Smooth Penalties

Smooth penalties can be applied directly to normalized optimization parameters.
Use separate weights for the L1 first-difference term and the L2
second-difference term:

```python
from quantum_control import (
    ParameterSmoothPenalty,
    ParameterizedControlProblem,
    PenalizedParameterizedProblem,
)

parameterized_problem = ParameterizedControlProblem(problem, parameterization)
penalty = ParameterSmoothPenalty(
    l1_weight=1e-3,
    l2_weight=1e-4,
)
penalized_problem = PenalizedParameterizedProblem(parameterized_problem, penalty)
```

The L1 term is `sum(abs(diff(parameters)))`, and the L2 term is
`sum(diff(parameters, n=2)**2)`. Both are evaluated in normalized parameter
space, not in physical amplitude units.

## Pulse Slew-Rate Control

Slew-rate control is not implemented as projection. A pulse that violates
`max_delta` is left unchanged, and the restriction is handled either as a smooth
penalty term or as explicit optimizer constraints.

```python
from quantum_control import ParameterizedControlProblem, PulseConstraints

constraints = PulseConstraints.from_slew_rate(
    dt=pulse.dt,
    max_slew_rate=0.2,
)
parameterized_problem = ParameterizedControlProblem(
    problem,
    parameterization,
    constraints=constraints,
    penalty_weight=10.0,
)
```
