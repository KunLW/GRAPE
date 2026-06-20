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
    IonTrapRFSystem,
    PerturbativeExpansionDifferentiator,
    PerturbativeExpansionEvolution,
    PerturbativeStepBuilder,
    PiecewiseConstantPulse,
)

sx = np.array([[0, 1], [1, 0]], dtype=complex)
sz = np.array([[1, 0], [0, -1]], dtype=complex)

system = IonTrapRFSystem(
    drift=np.zeros((2, 2), dtype=complex),
    controls=[sx],
    static_fluctuations=[0.01 * sz],
    control_fluctuations=[0.02 * sx],
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
`H(t) = alpha_1(t) I_spin ⊗ a†a + alpha_2(t) S_phi ⊗ (a + a†)`
as a two-channel fluctuating closed system. The pulse array has shape
`(n_steps, 2)`; column 0 is `alpha_1(t)` and column 1 is `alpha_2(t)`.
Optional `static_fluctuations` and `control_fluctuations` use the same
already-scaled `sigma H` convention as `IonTrapRFSystem`.

```python
import numpy as np

from quantum_control import (
    ControlProblem,
    EvolutionContext,
    NominalUnitaryEvolution,
    PiecewiseConstantPulse,
    StateTransferFidelity,
    UnitaryStepBuilder,
    spin_boson_control_system,
)
from quantum_control.differentiators.finite_difference import FiniteDifferenceDifferentiator

n_levels = 3
system = spin_boson_control_system(n_levels=n_levels, phi_s=0.0)
alpha = np.column_stack(
    [
        np.full(20, 0.2),  # alpha_1(t)
        np.full(20, 0.1),  # alpha_2(t)
    ]
)
pulse = PiecewiseConstantPulse(alpha, dt=0.05)
context = EvolutionContext(
    initial_state=np.eye(2 * n_levels, dtype=complex)[0],
    target_state=np.eye(2 * n_levels, dtype=complex)[n_levels + 1],
)

step_builder = UnitaryStepBuilder()
evolution = NominalUnitaryEvolution(step_builder)
objective = StateTransferFidelity(context.target_state)
differentiator = FiniteDifferenceDifferentiator(evolution, objective)

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
