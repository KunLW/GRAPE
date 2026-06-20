import numpy as np

from quantum_control import (
    ControlProblem,
    EvolutionContext,
    ExpansionFidelity,
    ExpansionStateAverageFidelity,
    IonTrapRFSystem,
    NominalUnitaryEvolution,
    ParameterizedControlProblem,
    PerturbativeExpansionDifferentiator,
    PerturbativeExpansionEvolution,
    PerturbativeStepBuilder,
    PiecewiseConstantPulse,
    PulseConstraints,
    StateTransferFidelity,
    StatePair,
    UnitaryStepBuilder,
    annihilation_operator,
    creation_operator,
    endpoint_masked_parameterization,
    number_operator,
    spin_boson_control_system,
    spin_phase_operator,
)
from quantum_control.differentiators.finite_difference import FiniteDifferenceDifferentiator


def two_level_problem(amplitudes, initial_state=None, target_state=None):
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    system = IonTrapRFSystem(
        drift=0.1 * sz,
        controls=[sx],
        static_fluctuations=[0.01 * sz],
        control_fluctuations=[0.02 * sx],
    )
    pulse = PiecewiseConstantPulse(np.asarray(amplitudes, dtype=float), dt=0.05)
    context = EvolutionContext(
        initial_state=np.array([1.0, 0.0], dtype=complex)
        if initial_state is None
        else np.asarray(initial_state, dtype=complex),
        target_state=np.array([0.0, 1.0], dtype=complex)
        if target_state is None
        else np.asarray(target_state, dtype=complex),
        compute_backward=True,
    )
    step_builder = PerturbativeStepBuilder()
    evolution = PerturbativeExpansionEvolution(step_builder, max_order=2)
    objective = ExpansionFidelity(max_order=2, drop_odd_average=True)
    differentiator = PerturbativeExpansionDifferentiator(step_builder, objective)
    return ControlProblem(
        system=system,
        pulse=pulse,
        context=context,
        evolution=evolution,
        objective=objective,
        differentiator=differentiator,
    )


def test_spin_boson_operators_use_truncated_oscillator_conventions():
    a = annihilation_operator(4)
    expected_a = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, np.sqrt(2.0), 0.0],
            [0.0, 0.0, 0.0, np.sqrt(3.0)],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=complex,
    )
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)

    assert a.shape == (4, 4)
    assert np.allclose(a, expected_a)
    assert np.allclose(np.diag(number_operator(4)), np.arange(4))
    assert np.allclose(spin_phase_operator(0.0), sx)
    assert np.allclose(spin_phase_operator(np.pi / 2.0), sy)


def test_spin_boson_control_system_builds_two_control_hamiltonians():
    n_levels = 3
    phi_s = 0.2
    alpha_1 = 0.3
    alpha_2 = -0.4
    system = spin_boson_control_system(n_levels=n_levels, phi_s=phi_s)
    a = annihilation_operator(n_levels)
    adag = a.conj().T
    expected_hamiltonian = (
        alpha_1 * np.kron(np.eye(2), adag @ a)
        + alpha_2 * np.kron(spin_phase_operator(phi_s), a + adag)
    )

    assert system.drift.shape == (6, 6)
    assert system.controls[0].shape == (6, 6)
    assert system.controls[1].shape == (6, 6)
    assert np.allclose(
        system.nominal_hamiltonian([alpha_1, alpha_2]),
        expected_hamiltonian,
    )


def test_spin_boson_control_system_accepts_fluctuations():
    n_levels = 3
    phi_s = 0.2
    static_fluctuation = 0.01 * np.eye(2 * n_levels, dtype=complex)
    control_fluctuations = [
        0.02 * np.kron(np.eye(2), number_operator(n_levels)),
        0.03
        * np.kron(
            spin_phase_operator(phi_s),
            annihilation_operator(n_levels) + creation_operator(n_levels),
        ),
    ]
    controls = np.array([0.4, -0.2])
    system = spin_boson_control_system(
        n_levels=n_levels,
        phi_s=phi_s,
        static_fluctuations=[static_fluctuation],
        control_fluctuations=control_fluctuations,
    )

    expected = (
        static_fluctuation
        + controls[0] * control_fluctuations[0]
        + controls[1] * control_fluctuations[1]
    )

    assert np.allclose(system.fluctuation_hamiltonian(controls), expected)
    assert np.allclose(system.fluctuation_control_derivative(0), control_fluctuations[0])
    assert np.allclose(system.fluctuation_control_derivative(1), control_fluctuations[1])


def test_spin_boson_problem_value_and_gradient_are_finite():
    n_levels = 2
    system = spin_boson_control_system(n_levels=n_levels, phi_s=0.0)
    pulse = PiecewiseConstantPulse(
        np.array(
            [
                [0.2, 0.1],
                [0.25, 0.15],
                [0.2, 0.05],
            ],
            dtype=float,
        ),
        dt=0.05,
    )
    context = EvolutionContext(
        initial_state=np.array([1.0, 0.0, 0.0, 0.0], dtype=complex),
        target_state=np.array([0.0, 0.0, 0.0, 1.0], dtype=complex),
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

    assert isinstance(value, float)
    assert np.isfinite(value)
    assert gradient.shape == (3, 2)
    assert np.all(np.isfinite(gradient))


def test_spin_boson_fluctuation_expansion_value_and_gradient_are_finite():
    n_levels = 2
    system = spin_boson_control_system(
        n_levels=n_levels,
        phi_s=0.0,
        static_fluctuations=[0.01 * np.eye(2 * n_levels, dtype=complex)],
        control_fluctuations=[
            0.02 * np.kron(np.eye(2), number_operator(n_levels)),
            0.03
            * np.kron(
                spin_phase_operator(0.0),
                annihilation_operator(n_levels) + creation_operator(n_levels),
            ),
        ],
    )
    pulse = PiecewiseConstantPulse(
        np.array(
            [
                [0.2, 0.1],
                [0.25, 0.15],
                [0.2, 0.05],
            ],
            dtype=float,
        ),
        dt=0.05,
    )
    context = EvolutionContext(
        initial_state=np.array([1.0, 0.0, 0.0, 0.0], dtype=complex),
        target_state=np.array([0.0, 0.0, 0.0, 1.0], dtype=complex),
    )
    step_builder = PerturbativeStepBuilder()
    objective = ExpansionFidelity(max_order=2)
    problem = ControlProblem(
        system=system,
        pulse=pulse,
        context=context,
        evolution=PerturbativeExpansionEvolution(step_builder, max_order=2),
        objective=objective,
        differentiator=PerturbativeExpansionDifferentiator(step_builder, objective),
    )

    value = problem.value()
    gradient = problem.gradient()

    assert isinstance(value, float)
    assert np.isfinite(value)
    assert gradient.shape == (3, 2)
    assert np.all(np.isfinite(gradient))


def test_expansion_evolution_returns_ordered_components():
    problem = two_level_problem(np.full((4, 1), 0.2))
    result = problem.evolution.evolve(problem.system, problem.pulse, problem.context)

    assert len(result.steps) == 4
    assert len(result.forward) == 5
    assert len(result.backward) == 5
    assert set(result.forward[-1].components) == {0, 1, 2}
    assert result.forward[-1].components[0].shape == (2,)


def test_ion_trap_fluctuation_hamiltonian_matches_noise_formula():
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    system = IonTrapRFSystem(
        drift=np.zeros((2, 2), dtype=complex),
        controls=[sx, sy],
        static_fluctuations=[0.01 * sz, 0.02 * sx],
        control_fluctuations=[0.03 * sx, 0.04 * sy],
    )
    controls = np.array([0.5, -0.25])

    expected = 0.01 * sz + 0.02 * sx + controls[0] * 0.03 * sx + controls[1] * 0.04 * sy

    assert np.allclose(system.fluctuation_hamiltonian(controls), expected)
    assert np.allclose(system.fluctuation_control_derivative(0), 0.03 * sx)
    assert np.allclose(system.fluctuation_control_derivative(1), 0.04 * sy)


def test_problem_value_is_scalar():
    problem = two_level_problem(np.full((4, 1), 0.2))
    value = problem.value()

    assert isinstance(value, float)
    assert np.isfinite(value)


def test_perturbative_gradient_matches_finite_difference_with_full_dv_derivative():
    amplitudes = np.array([[0.2], [0.25], [0.15]])
    problem = two_level_problem(amplitudes)
    analytic = problem.gradient()
    finite_difference = FiniteDifferenceDifferentiator(
        problem.evolution,
        problem.objective,
        epsilon=1e-6,
    ).gradient(problem.system, problem.pulse, problem.context)

    assert analytic.shape == amplitudes.shape
    assert np.allclose(analytic, finite_difference, rtol=2e-2, atol=2e-5)


def test_state_average_fidelity_matches_single_state_problem_for_one_pair():
    amplitudes = np.array([[0.2], [0.25], [0.15]])
    problem = two_level_problem(amplitudes)
    averaged = ExpansionStateAverageFidelity(
        system=problem.system,
        pulse=problem.pulse,
        evolution=problem.evolution,
        objective=problem.objective,
        differentiator=problem.differentiator,
        state_pairs=[
            StatePair(
                initial_state=problem.context.initial_state,
                target_state=problem.context.target_state,
            ),
        ],
    )

    assert np.allclose(averaged.value(), problem.value())
    assert np.allclose(averaged.gradient(), problem.gradient())


def test_state_average_fidelity_uses_weighted_value_and_gradient_average():
    amplitudes = np.array([[0.2], [0.25], [0.15]])
    initial = np.array([1.0, 0.0], dtype=complex)
    target_one = np.array([0.0, 1.0], dtype=complex)
    target_zero = np.array([1.0, 0.0], dtype=complex)
    problem_one = two_level_problem(amplitudes, initial_state=initial, target_state=target_one)
    problem_zero = two_level_problem(amplitudes, initial_state=initial, target_state=target_zero)
    averaged = ExpansionStateAverageFidelity(
        system=problem_one.system,
        pulse=problem_one.pulse,
        evolution=problem_one.evolution,
        objective=problem_one.objective,
        differentiator=problem_one.differentiator,
        state_pairs=[
            (initial, target_one, 2.0),
            (initial, target_zero, 1.0),
        ],
    )

    expected_value = (2.0 * problem_one.value() + problem_zero.value()) / 3.0
    expected_gradient = (2.0 * problem_one.gradient() + problem_zero.gradient()) / 3.0

    assert np.allclose(averaged.value(), expected_value)
    assert np.allclose(averaged.gradient(), expected_gradient)


def test_expansion_fidelity_drops_odd_average_and_keeps_second_order_terms():
    objective = ExpansionFidelity(max_order=2, drop_odd_average=True)
    amplitudes = {
        0: 2.0 + 0.0j,
        1: 3.0 + 0.0j,
        2: 5.0 + 0.0j,
    }

    value = objective.contract(amplitudes)
    no_average_drop = ExpansionFidelity(max_order=2, drop_odd_average=False).contract(amplitudes)

    assert np.allclose(value, 33.0)
    assert np.allclose(no_average_drop, 45.0)


def test_zero_fluctuation_expansion_reduces_to_nominal_state_fidelity():
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    system = IonTrapRFSystem(drift=0.1 * sz, controls=[sx])
    pulse = PiecewiseConstantPulse(np.array([[0.2], [0.25], [0.15]]), dt=0.05)
    context = EvolutionContext(
        initial_state=np.array([1.0, 0.0], dtype=complex),
        target_state=np.array([0.0, 1.0], dtype=complex),
    )

    nominal_result = NominalUnitaryEvolution(UnitaryStepBuilder()).evolve(system, pulse, context)
    expansion_result = PerturbativeExpansionEvolution(
        PerturbativeStepBuilder(),
        max_order=2,
    ).evolve(system, pulse, context)

    assert np.allclose(expansion_result.forward[-1].components[1], 0.0)
    assert np.allclose(expansion_result.forward[-1].components[2], 0.0)
    assert np.allclose(
        ExpansionFidelity(max_order=2).evaluate(expansion_result),
        StateTransferFidelity(context.target_state).evaluate(nominal_result),
    )


def test_endpoint_masked_parameterization_fixes_boundary_and_normalizes_free_points():
    parameterization = endpoint_masked_parameterization(
        n_steps=5,
        n_controls=2,
        lower=np.array([-2.0, -4.0]),
        upper=np.array([2.0, 4.0]),
    )
    parameters = np.array([-0.5, 0.5, 0.0, 0.25, 0.75, -0.25])

    amplitudes = parameterization.to_physical(parameters)

    assert amplitudes.shape == (5, 2)
    assert np.allclose(amplitudes[0], 0.0)
    assert np.allclose(amplitudes[-1], 0.0)
    assert np.allclose(parameterization.to_parameters(amplitudes), parameters)


def test_parameterized_problem_pulls_gradient_back_to_free_normalized_parameters():
    physical_amplitudes = np.array([[0.0], [0.2], [0.25], [0.0]])
    problem = two_level_problem(physical_amplitudes)
    parameterization = endpoint_masked_parameterization(
        n_steps=4,
        n_controls=1,
        lower=-1.0,
        upper=1.0,
    )
    parameterized_problem = ParameterizedControlProblem(problem, parameterization)
    parameters = parameterized_problem.initial_parameters()

    physical_gradient = problem.gradient()
    parameter_gradient = parameterized_problem.gradient(parameters)

    assert parameters.shape == (2,)
    assert parameter_gradient.shape == (2,)
    assert np.allclose(parameter_gradient, physical_gradient[1:-1, 0])


def test_pulse_constraints_penalize_slew_rate_without_projection():
    constraints = PulseConstraints(
        amplitude_lower=-1.0,
        amplitude_upper=1.0,
        max_delta=0.2,
    )
    amplitudes = np.array([[0.0], [0.5], [1.4], [-0.2]])

    assert not constraints.is_feasible(amplitudes)
    assert constraints.penalty(amplitudes) > 0.0
    assert constraints.penalty_gradient(amplitudes).shape == amplitudes.shape
    assert not hasattr(constraints, "project")


def test_parameterized_problem_applies_slew_penalty_without_changing_pulse():
    physical_amplitudes = np.array([[0.0], [0.8], [0.0], [0.0]])
    problem = two_level_problem(physical_amplitudes)
    constraints = PulseConstraints(max_delta=0.2)
    parameterization = endpoint_masked_parameterization(
        n_steps=4,
        n_controls=1,
        lower=-1.0,
        upper=1.0,
    )
    parameterized_problem = ParameterizedControlProblem(
        problem,
        parameterization,
        constraints=constraints,
        penalty_weight=10.0,
    )
    parameters = parameterized_problem.initial_parameters()
    pulse = parameterized_problem.pulse_from_parameters(parameters)

    assert np.allclose(pulse.amplitudes, physical_amplitudes)
    assert not constraints.is_feasible(pulse.amplitudes)
    assert parameterized_problem.value(parameters) < problem.value()
