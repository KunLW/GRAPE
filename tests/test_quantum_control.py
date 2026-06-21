import numpy as np

from quantum_control import (
    ControlProblem,
    EvolutionContext,
    ExpansionFidelity,
    ExpansionStateAverageFidelity,
    GrapeDifferentiator,
    IonTrapRFSystem,
    NominalUnitaryEvolution,
    ParameterSmoothPenalty,
    ParameterizedControlProblem,
    PenalizedParameterizedProblem,
    PerturbativeExpansionDifferentiator,
    PerturbativeExpansionEvolution,
    PerturbativeStepBuilder,
    PiecewiseConstantPulse,
    PulseConstraints,
    StateTransferFidelity,
    StatePair,
    UnitaryStepBuilder,
    annihilation_operator,
    closed_gate_fidelity,
    creation_operator,
    endpoint_masked_parameterization,
    motion_resolved_gate_state_pairs,
    ms_xx_pi_over_2_gate,
    number_operator,
    open_gate_fidelity,
    single_qubit_logical_test_states,
    spin_boson_control_system,
    spin_boson_initial_pulse,
    spin_boson_parameterization,
    spin_phase_operator,
    two_qubit_spin_phase_difference,
    two_qubit_logical_test_states,
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
    assert np.allclose(
        two_qubit_spin_phase_difference(0.0),
        np.kron(sx, np.eye(2)) - np.kron(np.eye(2), sx),
    )


def test_spin_boson_control_system_builds_two_control_hamiltonians():
    n_levels = 3
    phi_s = 0.2
    alpha_1 = 0.3
    alpha_2 = -0.4
    system = spin_boson_control_system(n_levels=n_levels, phi_s=phi_s)
    a = annihilation_operator(n_levels)
    adag = a.conj().T
    expected_hamiltonian = (
        alpha_1 * np.kron(np.eye(4), adag @ a)
        + alpha_2 * np.kron(two_qubit_spin_phase_difference(phi_s), a + adag)
    )

    assert system.drift.shape == (12, 12)
    assert system.controls[0].shape == (12, 12)
    assert system.controls[1].shape == (12, 12)
    assert np.allclose(
        system.nominal_hamiltonian([alpha_1, alpha_2]),
        expected_hamiltonian,
    )


def test_spin_boson_control_system_accepts_fluctuations():
    n_levels = 3
    phi_s = 0.2
    static_fluctuation = 0.01 * np.eye(4 * n_levels, dtype=complex)
    control_fluctuations = [
        0.02 * np.kron(np.eye(4), number_operator(n_levels)),
        0.03
        * np.kron(
            two_qubit_spin_phase_difference(phi_s),
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


def test_logical_gate_metric_states_are_normalized():
    single_states = single_qubit_logical_test_states()
    two_qubit_states = two_qubit_logical_test_states()

    assert len(single_states) == 6
    assert len(two_qubit_states) == 36
    assert all(np.allclose(np.vdot(state, state), 1.0) for state in single_states)
    assert all(np.allclose(np.vdot(state, state), 1.0) for state in two_qubit_states)


def test_ms_xx_pi_over_2_gate_matches_bell_target_convention():
    gate = ms_xx_pi_over_2_gate()
    zero_zero = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)
    expected = np.array([1.0, 0.0, 0.0, -1j], dtype=complex) / np.sqrt(2.0)

    assert np.allclose(gate.conj().T @ gate, np.eye(4))
    assert np.allclose(gate @ zero_zero, expected)


def test_motion_resolved_gate_state_pairs_use_spin_motion_ordering_and_weights():
    n_levels = 3
    pairs = motion_resolved_gate_state_pairs(ms_xx_pi_over_2_gate(), n_levels)
    first_pair = pairs[0]

    assert len(pairs) == 36 * n_levels
    assert np.allclose(sum(pair.weight for pair in pairs), n_levels)
    assert first_pair.initial_state.shape == (4 * n_levels,)
    assert first_pair.target_state.shape == (4 * n_levels,)
    assert np.allclose(first_pair.initial_state[0], 1.0)
    assert np.allclose(first_pair.initial_state[1:], 0.0)
    assert np.allclose(first_pair.weight, 1.0 / 36.0)


def test_zero_fluctuation_open_gate_fidelity_matches_closed_gate_fidelity():
    n_levels = 2
    system = spin_boson_control_system(n_levels=n_levels, phi_s=0.0)
    pulse = PiecewiseConstantPulse(
        np.array([[0.02, 0.01], [0.025, 0.015]], dtype=float),
        dt=0.005,
    )
    target_gate = ms_xx_pi_over_2_gate()

    closed = closed_gate_fidelity(system, pulse, target_gate, n_levels)
    opened = open_gate_fidelity(system, pulse, target_gate, n_levels)

    assert np.allclose(opened, closed)


def test_spin_boson_initial_pulse_uses_standard_units_and_full_alpha1_cycle():
    pulse = spin_boson_initial_pulse()
    alpha1_lower = 2.0 * np.pi * 1000.0
    alpha1_upper = 2.0 * np.pi * 600000.0
    alpha2_upper = 2.0 * np.pi * 20000.0
    alpha1 = pulse.amplitudes[:, 0]
    alpha2 = pulse.amplitudes[:, 1]

    assert pulse.amplitudes.shape == (200, 2)
    assert np.allclose(pulse.dt, 225.8e-6 / 200)
    assert np.all(alpha1 >= alpha1_lower)
    assert np.all(alpha1 <= alpha1_upper)
    assert alpha1[0] > alpha1[pulse.n_steps // 2]
    assert alpha1[-1] > alpha1[pulse.n_steps // 2]
    assert np.allclose(alpha1[0], alpha1[-1])
    assert np.all(alpha2 >= 0.0)
    assert np.all(alpha2 <= alpha2_upper)
    assert alpha2[pulse.n_steps // 2 - 1] > alpha2[0]
    assert alpha2[pulse.n_steps // 2 - 1] > alpha2[-1]
    assert np.allclose(alpha2[0], alpha2[-1])


def test_spin_boson_initial_pulse_allows_half_alpha1_cycle():
    pulse = spin_boson_initial_pulse(alpha1_cycles=0.5)

    assert np.all(np.diff(pulse.amplitudes[:, 0]) < 0.0)


def test_spin_boson_parameterization_uses_rad_s_bounds_and_round_trips():
    pulse = spin_boson_initial_pulse(n_steps=5)
    parameterization = spin_boson_parameterization(n_steps=pulse.n_steps)
    parameters = parameterization.to_parameters(pulse.amplitudes)
    reconstructed = parameterization.to_physical(parameters)

    assert np.allclose(
        parameterization.lower,
        np.array([2.0 * np.pi * 1000.0, 0.0]),
    )
    assert np.allclose(
        parameterization.upper,
        np.array([2.0 * np.pi * 600000.0, 2.0 * np.pi * 20000.0]),
    )
    assert np.allclose(reconstructed, pulse.amplitudes)
    assert parameterization.parameter_bounds(pulse.amplitudes.shape) == [(-1.0, 1.0)] * 10


def test_parameter_smooth_penalty_handles_constant_linear_and_curved_parameters():
    penalty = ParameterSmoothPenalty(l1_weight=2.0, l2_weight=3.0)
    constant_parameters = np.ones((5, 2))
    linear_parameters = np.column_stack(
        [
            np.linspace(1.0, 5.0, 6),
            np.linspace(2.0, 4.0, 6),
        ]
    )
    curved_parameters = np.array(
        [
            [1.0, 1.0],
            [3.0, 1.0],
            [2.0, 4.0],
            [6.0, 2.0],
            [4.0, 5.0],
        ],
        dtype=float,
    )

    assert np.allclose(penalty.value(constant_parameters, constant_parameters.shape), 0.0)
    assert penalty.l1_value(linear_parameters, linear_parameters.shape) > 0.0
    assert np.allclose(penalty.l2_value(linear_parameters, linear_parameters.shape), 0.0)
    assert penalty.l1_value(curved_parameters, curved_parameters.shape) > 0.0
    assert penalty.l2_value(curved_parameters, curved_parameters.shape) > 0.0


def test_parameter_smooth_l1_gradient_matches_finite_difference():
    penalty = ParameterSmoothPenalty(l1_weight=1.7)
    parameters = np.array(
        [
            [0.0, 0.3],
            [0.2, 0.1],
            [0.5, -0.2],
            [0.9, -0.6],
        ],
        dtype=float,
    )
    gradient = penalty.gradient(parameters, parameters.shape)
    epsilon = 1e-6
    finite_difference = np.zeros_like(parameters)

    for index in np.ndindex(parameters.shape):
        plus = np.array(parameters, copy=True)
        minus = np.array(parameters, copy=True)
        plus[index] += epsilon
        minus[index] -= epsilon
        finite_difference[index] = (
            penalty.value(plus, parameters.shape) - penalty.value(minus, parameters.shape)
        ) / (2.0 * epsilon)

    assert np.allclose(gradient, finite_difference, rtol=1e-5, atol=1e-9)


def test_parameter_smooth_l2_gradient_matches_finite_difference():
    penalty = ParameterSmoothPenalty(l2_weight=2.5)
    parameters = np.array(
        [
            [1.0, 1.0],
            [3.0, 1.0],
            [2.0, 4.0],
            [6.0, 2.0],
            [4.0, 5.0],
        ],
        dtype=float,
    )
    gradient = penalty.gradient(parameters, parameters.shape)
    epsilon = 1e-6
    finite_difference = np.zeros_like(parameters)

    for index in np.ndindex(parameters.shape):
        plus = np.array(parameters, copy=True)
        minus = np.array(parameters, copy=True)
        plus[index] += epsilon
        minus[index] -= epsilon
        finite_difference[index] = (
            penalty.value(plus, parameters.shape) - penalty.value(minus, parameters.shape)
        ) / (2.0 * epsilon)

    assert np.allclose(gradient, finite_difference, rtol=1e-5, atol=1e-10)


def test_spin_boson_problem_value_and_gradient_are_finite():
    n_levels = 2
    system = spin_boson_control_system(n_levels=n_levels, phi_s=0.0)
    pulse = spin_boson_initial_pulse(n_steps=20)
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

    assert isinstance(value, float)
    assert np.isfinite(value)
    assert gradient.shape == pulse.amplitudes.shape
    assert np.all(np.isfinite(gradient))


def test_spin_boson_parameterized_problem_uses_initial_pulse_helper():
    n_levels = 2
    pulse = spin_boson_initial_pulse(n_steps=10)
    parameterization = spin_boson_parameterization(n_steps=pulse.n_steps)
    system = spin_boson_control_system(n_levels=n_levels, phi_s=0.0)
    context = EvolutionContext(
        initial_state=np.eye(4 * n_levels, dtype=complex)[0],
        target_state=np.eye(4 * n_levels, dtype=complex)[3 * n_levels + 1],
    )
    step_builder = UnitaryStepBuilder()
    evolution = NominalUnitaryEvolution(step_builder)
    objective = StateTransferFidelity(context.target_state)
    problem = ControlProblem(
        system=system,
        pulse=pulse,
        context=context,
        evolution=evolution,
        objective=objective,
        differentiator=GrapeDifferentiator(step_builder),
    )
    parameterized_problem = ParameterizedControlProblem(problem, parameterization)
    parameters = parameterized_problem.initial_parameters()

    assert parameters.shape == pulse.amplitudes.shape
    assert np.isfinite(parameterized_problem.value(parameters))
    assert parameterized_problem.gradient(parameters).shape == pulse.amplitudes.shape
    assert parameterized_problem.parameter_bounds() == [(-1.0, 1.0)] * pulse.amplitudes.size


def test_spin_boson_grape_gradient_matches_finite_difference_approximately():
    n_levels = 2
    system = spin_boson_control_system(n_levels=n_levels, phi_s=0.0)
    pulse = PiecewiseConstantPulse(
        np.array(
            [
                [0.02, 0.01],
                [0.025, 0.015],
                [0.02, 0.005],
            ],
            dtype=float,
        ),
        dt=0.005,
    )
    context = EvolutionContext(
        initial_state=np.eye(4 * n_levels, dtype=complex)[0],
        target_state=np.eye(4 * n_levels, dtype=complex)[3 * n_levels + 1],
    )
    step_builder = UnitaryStepBuilder()
    evolution = NominalUnitaryEvolution(step_builder)
    objective = StateTransferFidelity(context.target_state)
    result = evolution.evolve(system, pulse, context)

    analytic = GrapeDifferentiator(step_builder).gradient(system, pulse, context, result)
    finite_difference = FiniteDifferenceDifferentiator(
        evolution,
        objective,
        epsilon=1e-7,
    ).gradient(system, pulse, context)

    assert analytic.shape == pulse.amplitudes.shape
    assert np.allclose(analytic, finite_difference, rtol=5e-2, atol=1e-8)


def test_grape_gradient_requires_target_state():
    n_levels = 2
    system = spin_boson_control_system(n_levels=n_levels, phi_s=0.0)
    pulse = PiecewiseConstantPulse(np.full((2, 2), 0.01), dt=0.005)
    context = EvolutionContext(
        initial_state=np.eye(4 * n_levels, dtype=complex)[0],
        target_state=None,
    )
    step_builder = UnitaryStepBuilder()
    result = NominalUnitaryEvolution(step_builder).evolve(system, pulse, context)

    try:
        GrapeDifferentiator(step_builder).gradient(system, pulse, context, result)
    except ValueError as exc:
        assert "target_state" in str(exc)
    else:
        raise AssertionError("Expected missing target_state to raise ValueError.")


def test_spin_boson_fluctuation_expansion_value_and_gradient_are_finite():
    n_levels = 2
    system = spin_boson_control_system(
        n_levels=n_levels,
        phi_s=0.0,
        static_fluctuations=[0.01 * np.eye(4 * n_levels, dtype=complex)],
        control_fluctuations=[
            0.02 * np.kron(np.eye(4), number_operator(n_levels)),
            0.03
            * np.kron(
                two_qubit_spin_phase_difference(0.0),
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
        initial_state=np.eye(4 * n_levels, dtype=complex)[0],
        target_state=np.eye(4 * n_levels, dtype=complex)[3 * n_levels + 1],
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


def test_penalized_parameterized_problem_subtracts_parameter_space_penalty():
    physical_amplitudes = np.array([[0.0], [0.2], [0.25], [0.0]])
    problem = two_level_problem(physical_amplitudes)
    parameterization = endpoint_masked_parameterization(
        n_steps=4,
        n_controls=1,
        lower=-1.0,
        upper=1.0,
    )
    parameterized_problem = ParameterizedControlProblem(problem, parameterization)
    penalty = ParameterSmoothPenalty(l1_weight=0.5, l2_weight=0.25)
    penalized_problem = PenalizedParameterizedProblem(parameterized_problem, penalty)
    parameters = parameterized_problem.initial_parameters()

    assert np.allclose(
        penalized_problem.value(parameters),
        parameterized_problem.value(parameters)
        - penalty.value(parameters, penalized_problem.parameter_shape),
    )
    assert np.allclose(
        penalized_problem.gradient(parameters),
        parameterized_problem.gradient(parameters)
        - penalty.gradient(parameters, penalized_problem.parameter_shape),
    )
    assert penalized_problem.gradient(parameters).shape == parameters.shape
    assert np.allclose(
        penalized_problem.pulse_from_parameters(parameters).amplitudes,
        physical_amplitudes,
    )
    assert np.allclose(
        penalized_problem.pulse_from_parameters(parameters.reshape(-1)).amplitudes,
        physical_amplitudes,
    )


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
