"""Independent-noise regressions, including oracles outside the GRAPE recursion."""

from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pytest

from experiments.driver.run_experiment import (
    build_objective_problem, calculate_kappa_metrics, default_experiment_config,
    perturbative_fidelity_terms,
)
from experiments.spin_boson.robustness_eval.run_error_budget import noise_sources
from quantum_control import (
    ControlProblem, EvolutionContext, ExpansionFidelity, FluctuationTerm,
    OpenSystem, PerturbativeExpansionDifferentiator, PerturbativeExpansionEvolution,
    PerturbativeStepBuilder, PiecewiseConstantPulse, StatePair,
    faithful_gate_fidelity, noisy_gate_fidelity,
)
from quantum_control.diagnostics.error_budget import ErrorBudgetConfig, evaluate_error_budget
from quantum_control.evaluation import density_matrix
from quantum_control.pulses.parameterization import BoundedAmplitudeParameterization


X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.diag([1, -1]).astype(complex)
ZERO = np.zeros((2, 2), dtype=complex)
GROUND = np.array([1, 0], dtype=complex)
EXCITED = np.array([0, 1], dtype=complex)


def term(name, operator, sigma, kind="static"):
    return FluctuationTerm(name, operator, name, sigma, kind)


def problem(system, pulse, builder=None, backward=True):
    builder = builder or PerturbativeStepBuilder(dW_method="frechet")
    objective = ExpansionFidelity()
    return ControlProblem(
        system, pulse, EvolutionContext(GROUND, EXCITED, compute_backward=backward),
        PerturbativeExpansionEvolution(builder), objective,
        PerturbativeExpansionDifferentiator(builder, objective),
    )


def mixed_system():
    return OpenSystem(0.3 * Z, [X, Y], (
        term("static-z", Z, 0.07), term("static-x", X, 0.09),
        term("control-x", X, 0.04, "control"),
        term("control-y", Y, 0.06, "control"),
    ))


@pytest.mark.parametrize("second_sign", [1, -1])
def test_two_sources_obey_variance_sum_instead_of_cancelling_or_squaring_sum(second_sign):
    sigma, n_steps = 0.1, 100
    system = OpenSystem(ZERO, [ZERO], (
        term("a", X, sigma), term("b", second_sign * X, sigma),
    ))
    pulse = PiecewiseConstantPulse(np.zeros((n_steps, 1)), 1 / n_steps)
    pairs = [StatePair(GROUND, GROUND)]
    result = noisy_gate_fidelity(system, pulse, pairs)
    # Two independent Gaussians give Var(xi_a +/- xi_b) = 2 sigma^2.
    exact = (1 + np.exp(-4 * sigma**2)) / 2
    assert faithful_gate_fidelity(system, pulse, pairs, hermite_points=7) == pytest.approx(exact, abs=1e-12)
    # The known missing same-slice second-order term is deliberately unchanged.
    assert result == pytest.approx(1 - 2 * sigma**2 * (1 - 1 / n_steps), abs=1e-12)
    assert abs(result - exact) < 3e-4


@pytest.mark.parametrize("v_method", ["leading", "frechet"])
def test_mixed_static_and_control_gradient_matches_finite_difference(v_method):
    pulse = PiecewiseConstantPulse(np.array([[0.3, -0.2], [0.7, 0.4], [-0.1, 0.5]]), 0.08)
    builder = PerturbativeStepBuilder(dW_method="frechet", V_method=v_method,
                                     v_derivative_epsilon=1e-5)
    p = problem(mixed_system(), pulse, builder)
    value, gradient = p.value_and_gradient()
    epsilon = 1e-5
    finite_difference = np.empty_like(gradient)
    for coordinate in np.ndindex(gradient.shape):
        plus, minus = pulse.amplitudes.copy(), pulse.amplitudes.copy()
        plus[coordinate] += epsilon
        minus[coordinate] -= epsilon
        finite_difference[coordinate] = (
            p.value(pulse.with_amplitudes(plus)) - p.value(pulse.with_amplitudes(minus))
        ) / (2 * epsilon)
    np.testing.assert_allclose(gradient, finite_difference, rtol=2e-7, atol=1e-10)
    assert value == p.value()
    np.testing.assert_array_equal(gradient, p.gradient())


def test_independent_corrections_and_gradients_are_additive_and_sign_invariant():
    system = mixed_system()
    pulse = PiecewiseConstantPulse(np.array([[0.3, -0.2], [0.7, 0.4], [-0.1, 0.5]]), 0.08)
    f0, g0 = problem(replace(system, noise_terms=()), pulse).value_and_gradient()
    combined, combined_gradient = problem(system, pulse).value_and_gradient()
    expected, expected_gradient = f0, g0.copy()
    for index in range(len(system.noise_terms)):
        selected = replace(system, noise_terms=tuple(
            replace(t, coefficient=t.coefficient if i == index else 0)
            for i, t in enumerate(system.noise_terms)
        ))
        value, gradient = problem(selected, pulse).value_and_gradient()
        expected += value - f0
        expected_gradient += gradient - g0
    assert combined == pytest.approx(expected, abs=1e-14)
    np.testing.assert_allclose(combined_gradient, expected_gradient, rtol=1e-12, atol=1e-14)
    flipped = replace(system, noise_terms=(replace(system.noise_terms[0], operator=-Z),) + system.noise_terms[1:])
    flipped_value, flipped_gradient = problem(flipped, pulse).value_and_gradient()
    assert flipped_value == pytest.approx(combined, abs=1e-14)
    np.testing.assert_allclose(flipped_gradient, combined_gradient, rtol=1e-12, atol=1e-14)
    reordered = replace(system, noise_terms=system.noise_terms[1:2] + system.noise_terms[:1] + system.noise_terms[2:])
    assert problem(reordered, pulse).value() == pytest.approx(combined, abs=1e-14)


def test_optimizer_evaluator_logging_and_diagnostics_use_same_average():
    system = mixed_system()
    pulse = PiecewiseConstantPulse(np.array([[0.3, -0.2], [0.7, 0.4]]), 0.08)
    pairs = [StatePair(GROUND, EXCITED, 0.7), StatePair(EXCITED, GROUND, 0.3)]
    config = default_experiment_config()
    config = replace(config, runtime=replace(config.runtime, workers=1))
    with build_objective_problem(config, system, pulse, pairs) as p:
        optimized_objective = p.value()
    evaluated = noisy_gate_fidelity(system, pulse, pairs)
    summary, rows = perturbative_fidelity_terms(system, pulse, pairs)
    budget = evaluate_error_budget(system, pulse, pairs, ErrorBudgetConfig(
        gradient_samples=2, fluctuation_scales=(1.0,),
    ))
    diagnostic = next(row["value"] for row in budget.rows if row["metric"] == "perturbative_fidelity")
    for value in (optimized_objective, summary["perturbative_open"], diagnostic):
        assert value == pytest.approx(evaluated, abs=1e-14)
    assert sum(row["closed_term"] + row["first_order_sq"] + row["second_order_cross"] for row in rows) == pytest.approx(evaluated, abs=1e-14)
    assert problem(system, pulse, backward=False).value() == pytest.approx(problem(system, pulse).value(), abs=1e-14)


def test_per_source_budget_preserves_second_control_and_skips_zero_quadrature_dimensions():
    system = OpenSystem(ZERO, [ZERO, ZERO], (
        term("first", X, 0.2, "control"), term("second", X, 0.1, "control"),
    ))
    pulse = PiecewiseConstantPulse(np.tile([0.0, 1.0], (8, 1)), 0.125)
    selected = noise_sources(system)[1]["system"]
    pairs = [StatePair(GROUND, GROUND)]
    assert noisy_gate_fidelity(selected, pulse, pairs) < 0.995
    with patch.object(density_matrix, "expm", wraps=density_matrix.expm) as expm:
        faithful = faithful_gate_fidelity(selected, pulse, pairs, hermite_points=5)
    assert faithful == pytest.approx((1 + np.exp(-0.02)) / 2, abs=1e-11)
    assert expm.call_count == pulse.n_steps * 5
    assert faithful_gate_fidelity(replace(selected, noise_terms=tuple(
        replace(t, coefficient=0) for t in selected.noise_terms
    )), pulse, pairs) == pytest.approx(1, abs=1e-14)


def test_unsupported_multi_channel_orders_and_unaveraged_mode_fail_explicitly():
    p = problem(mixed_system(), PiecewiseConstantPulse(np.ones((2, 2)), 0.1))
    with pytest.raises(ValueError, match="max_order <= 2"):
        PerturbativeExpansionEvolution(p.evolution.step_builder, max_order=4).evolve(p.system, p.pulse, p.context)
    result = p.evolution.evolve(p.system, p.pulse, p.context)
    with pytest.raises(ValueError, match="drop_odd_average=True"):
        ExpansionFidelity(drop_odd_average=False).evaluate(result)


def test_kappa_estimate_does_not_cancel_independent_opposite_operators():
    system = OpenSystem(ZERO, [ZERO], (term("a", X, 0.1), term("b", -X, 0.1)))
    pulse = PiecewiseConstantPulse(np.zeros((10, 1)), 0.1)
    metrics = calculate_kappa_metrics(system, system, pulse, BoundedAmplitudeParameterization(-1, 1))
    assert metrics["kappa_2"] == pytest.approx(np.sqrt(2) * 0.1)
