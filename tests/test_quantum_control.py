from datetime import datetime
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np

from experiments.reporting import (
    FidelityTermsLog,
    StepLog,
    export_pulse_controls,
    format_fidelity_terms_table_header,
    format_fidelity_terms_table_row,
    format_step_table_header,
    format_step_table_row,
    timestamped_experiment_dir,
    timestamped_report_path,
    write_fidelity_pair_terms_csv,
    write_fidelity_terms_csv,
    write_step_log_csv,
    write_experiment_report,
)
from physical_systems.spin_boson import (
    ms_xx_pi_over_2_gate,
    single_qubit_logical_test_states,
    two_qubit_logical_test_states,
    DEFAULT_ALPHA1_KHZ_BOUNDS,
    DEFAULT_ALPHA2_KHZ_BOUNDS,
    DEFAULT_LAMB_DICKE_ETA,
    annihilation_operator,
    creation_operator,
    motion_resolved_gate_state_pairs,
    number_operator,
    spin_boson_collapse_operators,
    spin_boson_control_system,
    spin_boson_initial_pulse,
    spin_boson_parameterization,
    spin_phase_operator,
    two_qubit_spin_phase_difference,
    two_qubit_spin_phase_mode,
)
from quantum_control import (
    ClosedSystem,
    SumProblem,
    ControlProblem,
    faithful_gate_fidelity,
    DecoherenceChannel,
    EvolutionContext,
    ExpansionFidelity,
    StateAverageProblem,
    GrapeDifferentiator,
    FluctuationTerm,
    LindbladCorrectedStateFidelity,
    LindbladExpansionDifferentiator,
    LindbladExpansionEvolution,
    OpenSystem,
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
    closed_gate_fidelity,
    endpoint_masked_parameterization,
    noisy_gate_fidelity,
)
from quantum_control.differentiators.finite_difference import FiniteDifferenceDifferentiator
from quantum_control.optimizers import ScipyOptimizer
from quantum_control.diagnostics.error_budget import (
    ErrorBudgetConfig,
    evaluate_error_budget,
    load_pulse_npz,
    write_error_budget_report,
)


ROOT = Path(__file__).resolve().parents[1]


def test_timestamped_report_path_uses_safe_slug_and_markdown_extension(tmp_path):
    generated_at = datetime(2026, 6, 21, 15, 30, 12)

    report_path = timestamped_report_path(tmp_path, "spin boson/perturbative", generated_at)

    assert report_path == (
        tmp_path / "spin_boson_perturbative_20260621_153012" / "report.md"
    )


def test_timestamped_experiment_dir_uses_safe_slug_and_timestamp(tmp_path):
    generated_at = datetime(2026, 6, 21, 15, 30, 12)

    experiment_dir = timestamped_experiment_dir(
        tmp_path,
        "spin boson/perturbative",
        generated_at,
    )

    assert experiment_dir == tmp_path / "spin_boson_perturbative_20260621_153012"


def test_write_experiment_report_includes_sections_and_relative_figures(tmp_path):
    output_dir = tmp_path / "outputs"
    run_dir = output_dir / "spin_boson_20260621_153012"
    run_dir.mkdir(parents=True)
    pulse_plot = run_dir / "pulse.png"
    propagation_plot = run_dir / "propagation.png"
    pulse_plot.write_bytes(b"pulse")
    propagation_plot.write_bytes(b"propagation")

    report_path = write_experiment_report(
        output_dir=output_dir,
        experiment_slug="spin_boson",
        title="Spin-Boson Test",
        configuration=[
            ("n_levels", 6),
            ("optimizer_options", {"maxiter": 1, "gtol": 1e-12}),
        ],
        results=[
            ("fidelity", 0.1, 0.25),
        ],
        optimizer=[
            ("success", True),
            ("message", "ok"),
        ],
        figures=[
            ("Pulse parameters", pulse_plot),
            ("State propagation", propagation_plot),
        ],
        generated_at=datetime(2026, 6, 21, 15, 30, 12),
    )

    markdown = report_path.read_text(encoding="utf-8")
    assert report_path == run_dir / "report.md"
    assert "# Spin-Boson Test" in markdown
    assert "## Configuration" in markdown
    assert "| n_levels | 6 |" in markdown
    assert "## Results" in markdown
    assert "| fidelity | 0.1 | 0.25 | 0.15 |" in markdown
    assert "## Optimizer" in markdown
    assert "| success | True |" in markdown
    assert "## Figures" in markdown
    assert "![Pulse parameters](pulse.png)" in markdown
    assert "![State propagation](propagation.png)" in markdown


def test_write_step_log_csv_includes_expected_headers_and_rows(tmp_path):
    log_path = tmp_path / "step_log.csv"

    write_step_log_csv(
        log_path,
        [
            {
                "step": 0,
                "close_fidelity": 0.1,
                "open_fidelity": 0.2,
                "raw_fidelity": 0.35,
                "l1_penalty": 0.01,
                "l2_penalty": 0.04,
                "cost_function": 0.3,
                "gradient_norm": 0.7,
            },
            {
                "step": 1,
                "close_fidelity": 0.4,
                "open_fidelity": 0.5,
                "raw_fidelity": 0.7,
                "l1_penalty": 0.03,
                "l2_penalty": 0.07,
                "cost_function": 0.6,
                "gradient_norm": 0.8,
            },
        ],
    )

    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == (
        "step,close_fidelity,open_fidelity,raw_fidelity,l1_penalty,l2_penalty,"
        "cost_function,gradient_norm"
    )
    assert lines[1] == (
        "0,0.10000000000000001,0.20000000000000001,0.34999999999999998,"
        "0.01,0.040000000000000001,0.29999999999999999,0.69999999999999996"
    )
    assert lines[2] == (
        "1,0.40000000000000002,0.5,0.69999999999999996,"
        "0.029999999999999999,0.070000000000000007,0.59999999999999998,"
        "0.80000000000000004"
    )


def test_step_log_prints_table_header_and_rows(tmp_path, capsys):
    log = StepLog(tmp_path / "step_log.csv", print_steps=True)

    log.append(
        step=0,
        close_fidelity=0.1,
        open_fidelity=0.2,
        raw_fidelity=0.35,
        l1_penalty=0.01,
        l2_penalty=0.04,
        cost_function=0.3,
        gradient_norm=0.7,
    )

    output = capsys.readouterr().out.splitlines()
    assert output[0] == format_step_table_header()
    assert output[0].startswith("step  close fidelity")
    assert output[1] == format_step_table_row(
        {
            "step": 0,
            "close_fidelity": 0.1,
            "open_fidelity": 0.2,
            "raw_fidelity": 0.35,
            "l1_penalty": 0.01,
            "l2_penalty": 0.04,
            "cost_function": 0.3,
            "gradient_norm": 0.7,
        }
    )


def test_fidelity_terms_log_writes_csvs_and_prints_aligned_table(tmp_path, capsys):
    summary = {
        "step": 0,
        "closed_term": 0.9,
        "first_order_sq": 0.03,
        "second_order_cross": 0.08,
        "perturbative_open": 1.01,
        "correction": 0.11,
        "excess_over_1": 0.01,
        "max_pair_open": 0.2,
        "min_pair_open": -0.01,
    }
    pair = {
        "step": 0,
        "pair_index": 1,
        "weight": 0.5,
        "a0_real": 1.0,
        "a0_imag": 0.0,
        "a1_real": 0.1,
        "a1_imag": 0.2,
        "a2_real": 0.03,
        "a2_imag": -0.04,
        "closed_term": 0.5,
        "first_order_sq": 0.025,
        "second_order_cross": 0.03,
        "perturbative_open": 0.555,
        "dropped_order1_cross": 0.1,
    }
    log = FidelityTermsLog(
        tmp_path / "fidelity_terms.csv",
        tmp_path / "fidelity_terms_by_pair.csv",
        print_steps=True,
    )

    log.append(summary, [pair])

    output = capsys.readouterr().out.splitlines()
    assert output[0] == format_fidelity_terms_table_header()
    assert output[1] == format_fidelity_terms_table_row(summary)
    summary_lines = (tmp_path / "fidelity_terms.csv").read_text(encoding="utf-8").splitlines()
    pair_lines = (tmp_path / "fidelity_terms_by_pair.csv").read_text(encoding="utf-8").splitlines()
    assert summary_lines[0].startswith("step,closed_term,first_order_sq,second_order_cross")
    assert pair_lines[0].startswith("step,pair_index,weight,a0_real,a0_imag")


def test_fidelity_terms_csv_helpers_write_expected_headers(tmp_path):
    write_fidelity_terms_csv(
        tmp_path / "summary.csv",
        [
            {
                "step": 2,
                "closed_term": 0.8,
                "first_order_sq": 0.1,
                "second_order_cross": 0.2,
                "perturbative_open": 1.1,
                "correction": 0.3,
                "excess_over_1": 0.1,
                "max_pair_open": 0.4,
                "min_pair_open": 0.0,
            }
        ],
    )
    write_fidelity_pair_terms_csv(
        tmp_path / "pairs.csv",
        [
            {
                "step": 2,
                "pair_index": 3,
                "weight": 0.5,
                "a0_real": 1.0,
                "a0_imag": 0.0,
                "a1_real": 0.0,
                "a1_imag": 0.0,
                "a2_real": 0.0,
                "a2_imag": 0.0,
                "closed_term": 0.5,
                "first_order_sq": 0.0,
                "second_order_cross": 0.0,
                "perturbative_open": 0.5,
                "dropped_order1_cross": 0.0,
            }
        ],
    )

    assert "excess_over_1" in (tmp_path / "summary.csv").read_text(encoding="utf-8")
    assert "dropped_order1_cross" in (tmp_path / "pairs.csv").read_text(encoding="utf-8")


def test_export_pulse_controls_writes_npz_keys_and_csv_columns(tmp_path):
    pulse = PiecewiseConstantPulse(
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float),
        dt=0.25,
    )

    npz_path, csv_path = export_pulse_controls(
        pulse,
        tmp_path / "initial_pulse",
        rad_s_per_khz=2.0,
    )

    data = np.load(npz_path)
    assert set(data.files) == {
        "amplitudes",
        "dt",
        "time_s",
        "time_us",
        "rad_s_per_khz",
        "channel_names",
    }
    assert np.allclose(data["amplitudes"], pulse.amplitudes)
    assert float(data["dt"]) == 0.25
    assert np.allclose(data["time_s"], [0.125, 0.375])
    assert np.allclose(data["time_us"], [125000.0, 375000.0])
    assert float(data["rad_s_per_khz"]) == 2.0
    assert data["channel_names"].tolist() == ["alpha1", "alpha2"]

    lines = csv_path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == (
        "step_index,time_s,time_us,alpha1_rad_s,alpha2_rad_s,"
        "alpha1_khz,alpha2_khz"
    )
    assert lines[1] == "0,0.125,125000,1,2,0.5,1"
    assert lines[2] == "1,0.375,375000,3,4,1.5,2"


def test_load_pulse_npz_uses_dt_or_fallback(tmp_path):
    amplitudes = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=float)
    with_dt = tmp_path / "with_dt.npz"
    without_dt = tmp_path / "without_dt.npz"
    np.savez(with_dt, amplitudes=amplitudes, dt=0.05)
    np.savez(without_dt, amplitudes=amplitudes)

    pulse = load_pulse_npz(with_dt)
    fallback_pulse = load_pulse_npz(without_dt, fallback_dt=0.07)

    assert np.allclose(pulse.amplitudes, amplitudes)
    assert pulse.dt == 0.05
    assert fallback_pulse.dt == 0.07
    with np.testing.assert_raises_regex(ValueError, "fallback_dt"):
        load_pulse_npz(without_dt)


def open_system_from_matrices(
    drift, controls, static_fluctuations=(), control_fluctuations=(), collapse_operators=()
):
    """Wrap pre-scaled matrices into unit-strength noise terms on an OpenSystem."""
    noise_terms = [
        FluctuationTerm(name=f"static[{i}]", operator=m, definition="", coefficient=1.0, kind="static")
        for i, m in enumerate(static_fluctuations)
    ]
    noise_terms += [
        FluctuationTerm(name=f"control[{i}]", operator=m, definition="", coefficient=1.0, kind="control")
        for i, m in enumerate(control_fluctuations)
    ]
    noise_terms += [
        DecoherenceChannel(name=f"collapse[{i}]", operator=op, definition="", rate=1.0)
        for i, op in enumerate(collapse_operators)
    ]
    return OpenSystem(drift=drift, controls=controls, noise_terms=tuple(noise_terms))


def two_level_problem(amplitudes, initial_state=None, target_state=None):
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    system = open_system_from_matrices(
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
        two_qubit_spin_phase_mode(0.0, (0.5, 0.5)),
        0.5 * (np.kron(sx, np.eye(2)) + np.kron(np.eye(2), sx)),
    )
    assert np.allclose(
        two_qubit_spin_phase_difference(0.0),
        0.5 * (np.kron(sx, np.eye(2)) - np.kron(np.eye(2), sx)),
    )


def test_spin_boson_control_system_builds_two_control_hamiltonians():
    n_levels = 3
    phi_s = 0.2
    alpha_1 = 0.3
    alpha_2 = -0.4
    system = spin_boson_control_system(n_levels=n_levels, phi_s=phi_s)
    a = annihilation_operator(n_levels)
    adag = a.conj().T
    x1 = 0.5 * (a + adag)
    stretch_hamiltonian = (
        alpha_1 * np.kron(np.eye(4), adag @ a)
        + alpha_2
        * DEFAULT_LAMB_DICKE_ETA
        * np.kron(two_qubit_spin_phase_difference(phi_s), x1)
    )
    com_spin = two_qubit_spin_phase_mode(phi_s, (0.5, 0.5))
    com_hamiltonian = (
        alpha_1 * np.kron(np.eye(4), adag @ a)
        + alpha_2 * DEFAULT_LAMB_DICKE_ETA * np.kron(com_spin, x1)
    )
    com_system = spin_boson_control_system(
        n_levels=n_levels,
        phi_s=phi_s,
        mode_vector=(0.5, 0.5),
    )

    assert system.drift.shape == (12, 12)
    assert system.controls[0].shape == (12, 12)
    assert system.controls[1].shape == (12, 12)
    assert np.allclose(
        system.nominal_hamiltonian([alpha_1, alpha_2]),
        stretch_hamiltonian,
    )
    assert np.allclose(
        com_system.nominal_hamiltonian([alpha_1, alpha_2]),
        com_hamiltonian,
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

    assert len(single_states) == 4
    assert len(two_qubit_states) == 16
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

    assert len(pairs) == 16 * n_levels
    assert np.allclose(sum(pair.weight for pair in pairs), n_levels)
    assert first_pair.initial_state.shape == (4 * n_levels,)
    assert first_pair.target_state.shape == (4 * n_levels,)
    assert np.allclose(first_pair.initial_state[0], 1.0)
    assert np.allclose(first_pair.initial_state[1:], 0.0)
    assert np.allclose(first_pair.weight, 1.0 / 16.0)


def test_zero_fluctuation_noisy_gate_fidelity_matches_closed_gate_fidelity():
    n_levels = 2
    system = spin_boson_control_system(n_levels=n_levels, phi_s=0.0)
    pulse = PiecewiseConstantPulse(
        np.array([[0.02, 0.01], [0.025, 0.015]], dtype=float),
        dt=0.005,
    )
    target_gate = ms_xx_pi_over_2_gate()

    state_pairs = motion_resolved_gate_state_pairs(target_gate, n_levels)
    closed = closed_gate_fidelity(system, pulse, state_pairs)
    noisy = noisy_gate_fidelity(system, pulse, state_pairs)

    assert np.allclose(noisy, closed)


def test_noisy_gate_fidelity_includes_decoherence_correction():
    n_levels = 2
    system = spin_boson_control_system(n_levels=n_levels, phi_s=0.0)
    pulse = PiecewiseConstantPulse(
        np.array([[0.02, 0.01], [0.025, 0.015]], dtype=float),
        dt=0.005,
    )
    state_pairs = motion_resolved_gate_state_pairs(ms_xx_pi_over_2_gate(), n_levels)
    collapse_operators = spin_boson_collapse_operators(
        n_levels=n_levels,
        gamma_heating=0.5,
        gamma_motional_dephasing=0.2,
        gamma_spin_dephasing=0.1,
    )

    without = noisy_gate_fidelity(system, pulse, state_pairs)
    with_decoherence = noisy_gate_fidelity(
        system, pulse, state_pairs, collapse_operators=collapse_operators
    )

    assert with_decoherence < without


def test_open_grape_optimization_improves_noisy_gate_fidelity():
    """Open-system GRAPE end-to-end: optimize the combined objective
    (fluctuation expansion + Lindblad correction) on an ``OpenSystem`` built
    from declarative noise terms, and verify against ``noisy_gate_fidelity``.
    """
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    sigma_minus = np.array([[0, 1], [0, 0]], dtype=complex)
    system = OpenSystem(
        drift=np.zeros((2, 2), dtype=complex),
        controls=[sx],
        noise_terms=[
            FluctuationTerm(
                name="dephasing", operator=sz, definition="sigma_z",
                coefficient=0.01, kind="static",
            ),
            DecoherenceChannel(
                name="decay", operator=sigma_minus, definition="sigma_minus",
                rate=0.02,
            ),
        ],
    )
    pulse = PiecewiseConstantPulse(np.full((8, 1), 2.0), dt=0.05)
    zero = np.array([1.0, 0.0], dtype=complex)
    one = np.array([0.0, 1.0], dtype=complex)
    state_pairs = (StatePair(zero, one, 1.0),)

    step_builder = PerturbativeStepBuilder()
    expansion_objective = ExpansionFidelity(max_order=2, drop_odd_average=True)
    expansion_problem = StateAverageProblem(
        system=system,
        pulse=pulse,
        evolution=PerturbativeExpansionEvolution(step_builder, max_order=2),
        objective=expansion_objective,
        differentiator=PerturbativeExpansionDifferentiator(step_builder, expansion_objective),
        state_pairs=state_pairs,
        normalize_weights=False,
    )
    unitary_builder = UnitaryStepBuilder()
    decoherence_problem = StateAverageProblem(
        system=system,
        pulse=pulse,
        evolution=LindbladExpansionEvolution(
            unitary_builder, collapse_operators=system.collapse_operators
        ),
        objective=LindbladCorrectedStateFidelity(include_closed=False),
        differentiator=LindbladExpansionDifferentiator(unitary_builder, include_closed=False),
        state_pairs=state_pairs,
        normalize_weights=False,
    )
    with SumProblem(expansion_problem, decoherence_problem) as problem:
        initial = noisy_gate_fidelity(
            system, pulse, state_pairs, collapse_operators=system.collapse_operators
        )
        # The optimizer objective is exactly the reported noisy gate fidelity.
        assert np.allclose(problem.value(pulse), initial)

        optimizer = ScipyOptimizer(method="L-BFGS-B", maximize=True, options={"maxiter": 40})
        result = optimizer.optimize(problem)
        final_pulse = result.optimized_pulse
        final = noisy_gate_fidelity(
            system, final_pulse, state_pairs, collapse_operators=system.collapse_operators
        )

    assert final > initial
    assert final > 0.95


def test_faithful_gate_fidelity_matches_closed_fidelity_without_noise():
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    system = ClosedSystem(drift=0.1 * sz, controls=[sx])
    pulse = PiecewiseConstantPulse(np.array([[0.2], [0.25], [0.15]]), dt=0.05)
    zero = np.array([1.0, 0.0], dtype=complex)
    one = np.array([0.0, 1.0], dtype=complex)
    state_pairs = (StatePair(zero, one, 0.6), StatePair(one, zero, 0.4))

    faithful = faithful_gate_fidelity(system, pulse, state_pairs)
    closed = closed_gate_fidelity(system, pulse, state_pairs)

    assert np.allclose(faithful, closed, atol=1e-10)


def test_faithful_gate_fidelity_matches_exact_lindblad_oracle():
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    sigma_minus = np.array([[0, 1], [0, 0]], dtype=complex)
    gamma = 0.05
    system = open_system_from_matrices(
        drift=0.1 * sz,
        controls=[sx],
        collapse_operators=(np.sqrt(gamma) * sigma_minus,),
    )
    pulse = PiecewiseConstantPulse(np.array([[0.2], [0.25], [0.15]]), dt=0.05)
    zero = np.array([1.0, 0.0], dtype=complex)
    one = np.array([0.0, 1.0], dtype=complex)
    state_pairs = (StatePair(zero, one, 1.0),)

    faithful = faithful_gate_fidelity(system, pulse, state_pairs)
    rho_final = exact_lindblad_final_state(system, pulse, zero)
    oracle = float(np.real(one.conj() @ rho_final @ one))
    assert np.allclose(faithful, oracle, atol=1e-12)

    # small kappa_3: the perturbative metric agrees to first order
    perturbative = noisy_gate_fidelity(
        system, pulse, state_pairs, collapse_operators=system.collapse_operators
    )
    kappa_3 = pulse.n_steps * pulse.dt * gamma
    assert abs(faithful - perturbative) < kappa_3**2


def test_faithful_gate_fidelity_matches_perturbative_for_small_fluctuations():
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    sigma = 0.05
    system = OpenSystem(
        drift=np.zeros((2, 2), dtype=complex),
        controls=[sx],
        noise_terms=[
            FluctuationTerm(
                name="dephasing", operator=sz, definition="sigma_z",
                coefficient=sigma, kind="static",
            ),
        ],
    )
    zero = np.array([1.0, 0.0], dtype=complex)
    one = np.array([0.0, 1.0], dtype=complex)
    state_pairs = (StatePair(zero, one, 1.0),)
    # same physical pulse at two step resolutions (total time 0.3 both ways)
    coarse_pulse = PiecewiseConstantPulse(np.full((6, 1), 2.5), dt=0.05)
    fine_pulse = PiecewiseConstantPulse(np.full((96, 1), 2.5), dt=0.003125)

    faithful = faithful_gate_fidelity(system, fine_pulse, state_pairs, hermite_points=9)
    perturbative = noisy_gate_fidelity(system, fine_pulse, state_pairs)
    closed = closed_gate_fidelity(system, fine_pulse, state_pairs)

    # the fluctuation correction is real and the faithful average captures it
    correction = abs(closed - faithful)
    assert correction > 1e-5
    # at fine dt the second-order expansion reproduces >90% of the correction
    assert abs(faithful - perturbative) < 0.1 * correction
    # the faithful value is dt-exact; the perturbative metric converges to it
    coarse_gap = abs(
        faithful_gate_fidelity(system, coarse_pulse, state_pairs, hermite_points=9)
        - noisy_gate_fidelity(system, coarse_pulse, state_pairs)
    )
    assert abs(faithful - perturbative) < coarse_gap / 4
    # quadrature is converged: more nodes do not move the value
    refined = faithful_gate_fidelity(system, fine_pulse, state_pairs, hermite_points=15)
    assert np.allclose(faithful, refined, atol=1e-9)


def test_value_and_gradient_matches_separate_calls_bitwise(tmp_path):
    """The fused evaluation must be numerically identical to separate calls
    through the full driver stack (SumProblem -> parameterized -> penalized),
    so switching the optimizer to jac=True cannot change any trajectory."""
    path = _yaml_config_file(
        tmp_path,
        "\n".join(
            [
                "system:",
                "  params:",
                "    n_levels: 2",
                "  noise:",
                "    decoherence:",
                "      enabled: true",
                "      gamma_heating: 0.5",
                "pulse:",
                "  n_steps: 4",
                "  random_seed: 7",
                "",
            ]
        ),
    )
    config = sbo.parse_args(["--config", str(path)])
    _system, open_system = sbo.build_systems(config)
    initial_pulse = sbo.build_initial_pulse(config)
    parameterization = sbo.build_parameterization(config, initial_pulse)
    state_pairs = sbo.build_state_pairs(config)
    with sbo.build_objective_problem(config, open_system, initial_pulse, state_pairs) as problem:
        assert isinstance(problem, SumProblem)
        penalized = PenalizedParameterizedProblem(
            ParameterizedControlProblem(problem, parameterization),
            ParameterSmoothPenalty(l1_weight=5e-4, l2_weight=1e-4),
        )
        parameters = penalized.initial_parameters().reshape(-1)

        fused_value, fused_gradient = penalized.value_and_gradient(parameters)

        assert fused_value == penalized.value(parameters)
        assert np.array_equal(fused_gradient, penalized.gradient(parameters))


def test_spin_boson_initial_pulse_uses_standard_units_and_full_alpha1_cycle():
    pulse = spin_boson_initial_pulse()
    alpha1_lower = DEFAULT_ALPHA1_KHZ_BOUNDS[0] * 2.0 * np.pi * 1000.0
    alpha1_upper = DEFAULT_ALPHA1_KHZ_BOUNDS[1] * 2.0 * np.pi * 1000.0
    alpha2_upper = DEFAULT_ALPHA2_KHZ_BOUNDS[1] * 2.0 * np.pi * 1000.0
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
        np.array(
            [
                DEFAULT_ALPHA1_KHZ_BOUNDS[0] * 2.0 * np.pi * 1000.0,
                DEFAULT_ALPHA2_KHZ_BOUNDS[0] * 2.0 * np.pi * 1000.0,
            ]
        ),
    )
    assert np.allclose(
        parameterization.upper,
        np.array(
            [
                DEFAULT_ALPHA1_KHZ_BOUNDS[1] * 2.0 * np.pi * 1000.0,
                DEFAULT_ALPHA2_KHZ_BOUNDS[1] * 2.0 * np.pi * 1000.0,
            ]
        ),
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


def test_open_system_fluctuation_hamiltonian_matches_noise_formula():
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    system = open_system_from_matrices(
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


def test_perturbative_gradient_frechet_dv_derivative_matches_finite_difference():
    amplitudes = np.array([[0.2], [0.25], [0.15]])
    problem = two_level_problem(amplitudes)
    step_builder = PerturbativeStepBuilder(dW_method="frechet")
    problem = ControlProblem(
        system=problem.system,
        pulse=problem.pulse,
        context=problem.context,
        evolution=PerturbativeExpansionEvolution(step_builder, max_order=2),
        objective=problem.objective,
        differentiator=PerturbativeExpansionDifferentiator(
            step_builder,
            problem.objective,
        ),
    )
    analytic = problem.gradient()
    finite_difference = FiniteDifferenceDifferentiator(
        problem.evolution,
        problem.objective,
        epsilon=1e-6,
    ).gradient(problem.system, problem.pulse, problem.context)

    assert analytic.shape == amplitudes.shape
    assert np.allclose(analytic, finite_difference, rtol=1e-6, atol=1e-9)


def test_perturbative_v_method_leading_matches_default_and_frechet_commuting_case():
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    system = open_system_from_matrices(
        drift=0.2 * sz,
        controls=[0.3 * sz],
        static_fluctuations=[0.01 * sz],
        control_fluctuations=[0.02 * sz],
    )
    controls = np.array([0.4])
    default_step = PerturbativeStepBuilder().build_step(system, controls, 0.05)
    leading_step = PerturbativeStepBuilder(V_method="leading").build_step(system, controls, 0.05)
    frechet_step = PerturbativeStepBuilder(V_method="frechet").build_step(system, controls, 0.05)

    assert np.allclose(leading_step.W, default_step.W)
    assert np.allclose(leading_step.V, default_step.V)
    assert np.allclose(frechet_step.V, leading_step.V)


def test_perturbative_v_method_frechet_is_finite_for_noncommuting_case():
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    system = open_system_from_matrices(
        drift=0.2 * sz,
        controls=[sx],
        static_fluctuations=[0.03 * sx],
        control_fluctuations=[0.04 * sz],
    )
    controls = np.array([0.5])
    leading_step = PerturbativeStepBuilder(V_method="leading").build_step(system, controls, 0.2)
    frechet_step = PerturbativeStepBuilder(V_method="frechet").build_step(system, controls, 0.2)

    assert np.all(np.isfinite(frechet_step.V))
    assert not np.allclose(frechet_step.V, leading_step.V, rtol=1e-8, atol=1e-12)


def test_state_average_fidelity_matches_single_state_problem_for_one_pair():
    amplitudes = np.array([[0.2], [0.25], [0.15]])
    problem = two_level_problem(amplitudes)
    averaged = StateAverageProblem(
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


def test_error_budget_evaluates_toy_problem_and_writes_report(tmp_path):
    problem = two_level_problem(np.array([[0.2], [0.25]], dtype=float))
    state_pairs = [
        StatePair(
            initial_state=problem.context.initial_state,
            target_state=problem.context.target_state,
        )
    ]

    report = evaluate_error_budget(
        problem.system,
        problem.pulse,
        state_pairs,
        ErrorBudgetConfig(
            gradient_samples=2,
            fluctuation_scales=(0.5,),
            random_seed=7,
        ),
    )
    outputs = write_error_budget_report(report, tmp_path)

    assert outputs["markdown"].exists()
    assert outputs["csv"].exists()
    markdown = outputs["markdown"].read_text(encoding="utf-8")
    assert "## Summary" in markdown
    assert "| W error |" in markdown
    assert "| dW error |" in markdown
    assert "| V fidelity error |" in markdown
    assert "| truncation fidelity error |" in markdown
    assert "| sigmaT squared estimate |" in markdown
    assert "| optimization perturbative fidelity |" in markdown
    assert any(row["metric"] == "unitarity_fro_max" for row in report.rows)
    assert any(row["metric"] == "norm_first_minus_fd" for row in report.rows)
    assert any(row["metric"] == "fidelity_leading_minus_frechet" for row in report.rows)
    assert any(row["metric"] == "sigmaT_squared_estimate" for row in report.rows)
    finite_values = [
        row["value"]
        for row in report.rows
        if row["available"] and isinstance(row["value"], (float, int, np.floating))
    ]
    assert finite_values
    assert np.all(np.isfinite(finite_values))


def test_error_budget_cli_smoke_test(tmp_path):
    pulse_path = tmp_path / "pulse.npz"
    output_dir = tmp_path / "report"
    factory_path = tmp_path / "toy_factory.py"
    np.savez(pulse_path, amplitudes=np.array([[0.2], [0.25]], dtype=float), dt=0.05)
    factory_path.write_text(
        "\n".join(
            [
                "import numpy as np",
                "from quantum_control import FluctuationTerm, OpenSystem, StatePair",
                "",
                "def build():",
                "    sx = np.array([[0, 1], [1, 0]], dtype=complex)",
                "    sz = np.array([[1, 0], [0, -1]], dtype=complex)",
                "    system = OpenSystem(",
                "        drift=0.1 * sz,",
                "        controls=[sx],",
                "        noise_terms=[",
                "            FluctuationTerm(name='s', operator=0.01 * sz, definition='', coefficient=1.0, kind='static'),",
                "            FluctuationTerm(name='c', operator=0.02 * sx, definition='', coefficient=1.0, kind='control'),",
                "        ],",
                "    )",
                "    pair = StatePair(",
                "        np.array([1.0, 0.0], dtype=complex),",
                "        np.array([0.0, 1.0], dtype=complex),",
                "    )",
                "    return system, [pair], {'label': 'toy'}",
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "experiments/evaluate_error_budget.py",
            "--pulse-npz",
            str(pulse_path),
            "--system-factory",
            "toy_factory:build",
            "--output-dir",
            str(output_dir),
            "--gradient-samples",
            "1",
            "--scales",
            "0.5",
        ],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(tmp_path)},
        capture_output=True,
        text=True,
        check=True,
    )

    assert "error_budget_md=" in result.stdout
    assert (output_dir / "error_budget.md").exists()
    assert (output_dir / "error_budget.csv").exists()


def test_state_average_fidelity_uses_weighted_value_and_gradient_average():
    amplitudes = np.array([[0.2], [0.25], [0.15]])
    initial = np.array([1.0, 0.0], dtype=complex)
    target_one = np.array([0.0, 1.0], dtype=complex)
    target_zero = np.array([1.0, 0.0], dtype=complex)
    problem_one = two_level_problem(amplitudes, initial_state=initial, target_state=target_one)
    problem_zero = two_level_problem(amplitudes, initial_state=initial, target_state=target_zero)
    averaged = StateAverageProblem(
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


def test_state_average_fidelity_parallel_matches_serial_value_and_gradient():
    amplitudes = np.array([[0.2], [0.25], [0.15]])
    initial = np.array([1.0, 0.0], dtype=complex)
    target_one = np.array([0.0, 1.0], dtype=complex)
    target_zero = np.array([1.0, 0.0], dtype=complex)
    problem = two_level_problem(amplitudes, initial_state=initial, target_state=target_one)
    state_pairs = [
        (initial, target_one, 2.0),
        (initial, target_zero, 1.0),
    ]
    serial = StateAverageProblem(
        system=problem.system,
        pulse=problem.pulse,
        evolution=problem.evolution,
        objective=problem.objective,
        differentiator=problem.differentiator,
        state_pairs=state_pairs,
    )
    parallel = StateAverageProblem(
        system=problem.system,
        pulse=problem.pulse,
        evolution=problem.evolution,
        objective=problem.objective,
        differentiator=problem.differentiator,
        state_pairs=state_pairs,
        n_workers=2,
    )

    try:
        assert np.allclose(parallel.value(), serial.value())
        assert np.allclose(parallel.gradient(), serial.gradient())
    finally:
        parallel.shutdown()


def test_state_average_fidelity_requires_positive_worker_count():
    problem = two_level_problem(np.array([[0.2], [0.25], [0.15]]))

    try:
        StateAverageProblem(
            system=problem.system,
            pulse=problem.pulse,
            evolution=problem.evolution,
            objective=problem.objective,
            differentiator=problem.differentiator,
            state_pairs=[
                (problem.context.initial_state, problem.context.target_state),
            ],
            n_workers=0,
        )
    except ValueError as exc:
        assert "n_workers" in str(exc)
    else:
        raise AssertionError("Expected n_workers=0 to raise ValueError.")


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
    system = ClosedSystem(drift=0.1 * sz, controls=[sx])
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


def two_level_lindblad_setup(amplitudes, gamma=0.02, dt=0.05, dW_method="first_order"):
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    sigma_minus = np.array([[0, 1], [0, 0]], dtype=complex)
    collapse_operators = (
        (np.sqrt(gamma) * sigma_minus, np.sqrt(0.5 * gamma) * sz) if gamma > 0.0 else ()
    )
    system = open_system_from_matrices(
        drift=0.1 * sz,
        controls=[sx],
        collapse_operators=collapse_operators,
    )
    pulse = PiecewiseConstantPulse(np.asarray(amplitudes, dtype=float), dt=dt)
    context = EvolutionContext(
        initial_state=np.array([1.0, 0.0], dtype=complex),
        target_state=np.array([1.0, 1.0], dtype=complex) / np.sqrt(2.0),
        compute_backward=True,
    )
    step_builder = UnitaryStepBuilder(dW_method=dW_method)
    evolution = LindbladExpansionEvolution(step_builder)
    objective = LindbladCorrectedStateFidelity()
    differentiator = LindbladExpansionDifferentiator(step_builder)
    return system, pulse, context, evolution, objective, differentiator


def exact_lindblad_final_state(system, pulse, initial_state):
    from scipy.linalg import expm

    dimension = initial_state.shape[0]
    identity = np.eye(dimension, dtype=complex)
    dissipator = np.zeros((dimension**2, dimension**2), dtype=complex)
    for operator in system.collapse_operators:
        product = operator.conj().T @ operator
        dissipator = dissipator + (
            np.kron(operator, operator.conj())
            - 0.5 * np.kron(product, identity)
            - 0.5 * np.kron(identity, product.T)
        )
    rho = np.outer(initial_state, initial_state.conj()).reshape(-1)
    for step_index in range(pulse.n_steps):
        hamiltonian = system.nominal_hamiltonian(pulse.controls_at(step_index))
        liouvillian = (
            -1j * (np.kron(hamiltonian, identity) - np.kron(identity, hamiltonian.T))
            + dissipator
        )
        rho = expm(pulse.dt * liouvillian) @ rho
    return rho.reshape(dimension, dimension)


def test_lindblad_zero_rate_matches_closed_fidelity_and_grape_gradient():
    amplitudes = np.array([[0.2], [0.25], [0.15]])
    system, pulse, context, evolution, objective, differentiator = two_level_lindblad_setup(
        amplitudes, gamma=0.0
    )
    result = evolution.evolve(system, pulse, context)

    step_builder = UnitaryStepBuilder()
    nominal_result = NominalUnitaryEvolution(step_builder).evolve(system, pulse, context)
    closed_value = StateTransferFidelity(context.target_state).evaluate(nominal_result)
    grape_gradient = GrapeDifferentiator(step_builder).gradient(
        system, pulse, context, nominal_result
    )

    assert np.isclose(objective.evaluate(result), closed_value, rtol=1e-12)
    assert np.allclose(
        differentiator.gradient(system, pulse, context, result),
        grape_gradient,
        rtol=1e-12,
        atol=1e-14,
    )


def test_lindblad_correction_amplitude_is_real():
    amplitudes = np.array([[0.2], [0.25], [0.15], [0.1]])
    system, pulse, context, evolution, _, _ = two_level_lindblad_setup(amplitudes)
    result = evolution.evolve(system, pulse, context)

    target_state = result.backward[-1].components[0]
    correction_amplitude = np.vdot(target_state, result.forward[-1].components[1])

    assert abs(np.imag(correction_amplitude)) < 1e-12 * max(
        1.0, abs(correction_amplitude)
    )


def test_lindblad_correction_only_flag_splits_closed_term():
    amplitudes = np.array([[0.2], [0.25], [0.15]])
    system, pulse, context, evolution, objective, differentiator = two_level_lindblad_setup(
        amplitudes
    )
    result = evolution.evolve(system, pulse, context)

    correction_objective = LindbladCorrectedStateFidelity(include_closed=False)
    correction_differentiator = LindbladExpansionDifferentiator(include_closed=False)
    closed_amplitude = np.vdot(
        result.backward[-1].components[0], result.forward[-1].components[0]
    )

    assert np.isclose(
        objective.evaluate(result),
        correction_objective.evaluate(result) + abs(closed_amplitude) ** 2,
        rtol=1e-12,
    )
    step_builder = UnitaryStepBuilder()
    nominal_result = NominalUnitaryEvolution(step_builder).evolve(system, pulse, context)
    grape_gradient = GrapeDifferentiator(step_builder).gradient(
        system, pulse, context, nominal_result
    )
    assert np.allclose(
        differentiator.gradient(system, pulse, context, result),
        correction_differentiator.gradient(system, pulse, context, result)
        + grape_gradient,
        rtol=1e-10,
        atol=1e-13,
    )


def test_lindblad_correction_matches_exact_master_equation():
    rng = np.random.default_rng(7)
    amplitudes = 0.2 + 0.1 * rng.random((50, 1))
    gamma = 0.01
    system, pulse, context, evolution, objective, _ = two_level_lindblad_setup(
        amplitudes, gamma=gamma, dt=0.04
    )
    result = evolution.evolve(system, pulse, context)
    corrected_value = objective.evaluate(result)

    rho_final = exact_lindblad_final_state(system, pulse, context.initial_state)
    exact_value = float(
        np.real(np.vdot(context.target_state, rho_final @ context.target_state))
    )
    closed_amplitude = np.vdot(
        result.backward[-1].components[0], result.forward[-1].components[0]
    )
    closed_value = float(abs(closed_amplitude) ** 2)

    gap = abs(exact_value - closed_value)
    assert gap > 1e-4  # the decoherence effect is resolvable
    assert abs(corrected_value - exact_value) < 0.2 * gap


def test_lindblad_gradient_matches_finite_difference():
    amplitudes = np.array([[0.2], [0.25], [0.15]])
    system, pulse, context, evolution, objective, differentiator = two_level_lindblad_setup(
        amplitudes, dt=0.005
    )
    result = evolution.evolve(system, pulse, context)

    analytic = differentiator.gradient(system, pulse, context, result)
    finite_difference = FiniteDifferenceDifferentiator(
        evolution,
        objective,
        epsilon=1e-6,
    ).gradient(system, pulse, context)

    assert analytic.shape == amplitudes.shape
    assert np.allclose(analytic, finite_difference, rtol=2e-2, atol=2e-5)


def test_lindblad_gradient_frechet_matches_finite_difference():
    amplitudes = np.array([[0.2], [0.25], [0.15]])
    system, pulse, context, evolution, objective, differentiator = two_level_lindblad_setup(
        amplitudes, dW_method="frechet"
    )
    result = evolution.evolve(system, pulse, context)

    analytic = differentiator.gradient(system, pulse, context, result)
    finite_difference = FiniteDifferenceDifferentiator(
        evolution,
        objective,
        epsilon=1e-6,
    ).gradient(system, pulse, context)

    assert analytic.shape == amplitudes.shape
    assert np.allclose(analytic, finite_difference, rtol=1e-6, atol=1e-9)


def test_lindblad_state_average_fidelity_runs_with_lindblad_triple():
    n_levels = 2
    dimension = 4 * n_levels
    collapse_operators = spin_boson_collapse_operators(
        n_levels,
        gamma_heating=0.5,
        gamma_motional_dephasing=0.2,
    )
    system = spin_boson_control_system(
        n_levels=n_levels,
        phi_s=0.0,
        collapse_operators=collapse_operators,
    )
    pulse = PiecewiseConstantPulse(np.full((3, 2), 0.02), dt=0.005)
    step_builder = UnitaryStepBuilder()
    state_pairs = [
        StatePair(
            np.eye(dimension, dtype=complex)[0],
            np.eye(dimension, dtype=complex)[3 * n_levels],
        ),
        StatePair(
            np.eye(dimension, dtype=complex)[n_levels],
            np.eye(dimension, dtype=complex)[2 * n_levels],
        ),
    ]
    problem = StateAverageProblem(
        system=system,
        pulse=pulse,
        evolution=LindbladExpansionEvolution(step_builder),
        objective=LindbladCorrectedStateFidelity(),
        differentiator=LindbladExpansionDifferentiator(step_builder),
        state_pairs=state_pairs,
    )

    assert isinstance(system, OpenSystem)
    assert len(collapse_operators) == 2
    value = problem.value()
    gradient = problem.gradient()
    assert np.isfinite(value)
    assert gradient.shape == pulse.amplitudes.shape
    assert np.all(np.isfinite(gradient))


# --- YAML experiment configuration (experiments/run_experiment.py) ---


from dataclasses import replace as _dc_replace

import pytest as _pytest

from experiments.config_io import (
    config_to_yaml_str,
    load_experiment_config,
    write_config_snapshot,
)
from physical_systems import get_system
from experiments import run_experiment as sbo


def _yaml_config_file(tmp_path, text):
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_yaml_config_round_trip(tmp_path):
    config = sbo.default_experiment_config()
    config = _dc_replace(
        config,
        system=_dc_replace(
            config.system,
            params=_dc_replace(
                config.system.params,
                phi_s=0.25,
                mode_vector=(0.5, 0.5),
                alpha1_noise_fraction=0.0,
            ),
            noise=_dc_replace(
                config.system.noise,
                decoherence=_dc_replace(
                    config.system.noise.decoherence,
                    enabled=True,
                    gamma_heating=12.5,
                ),
            ),
        ),
        pulse=_dc_replace(config.pulse, n_steps=17, random_seed=42),
        runtime=_dc_replace(config.runtime, initial_pulse_npz=tmp_path / "p.npz"),
    )
    path = write_config_snapshot(config, tmp_path / "config.yaml")
    reloaded = load_experiment_config(path, sbo.default_experiment_config(), get_system)
    assert reloaded == config


def test_yaml_config_cli_precedence(tmp_path):
    path = _yaml_config_file(
        tmp_path,
        "optimizer:\n  maxiter: 7\nruntime:\n  workers: 3\n",
    )
    config = sbo.parse_args(["--config", str(path), "--maxiter", "3"])
    assert config.optimizer.maxiter == 3
    assert config.runtime.workers == 3
    assert config.pulse.n_steps == sbo.default_experiment_config().pulse.n_steps


def test_yaml_config_physics_knobs_reach_system(tmp_path):
    path = _yaml_config_file(
        tmp_path,
        "\n".join(
            [
                "system:",
                "  params:",
                "    n_levels: 3",
                "    mode_vector: [0.5, 0.5]",
                "    alpha1_noise_fraction: 0.0",
                "  noise:",
                "    fluctuations:",
                "      sigma_static_motional_frequency: 42.0",
                "pulse:",
                "  n_steps: 9",
                "",
            ]
        ),
    )
    config = sbo.parse_args(["--config", str(path)])
    _system, open_system = sbo.build_systems(config)
    by_name = {term.name: term for term in open_system.fluctuation_terms}
    assert by_name["motion-shift"].coefficient == 42.0
    assert "mode=(0.5, 0.5)" in by_name["alpha2-rel"].definition
    pulse_a = sbo.build_initial_pulse(config)
    pulse_b = sbo.build_initial_pulse(config)
    np.testing.assert_allclose(pulse_a.amplitudes, pulse_b.amplitudes)


def test_yaml_config_cosine_initial_pulse_shape(tmp_path):
    path = _yaml_config_file(
        tmp_path,
        "\n".join(
            [
                "system:",
                "  params:",
                "    n_levels: 2",
                "    initial_pulse_shape: cosine",
                "    alpha1_cycles: 2.0",
                "pulse:",
                "  n_steps: 16",
                "",
            ]
        ),
    )
    config = sbo.parse_args(["--config", str(path)])
    pulse = sbo.build_initial_pulse(config)
    expected = spin_boson_initial_pulse(
        n_steps=16,
        total_time_us=config.pulse.total_time_us,
        alpha1_cycles=2.0,
    )
    np.testing.assert_allclose(pulse.amplitudes, expected.amplitudes)

    bad = _yaml_config_file(tmp_path, "system:\n  params:\n    initial_pulse_shape: triangle\n")
    bad_config = sbo.parse_args(["--config", str(bad)])
    with _pytest.raises(ValueError, match="cosine"):
        sbo.build_initial_pulse(bad_config)


def test_output_prefix_defaults_to_system_name(tmp_path):
    config = sbo.default_experiment_config()
    assert sbo.output_prefix(config) == "spin_boson"

    path = _yaml_config_file(tmp_path, "output:\n  prefix: warmup\n")
    config = sbo.parse_args(["--config", str(path)])
    assert sbo.output_prefix(config) == "warmup"
    assert "prefix: warmup" in config_to_yaml_str(config)

    reloaded = load_experiment_config(
        _yaml_config_file(tmp_path, config_to_yaml_str(config)),
        sbo.default_experiment_config(),
        get_system,
    )
    assert reloaded.output.prefix == "warmup"


def test_yaml_config_enabled_flags(tmp_path):
    path = _yaml_config_file(
        tmp_path,
        "\n".join(
            [
                "system:",
                "  noise:",
                "    decoherence:",
                "      enabled: false",
                "      gamma_heating: 50.0",
                "    fluctuations:",
                "      enabled: false",
                "",
            ]
        ),
    )
    config = sbo.parse_args(["--config", str(path)])
    _system, open_system = sbo.build_systems(config)
    assert open_system.noise_terms == ()
    assert open_system.collapse_operators == ()
    assert len(open_system.static_fluctuations) == 0

    cli_config = sbo.parse_args(["--gamma-heating", "50.0"])
    assert cli_config.system.noise.decoherence.enabled
    _cli_system, cli_open_system = sbo.build_systems(cli_config)
    assert len(cli_open_system.collapse_operators) == 1


def test_yaml_config_unknown_keys_raise(tmp_path):
    path = _yaml_config_file(tmp_path, "system:\n  params:\n    gamma_heat: 1.0\n")
    with _pytest.raises(ValueError, match="gamma_heat"):
        sbo.parse_args(["--config", str(path)])

    path = _yaml_config_file(tmp_path, "system:\n  type: nv_center\n")
    with _pytest.raises(ValueError, match="spin_boson"):
        sbo.parse_args(["--config", str(path)])

    path = _yaml_config_file(tmp_path, "system:\n  params:\n    target_gate: bogus_gate\n")
    config = sbo.parse_args(["--config", str(path)])
    with _pytest.raises(ValueError, match="ms_xx_pi_over_2"):
        sbo.build_target_gate(config)


def test_close_grape_flag_and_legacy_namespace_coerce():
    config = sbo.parse_args(["--close-grape"])
    assert not config.system.noise.fluctuations.enabled

    legacy = SimpleNamespace(include_fluctuations=False, maxiter=2, n_steps=7)
    coerced = sbo._coerce_experiment_config(legacy)
    assert not coerced.system.noise.fluctuations.enabled
    assert coerced.optimizer.maxiter == 2
    assert coerced.pulse.n_steps == 7


def test_experiment_writes_reloadable_config_snapshot(tmp_path):
    config = sbo.default_experiment_config()
    config = _dc_replace(
        config,
        system=_dc_replace(
            config.system,
            params=_dc_replace(config.system.params, n_levels=3),
        ),
        pulse=_dc_replace(config.pulse, n_steps=5, random_seed=7),
        optimizer=_dc_replace(config.optimizer, maxiter=1),
        runtime=_dc_replace(config.runtime, no_progress=True),
    )
    outcome = sbo.run_perturbative_experiment(
        config,
        output_root=tmp_path,
        print_report=False,
    )
    snapshot = outcome["outputs"]["config_snapshot"]
    assert snapshot.exists()
    reloaded = load_experiment_config(
        snapshot, sbo.default_experiment_config(), get_system
    )
    assert reloaded == config
