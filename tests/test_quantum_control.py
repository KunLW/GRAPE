from datetime import datetime
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

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
from experiments.spin_boson_perturbative_initial_sweep import (
    build_sweep_run_specs,
    load_custom_initial_parameters,
    noise_initial_parameters,
    random_initial_parameters,
    reference_pulse_and_parameterization,
    write_summary_csv,
    write_summary_markdown,
)
from experiments.spin_boson_perturbative_lbfgsb import (
    calculate_kappa_metrics,
    load_custom_initial_parameters as load_single_custom_initial_parameters,
    run_perturbative_experiment,
    spin_boson_noise_term_specs,
    spin_boson_noisy_control_system,
)
from quantum_control import (
    ControlProblem,
    DEFAULT_ALPHA1_KHZ_BOUNDS,
    DEFAULT_ALPHA2_KHZ_BOUNDS,
    DEFAULT_LAMB_DICKE_ETA,
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
    two_qubit_spin_phase_mode,
    two_qubit_spin_phase_difference,
    two_qubit_logical_test_states,
)
from quantum_control.differentiators.finite_difference import FiniteDifferenceDifferentiator
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


def test_sweep_noise_initial_parameters_are_reproducible_and_clipped():
    base = np.zeros((4, 2), dtype=float)

    first = noise_initial_parameters(base, np.random.default_rng(123), 0.3)
    second = noise_initial_parameters(base, np.random.default_rng(123), 0.3)
    clipped = noise_initial_parameters(np.full((4, 2), 0.95), np.random.default_rng(1), 10.0)

    assert np.allclose(first, second)
    assert np.all(clipped >= -1.0)
    assert np.all(clipped <= 1.0)


def test_sweep_random_initial_parameters_respect_bounds_and_endpoint_mask():
    reference_pulse, parameterization = reference_pulse_and_parameterization(
        n_steps=5,
        alpha1_cycles=1.0,
    )

    parameters = random_initial_parameters(reference_pulse.amplitudes.shape, np.random.default_rng(5))
    pulse = PiecewiseConstantPulse(parameterization.to_physical(parameters), reference_pulse.dt)

    assert np.all(parameters >= -1.0)
    assert np.all(parameters <= 1.0)
    assert np.allclose(pulse.amplitudes[[0, -1], 1], 0.0)


def test_sweep_custom_npz_imports_pulse_and_records_matching_dt(tmp_path):
    reference_pulse, parameterization = reference_pulse_and_parameterization(
        n_steps=5,
        alpha1_cycles=1.0,
    )
    compatible_amplitudes = parameterization.to_physical(
        parameterization.to_parameters(reference_pulse.amplitudes)
    )
    npz_path = tmp_path / "pulse.npz"
    np.savez(npz_path, amplitudes=compatible_amplitudes, dt=reference_pulse.dt)

    parameters, metadata = load_custom_initial_parameters(
        npz_path,
        reference_pulse,
        parameterization,
    )

    assert np.allclose(parameterization.to_physical(parameters), compatible_amplitudes)
    assert metadata["source_dt"] == reference_pulse.dt
    assert metadata["dt_missing"] is False
    assert metadata["dt_mismatch"] is False
    assert metadata["warnings"] == []


def test_sweep_custom_npz_missing_dt_continues_with_warning(tmp_path):
    reference_pulse, parameterization = reference_pulse_and_parameterization(
        n_steps=5,
        alpha1_cycles=1.0,
    )
    compatible_amplitudes = parameterization.to_physical(
        parameterization.to_parameters(reference_pulse.amplitudes)
    )
    npz_path = tmp_path / "pulse_missing_dt.npz"
    np.savez(npz_path, amplitudes=compatible_amplitudes)

    _parameters, metadata = load_custom_initial_parameters(
        npz_path,
        reference_pulse,
        parameterization,
    )

    assert metadata["source_dt"] == "NA"
    assert metadata["dt_missing"] is True
    assert metadata["dt_mismatch"] is False
    assert "has no dt" in metadata["warnings"][0]


def test_sweep_custom_npz_mismatched_dt_continues_with_warning(tmp_path):
    reference_pulse, parameterization = reference_pulse_and_parameterization(
        n_steps=5,
        alpha1_cycles=1.0,
    )
    compatible_amplitudes = parameterization.to_physical(
        parameterization.to_parameters(reference_pulse.amplitudes)
    )
    npz_path = tmp_path / "pulse_mismatched_dt.npz"
    np.savez(npz_path, amplitudes=compatible_amplitudes, dt=2.0 * reference_pulse.dt)

    _parameters, metadata = load_custom_initial_parameters(
        npz_path,
        reference_pulse,
        parameterization,
    )

    assert metadata["source_dt"] == 2.0 * reference_pulse.dt
    assert metadata["dt_missing"] is False
    assert metadata["dt_mismatch"] is True
    assert "differs from experiment" in metadata["warnings"][0]


def test_single_perturbative_experiment_custom_npz_loader_allows_mismatched_dt(tmp_path):
    reference_pulse, parameterization = reference_pulse_and_parameterization(
        n_steps=5,
        alpha1_cycles=1.0,
    )
    compatible_amplitudes = parameterization.to_physical(
        parameterization.to_parameters(reference_pulse.amplitudes)
    )
    npz_path = tmp_path / "single_pulse.npz"
    np.savez(npz_path, amplitudes=compatible_amplitudes, dt=2.0 * reference_pulse.dt)

    parameters, metadata = load_single_custom_initial_parameters(
        npz_path,
        reference_pulse,
        parameterization,
    )

    assert np.allclose(parameterization.to_physical(parameters), compatible_amplitudes)
    assert metadata["dt_mismatch"] is True
    assert metadata["dt_missing"] is False


def test_perturbative_experiment_writes_latest_checkpoint_outputs(tmp_path):
    args = SimpleNamespace(
        maxiter=1,
        n_steps=5,
        alpha1_cycles=1.0,
        l1_smooth_weight=0.0,
        l2_smooth_weight=0.0,
        workers=1,
        print_step=False,
        print_fidelity_terms=False,
        initial_pulse_npz=None,
        no_progress=True,
    )

    result = run_perturbative_experiment(
        args,
        experiment_dir=tmp_path / "run",
        print_report=False,
    )

    outputs = result["outputs"]
    assert outputs["latest_pulse_npz"].exists()
    assert outputs["latest_pulse_csv"].exists()
    assert outputs["latest_parameters"].exists()
    data = np.load(outputs["latest_parameters"])
    assert int(data["step"]) >= 0
    report = result["report_path"].read_text(encoding="utf-8")
    assert "| latest_pulse_npz | latest_pulse.npz |" in report
    assert "| latest_parameters | latest_parameters.npz |" in report
    assert "## Preview" in report
    assert "## Noise Terms" in report
    assert "## System Construction Script" in report
    assert "### Kappa Diagnostics" in report
    assert "kappa_1" in report
    assert "kappa_2" in report
    assert "max_boundary_corner" in report
    assert "over alpha bounds" in report
    assert "H_nominal(alpha)" in report
    assert "H_fluctuation(alpha)" in report
    assert "kappa_boundary_corner_count" in report
    assert "static[0]" in report
    assert "314.159" in report
    assert "kron(0.5 * (sz ⊗ I + I ⊗ sz), I_motion)" in report
    assert "static[1]" in report
    assert "1256.637" in report
    assert "kron(I_spin, number_operator)" in report
    assert "control[0]" in report
    assert "0.0003" in report
    assert "alpha1(t) * control[0]" in report
    assert "control[1]" in report
    assert "0.0006" in report
    assert "eta * kron(S_phi(mode=(0.5, -0.5)), X1)" in report
    assert "eta=0.075" in report
    assert "alpha2(t) * control[1]" in report
    assert "## Results" in report
    assert report.index("## Preview") < report.index("## Results")


def test_perturbative_experiment_prints_kappa_when_step_printing_enabled(tmp_path, capsys):
    args = SimpleNamespace(
        maxiter=5,
        n_steps=5,
        alpha1_cycles=1.0,
        l1_smooth_weight=0.0,
        l2_smooth_weight=0.0,
        workers=1,
        print_step=True,
        print_fidelity_terms=False,
        initial_pulse_npz=None,
        no_progress=True,
    )

    with patch(
        "experiments.spin_boson_perturbative_lbfgsb.ScipyOptimizer.optimize_parameters",
        side_effect=KeyboardInterrupt,
    ):
        run_perturbative_experiment(
            args,
            experiment_dir=tmp_path / "print_kappa",
            print_report=False,
        )

    output = capsys.readouterr().out
    assert "=== Optimization Preview ===" in output
    assert "[Kappa Diagnostics]" in output
    assert "kappa_1" in output
    assert "kappa_2" in output
    assert "kappa_1_corner" in output
    assert "kappa_2_corner" in output


def test_spin_boson_noise_specs_drive_noisy_system_terms():
    specs = spin_boson_noise_term_specs(n_levels=3, phi_s=0.0)
    noisy_system = spin_boson_noisy_control_system(n_levels=3, phi_s=0.0)
    matrices = tuple(noisy_system.static_fluctuations) + tuple(noisy_system.control_fluctuations)

    assert [spec["name"] for spec in specs] == ["static[0]", "static[1]", "control[0]", "control[1]"]
    assert [spec["coefficient"] for spec in specs] == [314.159, 1256.637, 0.0003, 0.0006]
    for spec, matrix in zip(specs, matrices, strict=True):
        assert np.allclose(spec["matrix"], spec["coefficient"] * spec["operator"])
        assert np.allclose(spec["matrix"], matrix)


def test_kappa_metrics_use_alpha_bounds_corners_not_initial_pulse_samples():
    class ToySystem:
        def nominal_hamiltonian(self, controls):
            return np.array([[controls[0] + 2.0 * controls[1]]], dtype=complex)

    class ToyNoisySystem:
        def fluctuation_hamiltonian(self, controls):
            return np.array([[5.0 * controls[0] - controls[1]]], dtype=complex)

    class ToyBaseParameterization:
        def _bounds_for(self, shape):
            return (
                np.tile(np.array([-1.0, -2.0]), (shape[0], 1)),
                np.tile(np.array([3.0, 4.0]), (shape[0], 1)),
            )

    pulse = PiecewiseConstantPulse(
        amplitudes=np.array([[0.0, 0.0], [0.5, 0.5]], dtype=float),
        dt=0.1,
    )
    parameterization = SimpleNamespace(base=ToyBaseParameterization())

    metrics = calculate_kappa_metrics(ToySystem(), ToyNoisySystem(), pulse, parameterization)

    assert metrics["boundary_corner_count"] == 4
    assert np.isclose(metrics["kappa_1"], 1.1)
    assert np.isclose(metrics["kappa_2"], 3.4)
    assert metrics["kappa_1_alpha"] == (3.0, 4.0)
    assert metrics["kappa_2_alpha"] == (3.0, -2.0)


def test_perturbative_experiment_writes_preview_before_optimizer(tmp_path):
    args = SimpleNamespace(
        maxiter=5,
        n_steps=5,
        alpha1_cycles=1.0,
        l1_smooth_weight=0.0,
        l2_smooth_weight=0.0,
        workers=1,
        print_step=False,
        print_fidelity_terms=False,
        initial_pulse_npz=None,
        no_progress=True,
    )
    report_path = tmp_path / "preview_before_optimizer" / "report.md"

    def assert_preview_written(*_args, **_kwargs):
        assert report_path.exists()
        markdown = report_path.read_text(encoding="utf-8")
        assert "## Preview" in markdown
        assert "## Noise Terms" in markdown
        assert "## System Construction Script" in markdown
        assert "_Optimization has not completed yet. Final results will be appended here._" in markdown
        raise KeyboardInterrupt

    with patch(
        "experiments.spin_boson_perturbative_lbfgsb.ScipyOptimizer.optimize_parameters",
        side_effect=assert_preview_written,
    ):
        result = run_perturbative_experiment(
            args,
            experiment_dir=tmp_path / "preview_before_optimizer",
            print_report=False,
        )

    report = result["report_path"].read_text(encoding="utf-8")
    assert result["interrupted"] is True
    assert "_Optimization has not completed yet" not in report
    assert "## Results" in report
    assert report.index("## Preview") < report.index("## Results")


def test_perturbative_experiment_writes_fidelity_term_diagnostics(tmp_path):
    args = SimpleNamespace(
        maxiter=1,
        n_steps=5,
        alpha1_cycles=1.0,
        l1_smooth_weight=0.0,
        l2_smooth_weight=0.0,
        workers=1,
        print_step=False,
        print_fidelity_terms=False,
        save_fidelity_terms=True,
        initial_pulse_npz=None,
        no_progress=True,
    )

    result = run_perturbative_experiment(
        args,
        experiment_dir=tmp_path / "diagnostic_run",
        print_report=False,
    )

    outputs = result["outputs"]
    assert outputs["fidelity_terms"].exists()
    assert outputs["fidelity_terms_by_pair"].exists()
    summary_lines = outputs["fidelity_terms"].read_text(encoding="utf-8").splitlines()
    pair_lines = outputs["fidelity_terms_by_pair"].read_text(encoding="utf-8").splitlines()
    assert summary_lines[0].startswith("step,closed_term,first_order_sq")
    assert "dropped_order1_cross" in pair_lines[0]
    assert len(summary_lines) >= 2
    assert len(pair_lines) > len(summary_lines)
    report = result["report_path"].read_text(encoding="utf-8")
    assert "| fidelity_terms | fidelity_terms.csv |" in report
    assert "| fidelity_terms_by_pair | fidelity_terms_by_pair.csv |" in report


def test_perturbative_experiment_interrupt_uses_latest_pulse_for_report(tmp_path):
    args = SimpleNamespace(
        maxiter=5,
        n_steps=5,
        alpha1_cycles=1.0,
        l1_smooth_weight=0.0,
        l2_smooth_weight=0.0,
        workers=1,
        print_step=False,
        print_fidelity_terms=False,
        initial_pulse_npz=None,
        no_progress=True,
    )

    with patch(
        "experiments.spin_boson_perturbative_lbfgsb.ScipyOptimizer.optimize_parameters",
        side_effect=KeyboardInterrupt,
    ):
        result = run_perturbative_experiment(
            args,
            experiment_dir=tmp_path / "interrupted_run",
            print_report=False,
        )

    assert result["interrupted"] is True
    assert result["result"].success is False
    assert "INTERRUPTED" in result["result"].message
    assert result["outputs"]["final_pulse_npz"].exists()
    assert result["outputs"]["final_pulse_csv"].exists()
    assert result["report_path"].exists()
    report = result["report_path"].read_text(encoding="utf-8")
    assert "| interrupted | True |" in report


def test_sweep_custom_npz_rejects_incompatible_shape_endpoint_and_bounds(tmp_path):
    reference_pulse, parameterization = reference_pulse_and_parameterization(
        n_steps=5,
        alpha1_cycles=1.0,
    )
    bad_shape = tmp_path / "bad_shape.npz"
    bad_endpoint = tmp_path / "bad_endpoint.npz"
    bad_bounds = tmp_path / "bad_bounds.npz"
    np.savez(bad_shape, amplitudes=reference_pulse.amplitudes[:-1], dt=reference_pulse.dt)
    endpoint_amplitudes = np.array(reference_pulse.amplitudes, copy=True)
    endpoint_amplitudes[0, 1] = 1.0
    np.savez(bad_endpoint, amplitudes=endpoint_amplitudes, dt=reference_pulse.dt)
    bounds_amplitudes = np.array(reference_pulse.amplitudes, copy=True)
    bounds_amplitudes[1, 0] = 1e12
    np.savez(bad_bounds, amplitudes=bounds_amplitudes, dt=reference_pulse.dt)

    for path, message in [
        (bad_shape, "shape"),
        (bad_endpoint, "endpoints"),
        (bad_bounds, "bounds"),
    ]:
        try:
            load_custom_initial_parameters(path, reference_pulse, parameterization)
        except ValueError as exc:
            assert message in str(exc)
        else:
            raise AssertionError(f"Expected {path} to be rejected.")


def test_sweep_summary_writers_include_expected_columns_and_paths(tmp_path):
    rows = [
        {
            "mode": "custom",
            "run_index": 1,
            "seed": "NA",
            "source_npz": "pulse.npz",
            "source_dt": "NA",
            "experiment_dt": 0.1,
            "dt_missing": True,
            "dt_mismatch": False,
            "success": True,
            "nit": 1,
            "nfev": 2,
            "initial_open_gate_fidelity": 0.1,
            "final_open_gate_fidelity": 0.3,
            "initial_close_gate_fidelity": 0.2,
            "final_close_gate_fidelity": 0.4,
            "initial_cost": 0.05,
            "final_cost": 0.25,
            "initial_l1_penalty": 0.0,
            "final_l1_penalty": 0.01,
            "initial_l2_penalty": 0.0,
            "final_l2_penalty": 0.02,
            "experiment_dir": tmp_path / "run",
        }
    ]

    write_summary_csv(tmp_path / "summary.csv", rows)
    write_summary_markdown(tmp_path / "summary.md", rows)

    csv_lines = (tmp_path / "summary.csv").read_text(encoding="utf-8").splitlines()
    markdown = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert csv_lines[0].startswith("mode,run_index,seed,source_npz,source_dt")
    assert "pulse.npz" in csv_lines[1]
    assert "Top Runs by Final Cost" in markdown
    assert "| mode | run_index | seed | final_cost | final_open_gate_fidelity | experiment_dir |" in markdown
    assert str(tmp_path / "run") in markdown


def test_sweep_run_specs_include_both_modes_with_deterministic_labels(tmp_path):
    args = SimpleNamespace(
        initial_mode="both",
        n_runs=2,
        seed=10,
        noise_scale=0.3,
        initial_pulse_npz=[],
        n_steps=5,
        alpha1_cycles=1.0,
    )

    specs = build_sweep_run_specs(args, tmp_path, datetime(2026, 6, 22, 12, 0, 0))

    assert [spec["label"] for spec in specs] == [
        "noise_seed_10",
        "noise_seed_11",
        "random_seed_12",
        "random_seed_13",
    ]
    assert [spec["run_index"] for spec in specs] == [1, 2, 3, 4]
    assert [spec["mode"] for spec in specs] == ["noise", "noise", "random", "random"]
    assert all(spec["experiment_dir"].parent == tmp_path for spec in specs)


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


def test_grape_forward_backward_state_indexing_matches_kth_boundary():
    w1 = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    w2 = np.array([[1.0, 0.0], [0.0, 1.0j]], dtype=complex)
    w3 = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
    steps = [w1, w2, w3]
    initial_state = np.array([1.0, 0.0], dtype=complex)
    target_state = np.array([1.0 / np.sqrt(2.0), 1.0j / np.sqrt(2.0)], dtype=complex)

    forward = GrapeDifferentiator._forward_states(steps, initial_state)
    backward = GrapeDifferentiator._backward_states(steps, target_state)

    assert np.allclose(forward[2], w2 @ w1 @ initial_state)
    assert np.allclose(backward[2], w3.conj().T @ target_state)
    local_state = np.array([0.25 - 0.5j, -0.75 + 0.125j], dtype=complex)
    expected_bra_contraction = target_state.conj().T @ w3 @ local_state
    assert np.allclose(np.vdot(backward[2], local_state), expected_bra_contraction)


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
    system = IonTrapRFSystem(
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
    system = IonTrapRFSystem(
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
                "from quantum_control import IonTrapRFSystem, StatePair",
                "",
                "def build():",
                "    sx = np.array([[0, 1], [1, 0]], dtype=complex)",
                "    sz = np.array([[1, 0], [0, -1]], dtype=complex)",
                "    system = IonTrapRFSystem(",
                "        drift=0.1 * sz,",
                "        controls=[sx],",
                "        static_fluctuations=[0.01 * sz],",
                "        control_fluctuations=[0.02 * sx],",
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
    serial = ExpansionStateAverageFidelity(
        system=problem.system,
        pulse=problem.pulse,
        evolution=problem.evolution,
        objective=problem.objective,
        differentiator=problem.differentiator,
        state_pairs=state_pairs,
    )
    parallel = ExpansionStateAverageFidelity(
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
        ExpansionStateAverageFidelity(
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
