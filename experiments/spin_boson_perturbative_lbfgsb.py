from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT_DIR / ".matplotlib"))
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from experiments.reporting import (
    FidelityTermsLog,
    StepLog,
    export_pulse_controls,
    timestamped_experiment_dir,
)
from quantum_control import (
    DEFAULT_ALPHA1_KHZ_BOUNDS,
    DEFAULT_ALPHA2_KHZ_BOUNDS,
    DEFAULT_LAMB_DICKE_ETA,
    ExpansionFidelity,
    ExpansionStateAverageFidelity,
    EvolutionContext,
    NominalUnitaryEvolution,
    ParameterSmoothPenalty,
    ParameterizedControlProblem,
    PenalizedParameterizedProblem,
    PerturbativeExpansionDifferentiator,
    PerturbativeExpansionEvolution,
    PerturbativeStepBuilder,
    StateTransferFidelity,
    UnitaryStepBuilder,
    annihilation_operator,
    closed_gate_fidelity,
    creation_operator,
    motion_resolved_gate_state_pairs,
    ms_xx_pi_over_2_gate,
    number_operator,
    open_gate_fidelity,
    spin_boson_control_system,
    spin_boson_initial_pulse,
    spin_boson_parameterization,
    two_qubit_spin_phase_mode,
)
from quantum_control.optimizers import ScipyOptimizer
from quantum_control.pulses.pulse import PiecewiseConstantPulse

N_LEVELS = 6
N_STEPS = 200
MAXITER = 20
RAD_S_PER_KHZ = 2.0 * np.pi * 1000.0
DEFAULT_L1_SMOOTH_WEIGHT = 0.0001
DEFAULT_L2_SMOOTH_WEIGHT = 0.00005


class OptimizationProgressBar:
    def __init__(self, maxiter, width=32):
        self.maxiter = max(1, int(maxiter))
        self.width = width
        self.iteration = 0

    def start(self):
        self._write()

    def __call__(self, _xk):
        self.iteration = min(self.iteration + 1, self.maxiter)
        self._write()

    def finish(self, result):
        final_iteration = min(int(getattr(result, "nit", self.iteration)), self.maxiter)
        if final_iteration != self.iteration:
            self.iteration = final_iteration
            self._write()
        sys.stderr.write("\n")
        sys.stderr.flush()

    def _write(self):
        filled = int(round(self.width * self.iteration / self.maxiter))
        bar = "#" * filled + "-" * (self.width - filled)
        sys.stderr.write(f"\roptimizing [{bar}] {self.iteration}/{self.maxiter}")
        sys.stderr.flush()


def basis_state(index, dimension):
    state = np.zeros(dimension, dtype=complex)
    state[index] = 1.0
    return state


def ms_bell_target_motion_ground(n_levels):
    dimension = 4 * n_levels
    target = np.zeros(dimension, dtype=complex)
    target[0] = 1.0 / np.sqrt(2.0)
    target[3 * n_levels] = -1j / np.sqrt(2.0)
    return target


def spin_boson_noisy_control_system(n_levels, phi_s):
    specs = spin_boson_noise_term_specs(n_levels, phi_s)
    return spin_boson_control_system(
        n_levels=n_levels,
        phi_s=phi_s,
        static_fluctuations=[spec["matrix"] for spec in specs if spec["kind"] == "static"],
        control_fluctuations=[spec["matrix"] for spec in specs if spec["kind"] == "control"],
    )


def spin_boson_noise_term_specs(n_levels, phi_s):
    spin_identity = np.eye(4, dtype=complex)
    motion_identity = np.eye(n_levels, dtype=complex)
    single_identity = np.eye(2, dtype=complex)
    sz = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    sz1_plus_sz2 = np.kron(sz, single_identity) + np.kron(single_identity, sz)
    number = number_operator(n_levels)
    x1 = 0.5 * (annihilation_operator(n_levels) + creation_operator(n_levels))
    s_phi = two_qubit_spin_phase_mode(phi_s, (0.5, -0.5))

    return [
        _noise_term_spec(
            kind="static",
            name="static[0]",
            coefficient=314.159 * 0.1,
            operator=np.kron(0.5 * sz1_plus_sz2, motion_identity),
            definition="kron(0.5 * (sz ⊗ I + I ⊗ sz), I_motion)",
            usage="added directly to H_fluctuation",
        ),
        _noise_term_spec(
            kind="static",
            name="static[1]",
            coefficient=300 * 0.1,
            operator=np.kron(spin_identity, number),
            definition="kron(I_spin, number_operator)",
            usage="added directly to H_fluctuation",
        ),
        _noise_term_spec(
            kind="control",
            name="control[0]",
            coefficient=0.0001,
            operator=np.kron(spin_identity, number),
            definition="kron(I_spin, number_operator)",
            usage="alpha1(t) * control[0]",
        ),
        _noise_term_spec(
            kind="control",
            name="control[1]",
            coefficient=0.0001,
            operator=DEFAULT_LAMB_DICKE_ETA * np.kron(s_phi, x1),
            definition="eta * kron(S_phi(mode=(0.5, -0.5)), X1), X1=(a + adag)/2, eta=0.075",
            usage="alpha2(t) * control[1]",
        ),
    ]


def _noise_term_spec(kind, name, coefficient, operator, definition, usage):
    coefficient = float(coefficient)
    operator = np.asarray(operator, dtype=complex)
    return {
        "kind": kind,
        "name": name,
        "coefficient": coefficient,
        "operator": operator,
        "definition": definition,
        "usage": usage,
        "matrix": coefficient * operator,
    }


class Alpha2EndpointZeroParameterization:
    def __init__(self, base):
        self.base = base

    def to_physical(self, normalized):
        amplitudes = self.base.to_physical(normalized)
        amplitudes[[0, -1], 1] = 0.0
        return amplitudes

    def to_parameters(self, amplitudes):
        parameters = self.base.to_parameters(amplitudes)
        lower, upper = self.base._bounds_for(amplitudes.shape)
        parameters[[0, -1], 1] = self._normalized_zero(lower[[0, -1], 1], upper[[0, -1], 1])
        return parameters

    def pullback_gradient(self, physical_gradient):
        gradient = self.base.pullback_gradient(physical_gradient)
        gradient[[0, -1], 1] = 0.0
        return gradient

    def parameter_bounds(self, shape):
        bounds = self.base.parameter_bounds(shape)
        lower, upper = self.base._bounds_for(shape)
        endpoint_value = self._normalized_zero(lower[[0, -1], 1], upper[[0, -1], 1])
        for row, value in zip((0, shape[0] - 1), endpoint_value):
            bounds[np.ravel_multi_index((row, 1), shape)] = (float(value), float(value))
        return bounds

    @staticmethod
    def _normalized_zero(lower, upper):
        return (0.0 - 0.5 * (upper + lower)) / (0.5 * (upper - lower))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run spin-boson perturbative open-gate L-BFGS-B experiment."
    )
    parser.add_argument("--maxiter", type=int, default=MAXITER)
    parser.add_argument("--n-steps", type=int, default=N_STEPS)
    parser.add_argument("--alpha1-cycles", type=float, default=1.0)
    parser.add_argument(
        "--l1-smooth-weight",
        type=float,
        default=DEFAULT_L1_SMOOTH_WEIGHT,
    )
    parser.add_argument(
        "--l2-smooth-weight",
        type=float,
        default=DEFAULT_L2_SMOOTH_WEIGHT,
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes for perturbative state-pair averaging.",
    )
    parser.add_argument(
        "--print-step",
        action="store_true",
        help="Print per-step close fidelity, open fidelity, and cost function.",
    )
    parser.add_argument(
        "--print-fidelity-terms",
        action="store_true",
        help="Print and save per-step perturbative fidelity term diagnostics.",
    )
    parser.add_argument(
        "--initial-pulse-npz",
        type=Path,
        default=None,
        help="Load a custom initial pulse .npz with an amplitudes array.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the optimization progress bar.",
    )
    return parser.parse_args()


class CombinedCallback:
    def __init__(self, *callbacks):
        self.callbacks = tuple(callback for callback in callbacks if callback is not None)

    def __call__(self, parameters):
        for callback in self.callbacks:
            callback(parameters)


def propagate_states(evolution, system, pulse, context):
    result = evolution.evolve(system, pulse, context)
    states = [context.initial_state]
    state = context.initial_state
    for step in result.W_steps:
        state = step @ state
        states.append(state)
    return np.asarray(states)


def add_experiment_note(fig, note):
    if not note:
        return
    fig.text(
        0.685,
        0.785,
        note,
        ha="right",
        va="bottom",
        fontsize=8,
        family="monospace",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.7},
    )


def plot_pulses(time_us, initial_pulse, final_pulse, output_path, note=None):
    initial_khz = initial_pulse.amplitudes / RAD_S_PER_KHZ
    final_khz = final_pulse.amplitudes / RAD_S_PER_KHZ

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].plot(time_us, initial_khz[:, 0], label="initial alpha1", linewidth=2)
    axes[0].plot(time_us, final_khz[:, 0], label="final alpha1", linewidth=2)
    axes[0].set_ylabel("alpha1 (kHz)")
    axes[0].legend(loc="best")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(time_us, initial_khz[:, 1], label="initial alpha2", linewidth=2)
    axes[1].plot(time_us, final_khz[:, 1], label="final alpha2", linewidth=2)
    axes[1].set_xlabel("time (us)")
    axes[1].set_ylabel("alpha2 (kHz)")
    axes[1].legend(loc="best")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Perturbative open-gate optimization: pulse parameters")
    add_experiment_note(fig, note)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_population_marginals(
    time_edges_us,
    initial_states,
    final_states,
    n_levels,
    output_path,
    note=None,
):
    initial_populations = np.abs(initial_states) ** 2
    final_populations = np.abs(final_states) ** 2
    initial_joint = initial_populations.reshape((-1, 4, n_levels))
    final_joint = final_populations.reshape((-1, 4, n_levels))
    initial_spin = initial_joint.sum(axis=2)
    final_spin = final_joint.sum(axis=2)
    initial_motion = initial_joint.sum(axis=1)
    final_motion = final_joint.sum(axis=1)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey="row")
    spin_labels = ["00", "01", "10", "11"]
    for spin_index, label in enumerate(spin_labels):
        axes[0, 0].plot(time_edges_us, initial_spin[:, spin_index], label=f"|{label}>")
        axes[0, 1].plot(time_edges_us, final_spin[:, spin_index], label=f"|{label}>")
    for level in range(n_levels):
        axes[1, 0].plot(time_edges_us, initial_motion[:, level], label=f"n={level}")
        axes[1, 1].plot(time_edges_us, final_motion[:, level], label=f"n={level}")

    axes[0, 0].set_title("Two-qubit spin population, initial pulse")
    axes[0, 1].set_title("Two-qubit spin population, optimized pulse")
    axes[1, 0].set_title("Motion population, initial pulse")
    axes[1, 1].set_title("Motion population, optimized pulse")
    axes[1, 0].set_xlabel("time (us)")
    axes[1, 1].set_xlabel("time (us)")
    for axis in axes:
        for item in axis:
            item.grid(True, alpha=0.3)
            item.set_ylim(-0.02, 1.02)
    axes[0, 0].set_ylabel("spin population")
    axes[1, 0].set_ylabel("motion population")

    state_handles, state_labels = axes[0, 0].get_legend_handles_labels()
    motion_handles, motion_labels = axes[1, 0].get_legend_handles_labels()
    axes[0, 1].legend(state_handles, state_labels, loc="best")
    axes[1, 1].legend(motion_handles, motion_labels, loc="best", ncol=2)
    fig.suptitle("Nominal state propagation after perturbative optimization")
    add_experiment_note(fig, note)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def format_experiment_note(args, result, metrics):
    return "\n".join(
        [
            "objective=open_gate_fidelity_expansion, target=MS_XX(pi/2)",
            (
                f"n_steps={args.n_steps}, maxiter={args.maxiter}, "
                f"workers={args.workers}, alpha1_cycles={args.alpha1_cycles:.6g}"
            ),
            (
                f"smooth: l1={args.l1_smooth_weight:.6g}, "
                f"l2={args.l2_smooth_weight:.6g}"
            ),
            (
                f"open gate: {metrics['initial_open_gate_fidelity']:.6g} -> "
                f"{metrics['final_open_gate_fidelity']:.6g}; "
                f"close gate: {metrics['initial_close_gate_fidelity']:.6g} -> "
                f"{metrics['final_close_gate_fidelity']:.6g}"
            ),
            (
                f"objective: {metrics['initial_penalized_objective']:.6g} -> "
                f"{metrics['final_penalized_objective']:.6g}; "
                f"nit={getattr(result, 'nit', 'NA')}, nfev={getattr(result, 'nfev', 'NA')}"
            ),
        ]
    )


def print_section(title, rows):
    print(f"\n[{title}]")
    width = max(len(name) for name, _ in rows)
    for name, value in rows:
        print(f"{name:<{width}} : {value}")


def format_value(value):
    if isinstance(value, float):
        return f"{value:.12g}"
    return str(value)


def print_experiment_report(args, result, metrics, outputs):
    print("\n=== Perturbative Open-Gate Optimization ===")
    print_section(
        "Configuration",
        [
            ("objective", "open_gate_fidelity_expansion"),
            ("target_gate", "MS_XX(pi/2)"),
            ("n_levels", N_LEVELS),
            ("n_steps", args.n_steps),
            ("maxiter", args.maxiter),
            ("workers", args.workers),
            ("alpha1_cycles", format_value(args.alpha1_cycles)),
            ("l1_smooth_weight", format_value(args.l1_smooth_weight)),
            ("l2_smooth_weight", format_value(args.l2_smooth_weight)),
        ],
    )
    print_section(
        "Fidelity",
        [
            ("single_state", _transition(metrics["initial_fidelity"], metrics["final_fidelity"])),
            (
                "close_gate",
                _transition(
                    metrics["initial_close_gate_fidelity"],
                    metrics["final_close_gate_fidelity"],
                ),
            ),
            (
                "open_gate",
                _transition(
                    metrics["initial_open_gate_fidelity"],
                    metrics["final_open_gate_fidelity"],
                ),
            ),
        ],
    )
    print_section(
        "Objective And Penalty",
        [
            (
                "penalized_objective",
                _transition(
                    metrics["initial_penalized_objective"],
                    metrics["final_penalized_objective"],
                ),
            ),
            ("l1_penalty", _transition(metrics["initial_l1_penalty"], metrics["final_l1_penalty"])),
            ("l2_penalty", _transition(metrics["initial_l2_penalty"], metrics["final_l2_penalty"])),
        ],
    )
    print_section(
        "Optimizer",
        [
            ("success", result.success),
            ("message", result.message),
            ("nit", getattr(result, "nit", "NA")),
            ("nfev", getattr(result, "nfev", "NA")),
        ],
    )
    print_section(
        "Outputs",
        [
            ("pulse_plot", outputs["pulse_plot"]),
            ("propagation_plot", outputs["propagation_plot"]),
            ("step_log", outputs["step_log"]),
            *(
                [
                    ("fidelity_terms", outputs["fidelity_terms"]),
                    ("fidelity_terms_by_pair", outputs["fidelity_terms_by_pair"]),
                ]
                if outputs.get("fidelity_terms") is not None
                else []
            ),
            ("initial_pulse_npz", outputs["initial_pulse_npz"]),
            ("initial_pulse_csv", outputs["initial_pulse_csv"]),
            ("final_pulse_npz", outputs["final_pulse_npz"]),
            ("final_pulse_csv", outputs["final_pulse_csv"]),
        ],
    )


def _transition(initial, final):
    delta = final - initial
    return f"{initial:.12g} -> {final:.12g} (delta {delta:+.3g})"


def write_optimization_preview_report(
    report_path,
    *,
    generated_at,
    experiment_dir,
    args,
    initial_pulse,
    parameterization,
    noisy_system,
    noise_specs,
    phi_s,
    state_pairs,
    optimizer_options,
    output_manifest,
    initial_metrics,
    kappa_metrics,
    custom_initial_metadata,
    extra_configuration,
    save_fidelity_terms,
):
    lower, upper = parameterization.base._bounds_for(initial_pulse.amplitudes.shape)
    bounds_lower_khz = lower[0] / RAD_S_PER_KHZ
    bounds_upper_khz = upper[0] / RAD_S_PER_KHZ
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Spin-Boson Perturbative Open-Gate Optimization",
        "",
        f"Generated at: {generated_at.isoformat(timespec='seconds')}",
        "",
        "## Preview",
        "",
        "### Configuration",
        "",
        _markdown_table(
            ("Parameter", "Value"),
            [
                ("experiment_dir", experiment_dir),
                ("objective", "open_gate_fidelity_expansion"),
                ("target_state", "(|00,0>-i|11,0>)/sqrt(2)"),
                ("target_gate", "MS_XX(pi/2)"),
                ("n_levels", N_LEVELS),
                ("n_steps", args.n_steps),
                ("dt_s", initial_pulse.dt),
                ("total_time_us", initial_pulse.n_steps * initial_pulse.dt * 1e6),
                ("phi_s", phi_s),
                ("eta", DEFAULT_LAMB_DICKE_ETA),
                ("alpha1_cycles", args.alpha1_cycles),
                ("alpha1_bounds_khz", f"{bounds_lower_khz[0]:.12g} to {bounds_upper_khz[0]:.12g}"),
                ("alpha2_bounds_khz", f"{bounds_lower_khz[1]:.12g} to {bounds_upper_khz[1]:.12g}"),
                ("alpha2_endpoint_constraint", "initial and final alpha2 fixed to 0"),
                ("static_fluctuation_count", len(noisy_system.static_fluctuations)),
                ("control_fluctuation_count", len(noisy_system.control_fluctuations)),
                ("max_order", 2),
                ("drop_odd_average", True),
                ("workers", args.workers),
                ("normalize_weights", False),
                ("no_progress", args.no_progress),
                ("print_step", args.print_step),
                ("print_fidelity_terms", getattr(args, "print_fidelity_terms", False)),
                ("save_fidelity_terms", save_fidelity_terms),
                ("state_pair_count", len(state_pairs)),
                ("l1_smooth_weight", args.l1_smooth_weight),
                ("l2_smooth_weight", args.l2_smooth_weight),
                *custom_initial_configuration(custom_initial_metadata),
                *extra_configuration,
            ],
        ),
        "",
        "### Optimizer",
        "",
        _markdown_table(
            ("Parameter", "Value"),
            [
                ("optimizer_method", "L-BFGS-B"),
                ("optimizer_maximize", True),
                ("optimizer_options", optimizer_options),
            ],
        ),
        "",
        "### Initial Metrics",
        "",
        _markdown_table(("Metric", "Value"), initial_metrics),
        "",
        "### Kappa Diagnostics",
        "",
        _markdown_table(("Metric", "Value", "Definition"), kappa_report_rows(kappa_metrics)),
        "",
        "### Output Manifest",
        "",
        _markdown_table(("Output", "Path"), output_manifest),
        "",
        "## Noise Terms",
        "",
        _markdown_table(
            (
                "Term",
                "Coefficient",
                "Definition",
                "Usage",
                "Shape",
                "Frobenius Norm",
                "Spectral Norm",
                "Zero",
            ),
            noise_term_rows(noisy_system, noise_specs),
        ),
        "",
        "## System Construction Script",
        "",
        "```python",
        render_system_construction_script(args, optimizer_options),
        "```",
        "",
        "## Results",
        "",
        "_Optimization has not completed yet. Final results will be appended here._",
        "",
    ]
    report_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return report_path


def append_optimization_results_report(
    report_path,
    *,
    metrics,
    result,
    figures,
    outputs,
    interrupted,
):
    report_path = Path(report_path)
    existing = report_path.read_text(encoding="utf-8") if report_path.exists() else ""
    placeholder = "## Results\n\n_Optimization has not completed yet. Final results will be appended here._\n"
    result_lines = [
        "## Results",
        "",
        _markdown_table(
            ("Metric", "Initial", "Final", "Delta"),
            [
                _result_row("single_state_fidelity", metrics["initial_fidelity"], metrics["final_fidelity"]),
                _result_row(
                    "close_gate_fidelity",
                    metrics["initial_close_gate_fidelity"],
                    metrics["final_close_gate_fidelity"],
                ),
                _result_row(
                    "open_gate_fidelity",
                    metrics["initial_open_gate_fidelity"],
                    metrics["final_open_gate_fidelity"],
                ),
                _result_row("l1_penalty", metrics["initial_l1_penalty"], metrics["final_l1_penalty"]),
                _result_row("l2_penalty", metrics["initial_l2_penalty"], metrics["final_l2_penalty"]),
                _result_row(
                    "penalized_objective",
                    metrics["initial_penalized_objective"],
                    metrics["final_penalized_objective"],
                ),
            ],
        ),
        "",
        "## Optimizer",
        "",
        _markdown_table(
            ("Parameter", "Value"),
            [
                ("success", result.success),
                ("message", result.message),
                ("nit", getattr(result, "nit", "NA")),
                ("nfev", getattr(result, "nfev", "NA")),
                ("interrupted", interrupted),
            ],
        ),
        "",
        "## Final Outputs",
        "",
        _markdown_table(("Output", "Path"), outputs.items()),
        "",
        "## Figures",
        "",
    ]
    for label, figure_path in figures:
        relative_path = Path(figure_path).relative_to(report_path.parent)
        result_lines.extend([f"### {_format_markdown_cell(label)}", "", f"![{label}]({relative_path.as_posix()})", ""])
    result_markdown = "\n".join(result_lines).rstrip() + "\n"
    if placeholder in existing:
        report_path.write_text(existing.replace(placeholder, result_markdown), encoding="utf-8")
    else:
        report_path.write_text(existing.rstrip() + "\n\n" + result_markdown, encoding="utf-8")
    return report_path


def noise_term_rows(system, specs):
    matrices = tuple(system.static_fluctuations) + tuple(system.control_fluctuations)
    rows = []
    for spec, matrix in zip(specs, matrices, strict=True):
        matrix = np.asarray(matrix)
        rows.append(
            (
                spec["name"],
                spec["coefficient"],
                spec["definition"],
                spec["usage"],
                "x".join(str(size) for size in matrix.shape),
                float(np.linalg.norm(matrix, ord="fro")),
                float(np.linalg.norm(matrix, ord=2)),
                bool(np.allclose(matrix, 0.0)),
            )
        )
    return rows


def calculate_kappa_metrics(system, noisy_system, pulse, parameterization):
    lower, upper = parameterization.base._bounds_for(pulse.amplitudes.shape)
    channel_bounds = tuple(zip(lower[0], upper[0], strict=True))
    boundary_controls = np.asarray(
        np.meshgrid(*(bound for bound in channel_bounds), indexing="ij"),
        dtype=float,
    ).reshape(len(channel_bounds), -1).T
    nominal_norms = []
    fluctuation_norms = []
    for controls in boundary_controls:
        nominal_norms.append(float(np.linalg.norm(system.nominal_hamiltonian(controls), ord=2)))
        fluctuation_norms.append(float(np.linalg.norm(noisy_system.fluctuation_hamiltonian(controls), ord=2)))
    nominal_norms = np.asarray(nominal_norms, dtype=float)
    fluctuation_norms = np.asarray(fluctuation_norms, dtype=float)
    kappa_1_corner = int(np.argmax(nominal_norms)) if nominal_norms.size else 0
    kappa_2_corner = int(np.argmax(fluctuation_norms)) if fluctuation_norms.size else 0
    total_time = float(pulse.n_steps * pulse.dt)
    return {
        "kappa_1": float(pulse.dt * nominal_norms[kappa_1_corner]),
        "kappa_2": float(total_time * fluctuation_norms[kappa_2_corner]),
        "kappa_1_corner": kappa_1_corner,
        "kappa_2_corner": kappa_2_corner,
        "kappa_1_alpha": tuple(float(value) for value in boundary_controls[kappa_1_corner]),
        "kappa_2_alpha": tuple(float(value) for value in boundary_controls[kappa_2_corner]),
        "kappa_1_h_norm": float(nominal_norms[kappa_1_corner]),
        "kappa_2_h_fluc_norm": float(fluctuation_norms[kappa_2_corner]),
        "dt_s": float(pulse.dt),
        "total_time_s": total_time,
        "boundary_corner_count": int(len(boundary_controls)),
    }


def kappa_report_rows(metrics):
    return [
        (
            "kappa_1",
            metrics["kappa_1"],
            "max_boundary_corner dt * ||H_nominal(alpha)||_2 over alpha bounds",
        ),
        (
            "kappa_2",
            metrics["kappa_2"],
            "max_boundary_corner T * ||H_fluctuation(alpha)||_2 over alpha bounds; fluctuation terms are already scaled",
        ),
        ("kappa_1_corner", metrics["kappa_1_corner"], "boundary corner attaining max ||H_nominal||_2"),
        ("kappa_2_corner", metrics["kappa_2_corner"], "boundary corner attaining max ||H_fluctuation||_2"),
        ("kappa_1_alpha", metrics["kappa_1_alpha"], "alpha values at kappa_1 corner in rad/s"),
        ("kappa_2_alpha", metrics["kappa_2_alpha"], "alpha values at kappa_2 corner in rad/s"),
        ("kappa_1_h_norm", metrics["kappa_1_h_norm"], "max ||H_nominal||_2"),
        ("kappa_2_h_fluc_norm", metrics["kappa_2_h_fluc_norm"], "max ||H_fluctuation||_2"),
        ("kappa_dt_s", metrics["dt_s"], "pulse time step"),
        ("kappa_total_time_s", metrics["total_time_s"], "pulse total duration"),
        ("kappa_boundary_corner_count", metrics["boundary_corner_count"], "number of alpha-boundary corners evaluated"),
    ]


def render_system_construction_script(args, optimizer_options):
    return "\n".join(
        [
            "phi_s = 0.0",
            f"eta = {DEFAULT_LAMB_DICKE_ETA!r}",
            f"system = spin_boson_control_system(n_levels={N_LEVELS}, phi_s=phi_s, eta=eta)",
            f"noisy_system = spin_boson_noisy_control_system(n_levels={N_LEVELS}, phi_s=phi_s)",
            f"alpha1_khz_bounds = {DEFAULT_ALPHA1_KHZ_BOUNDS!r}",
            f"alpha2_khz_bounds = {DEFAULT_ALPHA2_KHZ_BOUNDS!r}",
            "initial_pulse = _customized_initial_pulse(",
            f"    n_steps={args.n_steps},",
            "    alpha1_khz_bounds=alpha1_khz_bounds,",
            "    alpha2_khz_bounds=alpha2_khz_bounds,",
            f"    alpha1_cycles={args.alpha1_cycles!r},",
            ")",
            "parameterization = Alpha2EndpointZeroParameterization(",
            "    spin_boson_parameterization(",
            "        initial_pulse.n_steps,",
            "        alpha1_khz_bounds=alpha1_khz_bounds,",
            "        alpha2_khz_bounds=alpha2_khz_bounds,",
            "    )",
            ")",
            "target_gate = ms_xx_pi_over_2_gate()",
            f"state_pairs = motion_resolved_gate_state_pairs(target_gate, {N_LEVELS})",
            "step_builder = PerturbativeStepBuilder()",
            "expansion_objective = ExpansionFidelity(max_order=2, drop_odd_average=True)",
            "optimization_problem = ExpansionStateAverageFidelity(",
            "    system=noisy_system,",
            "    pulse=initial_pulse,",
            "    evolution=PerturbativeExpansionEvolution(step_builder, max_order=2),",
            "    objective=expansion_objective,",
            "    differentiator=PerturbativeExpansionDifferentiator(step_builder, expansion_objective),",
            "    state_pairs=state_pairs,",
            "    normalize_weights=False,",
            f"    n_workers={args.workers},",
            ")",
            "parameterized_problem = ParameterizedControlProblem(optimization_problem, parameterization)",
            "penalty = ParameterSmoothPenalty(",
            f"    l1_weight={args.l1_smooth_weight!r},",
            f"    l2_weight={args.l2_smooth_weight!r},",
            ")",
            "penalized_problem = PenalizedParameterizedProblem(parameterized_problem, penalty)",
            "optimizer = ScipyOptimizer(",
            "    method='L-BFGS-B',",
            "    maximize=True,",
            f"    options={optimizer_options!r},",
            ")",
        ]
    )


def print_optimization_preview(report_path, args, initial_metrics, kappa_metrics):
    print("\n=== Optimization Preview ===")
    print_section(
        "Configuration",
        [
            ("n_levels", N_LEVELS),
            ("n_steps", args.n_steps),
            ("maxiter", args.maxiter),
            ("workers", args.workers),
            ("alpha1_cycles", format_value(args.alpha1_cycles)),
            ("l1_smooth_weight", format_value(args.l1_smooth_weight)),
            ("l2_smooth_weight", format_value(args.l2_smooth_weight)),
        ],
    )
    print_section("Initial Metrics", [(name, format_value(value)) for name, value in initial_metrics])
    print_section(
        "Kappa Diagnostics",
        [
            ("kappa_1", format_value(kappa_metrics["kappa_1"])),
            ("kappa_2", format_value(kappa_metrics["kappa_2"])),
            ("kappa_1_corner", kappa_metrics["kappa_1_corner"]),
            ("kappa_2_corner", kappa_metrics["kappa_2_corner"]),
        ],
    )
    print(f"preview_report={report_path}")


def _result_row(name, initial, final):
    return (name, _format_markdown_value(initial), _format_markdown_value(final), _format_markdown_value(final - initial))


def _markdown_table(headers, rows):
    table = [
        "| " + " | ".join(_format_markdown_cell(header) for header in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        table.append("| " + " | ".join(_format_markdown_cell(value) for value in row) + " |")
    return "\n".join(table)


def _format_markdown_cell(value):
    return str(_format_markdown_value(value)).replace("|", "\\|").replace("\n", "<br>")


def _format_markdown_value(value):
    if isinstance(value, Path):
        return value.name
    if isinstance(value, float):
        return f"{value:.12g}"
    if value is None:
        return "disabled"
    return value


def _customized_initial_pulse(
    n_steps=200,
    total_time_us=225.8,
    alpha1_khz_bounds=DEFAULT_ALPHA1_KHZ_BOUNDS,
    alpha2_khz_bounds=DEFAULT_ALPHA2_KHZ_BOUNDS,
    alpha1_cycles=1.0,
):
    if n_steps < 1:
        raise ValueError("n_steps must be at least 1.")
    total_time = float(total_time_us) * 1e-6
    if total_time <= 0.0:
        raise ValueError("total_time_us must be positive.")
    if alpha1_cycles <= 0.0:
        raise ValueError("alpha1_cycles must be positive.")

    alpha1_lower, alpha1_upper = _khz_bounds_to_rad_s(alpha1_khz_bounds)
    alpha2_lower, alpha2_upper = _khz_bounds_to_rad_s(alpha2_khz_bounds)
    dt = total_time / n_steps
    normalized_time = (np.arange(n_steps, dtype=float) + 0.5) / n_steps

    alpha1_center = 0.5 * (alpha1_upper + alpha1_lower)
    alpha1_scale = 0.5 * (alpha1_upper - alpha1_lower)
    alpha1 = alpha1_center + 0.7 * alpha1_scale + 0.3 * alpha1_scale * np.random.randn(n_steps)
    alpha1 = np.clip(alpha1, alpha1_lower, alpha1_upper)
    alpha2 = alpha2_upper * np.ones(n_steps, dtype=float)

    return PiecewiseConstantPulse(
        amplitudes=np.column_stack([alpha1, alpha2]),
        dt=dt,
    )
def _khz_bounds_to_rad_s(bounds):
    lower, upper = np.asarray(bounds, dtype=float)
    if upper <= lower:
        raise ValueError("upper bounds must be greater than lower bounds.")
    return 2.0 * np.pi * 1000.0 * lower, 2.0 * np.pi * 1000.0 * upper


def load_custom_initial_parameters(npz_path, reference_pulse, parameterization, atol=1e-9):
    npz_path = Path(npz_path)
    with np.load(npz_path) as data:
        if "amplitudes" not in data.files:
            raise ValueError(f"{npz_path} does not contain required 'amplitudes'.")
        amplitudes = np.asarray(data["amplitudes"], dtype=float)
        source_dt = float(np.asarray(data["dt"]).reshape(())) if "dt" in data.files else None

    if amplitudes.shape != reference_pulse.amplitudes.shape:
        raise ValueError(
            f"{npz_path} amplitudes shape {amplitudes.shape} does not match "
            f"expected {reference_pulse.amplitudes.shape}."
        )
    if not np.allclose(amplitudes[[0, -1], 1], 0.0, atol=atol):
        raise ValueError(f"{npz_path} alpha2 endpoints must be zero.")

    parameters = parameterization.to_parameters(amplitudes)
    if np.any(parameters < -1.0 - atol) or np.any(parameters > 1.0 + atol):
        raise ValueError(f"{npz_path} amplitudes exceed the configured parameter bounds.")
    parameters = np.clip(parameters, -1.0, 1.0)

    experiment_dt = float(reference_pulse.dt)
    dt_missing = source_dt is None
    dt_mismatch = False if source_dt is None else not np.isclose(source_dt, experiment_dt)
    warnings = []
    if dt_missing:
        warnings.append(f"warning: {npz_path} has no dt; using experiment dt {experiment_dt:.12g}.")
    elif dt_mismatch:
        warnings.append(
            f"warning: {npz_path} dt={source_dt:.12g} differs from experiment "
            f"dt={experiment_dt:.12g}; using experiment dt."
        )

    return parameters, {
        "source_npz": str(npz_path),
        "source_dt": "NA" if source_dt is None else source_dt,
        "experiment_dt": experiment_dt,
        "dt_missing": dt_missing,
        "dt_mismatch": dt_mismatch,
        "warnings": warnings,
    }


def perturbative_fidelity_terms(system, pulse, target_gate, n_levels):
    step_builder = PerturbativeStepBuilder()
    objective = ExpansionFidelity(max_order=2, drop_odd_average=True)
    evolution = PerturbativeExpansionEvolution(step_builder, max_order=2)
    pair_rows = []
    for pair_index, pair in enumerate(motion_resolved_gate_state_pairs(target_gate, n_levels)):
        context = EvolutionContext(
            initial_state=pair.initial_state,
            target_state=pair.target_state,
        )
        amplitudes = objective.amplitudes(evolution.evolve(system, pulse, context))
        a0 = amplitudes.get(0, 0.0 + 0.0j)
        a1 = amplitudes.get(1, 0.0 + 0.0j)
        a2 = amplitudes.get(2, 0.0 + 0.0j)
        weight = float(pair.weight)
        closed_term = weight * float(np.abs(a0) ** 2)
        first_order_sq = weight * float(np.abs(a1) ** 2)
        second_order_cross = weight * float(2.0 * np.real(np.conj(a0) * a2))
        dropped_order1_cross = weight * float(2.0 * np.real(np.conj(a0) * a1))
        pair_rows.append(
            {
                "pair_index": pair_index,
                "weight": weight,
                "a0_real": float(np.real(a0)),
                "a0_imag": float(np.imag(a0)),
                "a1_real": float(np.real(a1)),
                "a1_imag": float(np.imag(a1)),
                "a2_real": float(np.real(a2)),
                "a2_imag": float(np.imag(a2)),
                "closed_term": closed_term,
                "first_order_sq": first_order_sq,
                "second_order_cross": second_order_cross,
                "perturbative_open": closed_term + first_order_sq + second_order_cross,
                "dropped_order1_cross": dropped_order1_cross,
            }
        )

    closed_term = float(sum(row["closed_term"] for row in pair_rows))
    first_order_sq = float(sum(row["first_order_sq"] for row in pair_rows))
    second_order_cross = float(sum(row["second_order_cross"] for row in pair_rows))
    perturbative_open = closed_term + first_order_sq + second_order_cross
    pair_open_values = [row["perturbative_open"] for row in pair_rows]
    summary = {
        "closed_term": closed_term,
        "first_order_sq": first_order_sq,
        "second_order_cross": second_order_cross,
        "perturbative_open": perturbative_open,
        "correction": first_order_sq + second_order_cross,
        "excess_over_1": perturbative_open - 1.0,
        "max_pair_open": float(max(pair_open_values)),
        "min_pair_open": float(min(pair_open_values)),
    }
    return summary, pair_rows


def run_perturbative_experiment(
    args,
    initial_parameters=None,
    run_label=None,
    output_root=OUTPUT_DIR,
    experiment_dir=None,
    generated_at=None,
    extra_configuration=(),
    print_report=True,
):
    phi_s = 0.0
    system = spin_boson_control_system(n_levels=N_LEVELS, phi_s=phi_s)
    noise_specs = spin_boson_noise_term_specs(n_levels=N_LEVELS, phi_s=phi_s)
    noisy_system = spin_boson_noisy_control_system(n_levels=N_LEVELS, phi_s=phi_s)


    initial_pulse = _customized_initial_pulse(
        n_steps=args.n_steps,
        alpha1_khz_bounds=DEFAULT_ALPHA1_KHZ_BOUNDS,
        alpha2_khz_bounds=DEFAULT_ALPHA2_KHZ_BOUNDS,
        alpha1_cycles=args.alpha1_cycles,
    )
    parameterization = Alpha2EndpointZeroParameterization(
        spin_boson_parameterization(
            initial_pulse.n_steps,
            alpha1_khz_bounds=DEFAULT_ALPHA1_KHZ_BOUNDS,
            alpha2_khz_bounds=DEFAULT_ALPHA2_KHZ_BOUNDS,
        )
    )
    custom_initial_metadata = None
    if initial_parameters is None and getattr(args, "initial_pulse_npz", None) is not None:
        initial_parameters, custom_initial_metadata = load_custom_initial_parameters(
            args.initial_pulse_npz,
            initial_pulse,
            parameterization,
        )
        for warning in custom_initial_metadata["warnings"]:
            print(warning, file=sys.stderr, flush=True)
    target_gate = ms_xx_pi_over_2_gate()
    state_pairs = motion_resolved_gate_state_pairs(target_gate, N_LEVELS)

    step_builder = PerturbativeStepBuilder()
    expansion_objective = ExpansionFidelity(max_order=2, drop_odd_average=True)
    optimization_problem = ExpansionStateAverageFidelity(
        system=noisy_system,
        pulse=initial_pulse,
        evolution=PerturbativeExpansionEvolution(step_builder, max_order=2),
        objective=expansion_objective,
        differentiator=PerturbativeExpansionDifferentiator(
            step_builder,
            expansion_objective,
        ),
        state_pairs=state_pairs,
        normalize_weights=False,
        n_workers=args.workers,
    )
    parameterized_problem = ParameterizedControlProblem(
        optimization_problem,
        parameterization,
    )
    penalty = ParameterSmoothPenalty(
        l1_weight=args.l1_smooth_weight,
        l2_weight=args.l2_smooth_weight,
    )
    penalized_problem = PenalizedParameterizedProblem(parameterized_problem, penalty)
    initial_parameters = (
        penalized_problem.initial_parameters()
        if initial_parameters is None
        else np.asarray(initial_parameters, dtype=float).reshape(
            penalized_problem.parameter_shape
        )
    )
    initial_objective = penalized_problem.value(initial_parameters)
    initial_l1_penalty = penalty.l1_value(
        initial_parameters,
        penalized_problem.parameter_shape,
    )
    initial_l2_penalty = penalty.l2_value(
        initial_parameters,
        penalized_problem.parameter_shape,
    )
    masked_initial_pulse = penalized_problem.pulse_from_parameters(initial_parameters)
    generated_at = generated_at or datetime.now()
    experiment_dir = Path(experiment_dir) if experiment_dir is not None else timestamped_experiment_dir(
        output_root,
        run_label or "spin_boson_perturbative",
        generated_at,
    )
    experiment_dir.mkdir(parents=True, exist_ok=True)

    optimizer_options = {"maxiter": args.maxiter, "gtol": 1e-12, "ftol": 1e-15}
    optimizer = ScipyOptimizer(
        method="L-BFGS-B",
        maximize=True,
        options=optimizer_options,
    )
    progress = (
        None
        if args.no_progress or args.print_step
        else OptimizationProgressBar(args.maxiter)
    )
    step_log_path = experiment_dir / "step_log.csv"
    fidelity_terms_path = experiment_dir / "fidelity_terms.csv"
    fidelity_pair_terms_path = experiment_dir / "fidelity_terms_by_pair.csv"
    latest_pulse_stem = experiment_dir / "latest_pulse"
    latest_parameters_path = experiment_dir / "latest_parameters.npz"
    pulse_path = experiment_dir / "spin_boson_perturbative_pulses.png"
    propagation_path = experiment_dir / "spin_boson_perturbative_state_propagation.png"
    initial_pulse_stem = experiment_dir / "initial_pulse"
    final_pulse_stem = experiment_dir / "final_pulse"
    report_path = experiment_dir / "report.md"
    step_log = StepLog(step_log_path, print_steps=args.print_step)
    save_fidelity_terms = getattr(
        args,
        "save_fidelity_terms",
        getattr(args, "print_fidelity_terms", False),
    )
    fidelity_terms_log = (
        FidelityTermsLog(
            fidelity_terms_path,
            fidelity_pair_terms_path,
            print_steps=getattr(args, "print_fidelity_terms", False),
        )
        if save_fidelity_terms
        else None
    )
    step_counter = {"value": 0}
    latest_state = {
        "step": 0,
        "parameters": np.asarray(initial_parameters, dtype=float),
        "pulse": masked_initial_pulse,
    }
    initial_preview_metrics = [
        ("initial_penalized_objective", initial_objective),
        ("initial_raw_fidelity", penalized_problem.raw_value(initial_parameters)),
        (
            "initial_close_gate_fidelity",
            closed_gate_fidelity(system, masked_initial_pulse, target_gate, N_LEVELS),
        ),
        (
            "initial_open_gate_fidelity",
            open_gate_fidelity(
                noisy_system,
                masked_initial_pulse,
                target_gate,
                N_LEVELS,
                n_workers=args.workers,
            ),
        ),
        ("initial_l1_penalty", initial_l1_penalty),
        ("initial_l2_penalty", initial_l2_penalty),
    ]
    kappa_metrics = calculate_kappa_metrics(system, noisy_system, masked_initial_pulse, parameterization)
    output_manifest = [
        ("pulse_plot", pulse_path.name),
        ("propagation_plot", propagation_path.name),
        ("step_log", step_log_path.name),
        ("fidelity_terms", fidelity_terms_path.name if save_fidelity_terms else "disabled"),
        (
            "fidelity_terms_by_pair",
            fidelity_pair_terms_path.name if save_fidelity_terms else "disabled",
        ),
        ("latest_pulse_npz", latest_pulse_stem.with_suffix(".npz").name),
        ("latest_pulse_csv", latest_pulse_stem.with_suffix(".csv").name),
        ("latest_parameters", latest_parameters_path.name),
        ("initial_pulse_npz", initial_pulse_stem.with_suffix(".npz").name),
        ("initial_pulse_csv", initial_pulse_stem.with_suffix(".csv").name),
        ("final_pulse_npz", final_pulse_stem.with_suffix(".npz").name),
        ("final_pulse_csv", final_pulse_stem.with_suffix(".csv").name),
    ]
    write_optimization_preview_report(
        report_path,
        generated_at=generated_at,
        experiment_dir=experiment_dir,
        args=args,
        initial_pulse=initial_pulse,
        parameterization=parameterization,
        noisy_system=noisy_system,
        noise_specs=noise_specs,
        phi_s=phi_s,
        state_pairs=state_pairs,
        optimizer_options=optimizer_options,
        output_manifest=output_manifest,
        initial_metrics=initial_preview_metrics,
        kappa_metrics=kappa_metrics,
        custom_initial_metadata=custom_initial_metadata,
        extra_configuration=extra_configuration,
        save_fidelity_terms=save_fidelity_terms,
    )
    if print_report or args.print_step or getattr(args, "print_fidelity_terms", False):
        print_optimization_preview(report_path, args, initial_preview_metrics, kappa_metrics)

    def record_step(step, parameters):
        pulse = penalized_problem.pulse_from_parameters(parameters)
        l1_penalty = penalty.l1_value(parameters, penalized_problem.parameter_shape)
        l2_penalty = penalty.l2_value(parameters, penalized_problem.parameter_shape)
        step_log.append(
            step=step,
            close_fidelity=closed_gate_fidelity(system, pulse, target_gate, N_LEVELS),
            open_fidelity=open_gate_fidelity(
                noisy_system,
                pulse,
                target_gate,
                N_LEVELS,
                n_workers=args.workers,
            ),
            cost_function=penalized_problem.value(parameters),
            raw_fidelity=penalized_problem.raw_value(parameters),
            l1_penalty=l1_penalty,
            l2_penalty=l2_penalty,
            gradient_norm=np.linalg.norm(penalized_problem.gradient(parameters)),
        )
        if fidelity_terms_log is not None:
            fidelity_summary, fidelity_pair_rows = perturbative_fidelity_terms(
                noisy_system,
                pulse,
                target_gate,
                N_LEVELS,
            )
            fidelity_summary["step"] = int(step)
            for row in fidelity_pair_rows:
                row["step"] = int(step)
            fidelity_terms_log.append(fidelity_summary, fidelity_pair_rows)
        latest_pulse_npz_path, latest_pulse_csv_path = export_pulse_controls(
            pulse,
            latest_pulse_stem,
            RAD_S_PER_KHZ,
        )
        np.savez(
            latest_parameters_path,
            parameters=np.asarray(parameters, dtype=float),
            step=int(step),
            pulse_npz=str(latest_pulse_npz_path),
            pulse_csv=str(latest_pulse_csv_path),
        )
        latest_state["step"] = int(step)
        latest_state["parameters"] = np.asarray(parameters, dtype=float).reshape(
            penalized_problem.parameter_shape
        )
        latest_state["pulse"] = pulse

    def step_callback(parameters):
        step_counter["value"] += 1
        record_step(step_counter["value"], parameters)

    callback = CombinedCallback(progress, step_callback)
    record_step(0, initial_parameters)
    if progress is not None:
        progress.start()
    interrupted = False
    try:
        try:
            result = optimizer.optimize_parameters(
                penalized_problem,
                initial_parameters=initial_parameters,
                callback=callback,
            )
            final_pulse = result.optimized_pulse
            final_parameters = result.x.reshape(penalized_problem.parameter_shape)
        except KeyboardInterrupt:
            interrupted = True
            final_parameters = np.asarray(latest_state["parameters"], dtype=float).reshape(
                penalized_problem.parameter_shape
            )
            final_pulse = latest_state["pulse"]
            result = SimpleNamespace(
                success=False,
                message=(
                    "INTERRUPTED: optimization stopped by user; "
                    "using latest accepted pulse for report."
                ),
                nit=int(latest_state["step"]),
                nfev="NA",
                x=final_parameters.reshape(-1),
                optimized_pulse=final_pulse,
            )
            print(
                "\ninterrupted=True; using latest accepted pulse for final report.",
                file=sys.stderr,
                flush=True,
            )
        final_objective = penalized_problem.value(final_parameters)
    finally:
        optimization_problem.shutdown()
    if progress is not None:
        progress.finish(result)
    final_l1_penalty = penalty.l1_value(
        final_parameters,
        penalized_problem.parameter_shape,
    )
    final_l2_penalty = penalty.l2_value(
        final_parameters,
        penalized_problem.parameter_shape,
    )

    if not np.allclose(masked_initial_pulse.amplitudes[[0, -1], 1], 0.0):
        raise RuntimeError("Initial alpha2 endpoints are not masked to zero.")
    if not np.allclose(final_pulse.amplitudes[[0, -1], 1], 0.0):
        raise RuntimeError("Optimized alpha2 endpoints are not masked to zero.")

    initial_close_gate_fidelity = closed_gate_fidelity(
        system,
        masked_initial_pulse,
        target_gate,
        N_LEVELS,
    )
    final_close_gate_fidelity = closed_gate_fidelity(
        system,
        final_pulse,
        target_gate,
        N_LEVELS,
    )
    initial_open_gate_fidelity = open_gate_fidelity(
        noisy_system,
        masked_initial_pulse,
        target_gate,
        N_LEVELS,
        n_workers=args.workers,
    )
    final_open_gate_fidelity = open_gate_fidelity(
        noisy_system,
        final_pulse,
        target_gate,
        N_LEVELS,
        n_workers=args.workers,
    )

    lower, upper = parameterization.base._bounds_for(final_pulse.amplitudes.shape)
    bound_tolerance = 1e-8
    if np.any(final_pulse.amplitudes < lower - bound_tolerance) or np.any(
        final_pulse.amplitudes > upper + bound_tolerance
    ):
        raise RuntimeError("Optimized pulse violates amplitude bounds.")

    dimension = 4 * N_LEVELS
    context = EvolutionContext(
        initial_state=basis_state(0, dimension),
        target_state=ms_bell_target_motion_ground(N_LEVELS),
    )
    nominal_evolution = NominalUnitaryEvolution(UnitaryStepBuilder())
    initial_single_state_fidelity = StateTransferFidelity(context.target_state).evaluate(
        nominal_evolution.evolve(system, masked_initial_pulse, context)
    )
    final_single_state_fidelity = StateTransferFidelity(context.target_state).evaluate(
        nominal_evolution.evolve(system, final_pulse, context)
    )
    initial_states = propagate_states(nominal_evolution, system, masked_initial_pulse, context)
    final_states = propagate_states(nominal_evolution, system, final_pulse, context)
    if not np.allclose(np.sum(np.abs(initial_states) ** 2, axis=1), 1.0, atol=1e-8):
        raise RuntimeError("Initial propagation populations are not normalized.")
    if not np.allclose(np.sum(np.abs(final_states) ** 2, axis=1), 1.0, atol=1e-8):
        raise RuntimeError("Final propagation populations are not normalized.")

    time_us = (np.arange(initial_pulse.n_steps) + 0.5) * initial_pulse.dt * 1e6
    time_edges_us = np.arange(initial_pulse.n_steps + 1) * initial_pulse.dt * 1e6
    initial_pulse_npz_path, initial_pulse_csv_path = export_pulse_controls(
        masked_initial_pulse,
        initial_pulse_stem,
        RAD_S_PER_KHZ,
    )
    final_pulse_npz_path, final_pulse_csv_path = export_pulse_controls(
        final_pulse,
        final_pulse_stem,
        RAD_S_PER_KHZ,
    )

    metrics = {
        "initial_fidelity": initial_single_state_fidelity,
        "final_fidelity": final_single_state_fidelity,
        "initial_close_gate_fidelity": initial_close_gate_fidelity,
        "final_close_gate_fidelity": final_close_gate_fidelity,
        "initial_open_gate_fidelity": initial_open_gate_fidelity,
        "final_open_gate_fidelity": final_open_gate_fidelity,
        "initial_l1_penalty": initial_l1_penalty,
        "final_l1_penalty": final_l1_penalty,
        "initial_l2_penalty": initial_l2_penalty,
        "final_l2_penalty": final_l2_penalty,
        "initial_penalized_objective": initial_objective,
        "final_penalized_objective": final_objective,
    }
    experiment_note = format_experiment_note(args, result, metrics)
    plot_pulses(time_us, masked_initial_pulse, final_pulse, pulse_path, note=experiment_note)
    plot_population_marginals(
        time_edges_us,
        initial_states,
        final_states,
        N_LEVELS,
        propagation_path,
        note=experiment_note,
    )

    for path in (
        pulse_path,
        propagation_path,
        step_log_path,
        latest_pulse_stem.with_suffix(".npz"),
        latest_pulse_stem.with_suffix(".csv"),
        latest_parameters_path,
        initial_pulse_npz_path,
        initial_pulse_csv_path,
        final_pulse_npz_path,
        final_pulse_csv_path,
    ):
        if not path.exists() or path.stat().st_size == 0:
            raise RuntimeError(f"Expected non-empty output at {path}.")
    if fidelity_terms_log is not None:
        for path in (fidelity_terms_path, fidelity_pair_terms_path):
            if not path.exists() or path.stat().st_size == 0:
                raise RuntimeError(f"Expected non-empty fidelity diagnostics at {path}.")

    outputs = {
        "pulse_plot": pulse_path,
        "propagation_plot": propagation_path,
        "step_log": step_log_path,
        "fidelity_terms": fidelity_terms_path if save_fidelity_terms else None,
        "fidelity_terms_by_pair": fidelity_pair_terms_path if save_fidelity_terms else None,
        "latest_pulse_npz": latest_pulse_stem.with_suffix(".npz"),
        "latest_pulse_csv": latest_pulse_stem.with_suffix(".csv"),
        "latest_parameters": latest_parameters_path,
        "initial_pulse_npz": initial_pulse_npz_path,
        "initial_pulse_csv": initial_pulse_csv_path,
        "final_pulse_npz": final_pulse_npz_path,
        "final_pulse_csv": final_pulse_csv_path,
    }
    append_optimization_results_report(
        report_path,
        metrics=metrics,
        result=result,
        figures=[
            ("Pulse parameters", pulse_path),
            ("State propagation", propagation_path),
        ],
        outputs=outputs,
        interrupted=interrupted,
    )
    if print_report:
        print_experiment_report(args, result, metrics, outputs)
        print(f"experiment_dir={experiment_dir}")
        print(f"step_log={step_log_path}")
        if save_fidelity_terms:
            print(f"fidelity_terms={fidelity_terms_path}")
            print(f"fidelity_terms_by_pair={fidelity_pair_terms_path}")
        print(f"latest_pulse_npz={latest_pulse_stem.with_suffix('.npz')}")
        print(f"latest_pulse_csv={latest_pulse_stem.with_suffix('.csv')}")
        print(f"latest_parameters={latest_parameters_path}")
        print(f"initial_pulse_npz={initial_pulse_npz_path}")
        print(f"initial_pulse_csv={initial_pulse_csv_path}")
        print(f"final_pulse_npz={final_pulse_npz_path}")
        print(f"final_pulse_csv={final_pulse_csv_path}")
        print(f"markdown_report={report_path}")

    return {
        "args": args,
        "result": result,
        "metrics": metrics,
        "outputs": outputs,
        "experiment_dir": experiment_dir,
        "report_path": report_path,
        "initial_parameters": initial_parameters,
        "final_parameters": final_parameters,
        "custom_initial_metadata": custom_initial_metadata,
        "interrupted": interrupted,
    }


def custom_initial_configuration(metadata):
    if metadata is None:
        return ()
    return (
        ("initial_pulse_source", "custom_npz"),
        ("source_npz", metadata["source_npz"]),
        ("source_dt", metadata["source_dt"]),
        ("experiment_dt", metadata["experiment_dt"]),
        ("dt_missing", metadata["dt_missing"]),
        ("dt_mismatch", metadata["dt_mismatch"]),
    )


def main():
    run_perturbative_experiment(parse_args())


if __name__ == "__main__":
    main()
