from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT_DIR / ".matplotlib"))
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from experiments.reporting import write_experiment_report
from quantum_control import (
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
    two_qubit_spin_phase_difference,
)
from quantum_control.optimizers import ScipyOptimizer


N_LEVELS = 6
N_STEPS = 200
MAXITER = 20
RAD_S_PER_KHZ = 2.0 * np.pi * 1000.0
DEFAULT_L1_SMOOTH_WEIGHT = 0.1
DEFAULT_L2_SMOOTH_WEIGHT = 1


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
    spin_identity = np.eye(4, dtype=complex)
    motion_identity = np.eye(n_levels, dtype=complex)
    single_identity = np.eye(2, dtype=complex)
    sz = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    sz1_plus_sz2 = np.kron(sz, single_identity) + np.kron(single_identity, sz)
    number = number_operator(n_levels)
    displacement = annihilation_operator(n_levels) + creation_operator(n_levels)
    s_phi = two_qubit_spin_phase_difference(phi_s)

    return spin_boson_control_system(
        n_levels=n_levels,
        phi_s=phi_s,
        static_fluctuations=[
            314.159 * np.kron(0.5 * sz1_plus_sz2, motion_identity),
            1256.637 * np.kron(spin_identity, number),
        ],
        control_fluctuations=[
            0.001 * np.kron(spin_identity, number),
            0.005 * 0.5 * np.kron(s_phi, displacement),
        ],
    )


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
        "--no-progress",
        action="store_true",
        help="Disable the optimization progress bar.",
    )
    return parser.parse_args()


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


def print_experiment_report(args, result, metrics, pulse_path, propagation_path):
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
            ("pulse_plot", pulse_path),
            ("propagation_plot", propagation_path),
        ],
    )


def _transition(initial, final):
    delta = final - initial
    return f"{initial:.12g} -> {final:.12g} (delta {delta:+.3g})"


def main():
    args = parse_args()
    phi_s = 0.0
    system = spin_boson_control_system(n_levels=N_LEVELS, phi_s=phi_s)
    noisy_system = spin_boson_noisy_control_system(n_levels=N_LEVELS, phi_s=phi_s)
    initial_pulse = spin_boson_initial_pulse(
        n_steps=args.n_steps,
        alpha1_cycles=args.alpha1_cycles,
    )
    parameterization = Alpha2EndpointZeroParameterization(
        spin_boson_parameterization(initial_pulse.n_steps)
    )
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
    initial_parameters = penalized_problem.initial_parameters()
    initial_objective = penalized_problem.value(initial_parameters)
    initial_l1_penalty = penalty.l1_value(
        initial_parameters,
        penalized_problem.parameter_shape,
    )
    initial_l2_penalty = penalty.l2_value(
        initial_parameters,
        penalized_problem.parameter_shape,
    )

    optimizer_options = {"maxiter": args.maxiter, "gtol": 1e-12, "ftol": 1e-15}
    optimizer = ScipyOptimizer(
        method="L-BFGS-B",
        maximize=True,
        options=optimizer_options,
    )
    progress = None if args.no_progress else OptimizationProgressBar(args.maxiter)
    if progress is not None:
        progress.start()
    try:
        result = optimizer.optimize_parameters(penalized_problem, callback=progress)
        final_pulse = result.optimized_pulse
        final_parameters = result.x.reshape(penalized_problem.parameter_shape)
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

    masked_initial_pulse = penalized_problem.pulse_from_parameters(initial_parameters)
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
    pulse_path = OUTPUT_DIR / "spin_boson_perturbative_pulses.png"
    propagation_path = OUTPUT_DIR / "spin_boson_perturbative_state_propagation.png"

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

    for path in (pulse_path, propagation_path):
        if not path.exists() or path.stat().st_size == 0:
            raise RuntimeError(f"Expected non-empty plot at {path}.")

    bounds_lower_khz = lower[0] / RAD_S_PER_KHZ
    bounds_upper_khz = upper[0] / RAD_S_PER_KHZ
    report_path = write_experiment_report(
        output_dir=OUTPUT_DIR,
        experiment_slug="spin_boson_perturbative",
        title="Spin-Boson Perturbative Open-Gate Optimization",
        configuration=[
            ("objective", "open_gate_fidelity_expansion"),
            ("target_state", "(|00,0>-i|11,0>)/sqrt(2)"),
            ("target_gate", "MS_XX(pi/2)"),
            ("n_levels", N_LEVELS),
            ("n_steps", args.n_steps),
            ("dt_s", initial_pulse.dt),
            ("total_time_us", initial_pulse.n_steps * initial_pulse.dt * 1e6),
            ("phi_s", phi_s),
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
            ("state_pair_count", len(state_pairs)),
            ("l1_smooth_weight", args.l1_smooth_weight),
            ("l2_smooth_weight", args.l2_smooth_weight),
            ("optimizer_method", "L-BFGS-B"),
            ("optimizer_maximize", True),
            ("optimizer_options", optimizer_options),
        ],
        results=[
            ("single_state_fidelity", metrics["initial_fidelity"], metrics["final_fidelity"]),
            (
                "close_gate_fidelity",
                metrics["initial_close_gate_fidelity"],
                metrics["final_close_gate_fidelity"],
            ),
            (
                "open_gate_fidelity",
                metrics["initial_open_gate_fidelity"],
                metrics["final_open_gate_fidelity"],
            ),
            ("l1_penalty", metrics["initial_l1_penalty"], metrics["final_l1_penalty"]),
            ("l2_penalty", metrics["initial_l2_penalty"], metrics["final_l2_penalty"]),
            (
                "penalized_objective",
                metrics["initial_penalized_objective"],
                metrics["final_penalized_objective"],
            ),
        ],
        optimizer=[
            ("success", result.success),
            ("message", result.message),
            ("nit", getattr(result, "nit", "NA")),
            ("nfev", getattr(result, "nfev", "NA")),
        ],
        figures=[
            ("Pulse parameters", pulse_path),
            ("State propagation", propagation_path),
        ],
    )

    print_experiment_report(args, result, metrics, pulse_path, propagation_path)
    print(f"markdown_report={report_path}")


if __name__ == "__main__":
    main()
