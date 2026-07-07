from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
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

from experiments.reporting import (
    StepLog,
    export_pulse_controls,
    timestamped_experiment_dir,
    write_experiment_report,
)
from physical_systems.spin_boson import (
    DEFAULT_LAMB_DICKE_ETA,
    annihilation_operator,
    creation_operator,
    motion_resolved_gate_state_pairs,
    number_operator,
    spin_boson_control_system,
    spin_boson_initial_pulse,
    spin_boson_parameterization,
    two_qubit_spin_phase_mode,
)
from quantum_control import (
    ControlProblem,
    EvolutionContext,
    GrapeDifferentiator,
    NominalUnitaryEvolution,
    ParameterSmoothPenalty,
    ParameterizedControlProblem,
    PenalizedParameterizedProblem,
    StateTransferFidelity,
    UnitaryStepBuilder,
    closed_gate_fidelity,
    ms_xx_pi_over_2_gate,
    noisy_gate_fidelity,
)
from quantum_control.optimizers import ScipyOptimizer


N_LEVELS = 6
MAXITER = 50
RAD_S_PER_KHZ = 2.0 * np.pi * 1000.0
DEFAULT_L1_SMOOTH_WEIGHT = 0.0
DEFAULT_L2_SMOOTH_WEIGHT = 1e-4


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
    x1 = 0.5 * (annihilation_operator(n_levels) + creation_operator(n_levels))
    s_phi = two_qubit_spin_phase_mode(phi_s, (0.5, -0.5))

    return spin_boson_control_system(
        n_levels=n_levels,
        phi_s=phi_s,
        static_fluctuations=[
            314.159 * np.kron(0.5 * sz1_plus_sz2, motion_identity),
            1256.637 * np.kron(spin_identity, number),
        ],
        control_fluctuations=[
            0.001 * np.kron(spin_identity, number),
            0.005 * DEFAULT_LAMB_DICKE_ETA * np.kron(s_phi, x1),
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
    parser = argparse.ArgumentParser(description="Run spin-boson L-BFGS-B experiment.")
    parser.add_argument("--maxiter", type=int, default=MAXITER)
    parser.add_argument("--alpha1-cycles", type=float, default=1.0)
    parser.add_argument(
        "--print-step",
        action="store_true",
        help="Print per-step close fidelity, open fidelity, and cost function.",
    )
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
    return parser.parse_args()


def propagate_states(evolution, system, pulse, context):
    result = evolution.evolve(system, pulse, context)
    states = [context.initial_state]
    state = context.initial_state
    for step in result.W_steps:
        state = step @ state
        states.append(state)
    return np.asarray(states)


def plot_pulses(time_us, initial_pulse, final_pulse, output_path):
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

    fig.suptitle("Initial pulse parameter and final pulse parameter")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_population_marginals(time_edges_us, initial_states, final_states, output_path):
    initial_populations = np.abs(initial_states) ** 2
    final_populations = np.abs(final_states) ** 2
    initial_joint = initial_populations.reshape((-1, 4, N_LEVELS))
    final_joint = final_populations.reshape((-1, 4, N_LEVELS))
    initial_spin = initial_joint.sum(axis=2)
    final_spin = final_joint.sum(axis=2)
    initial_motion = initial_joint.sum(axis=1)
    final_motion = final_joint.sum(axis=1)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey="row")
    spin_labels = ["00", "01", "10", "11"]
    for spin_index, label in enumerate(spin_labels):
        axes[0, 0].plot(time_edges_us, initial_spin[:, spin_index], label=f"|{label}>")
        axes[0, 1].plot(time_edges_us, final_spin[:, spin_index], label=f"|{label}>")
    for level in range(N_LEVELS):
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
    fig.suptitle("State propagation: two-qubit spin and motion marginals")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    phi_s = 0.0
    system = spin_boson_control_system(n_levels=N_LEVELS, phi_s=phi_s)
    noisy_system = spin_boson_noisy_control_system(n_levels=N_LEVELS, phi_s=phi_s)
    initial_pulse = spin_boson_initial_pulse(alpha1_cycles=args.alpha1_cycles)
    parameterization = Alpha2EndpointZeroParameterization(
        spin_boson_parameterization(initial_pulse.n_steps)
    )
    dimension = 4 * N_LEVELS
    context = EvolutionContext(
        initial_state=basis_state(0, dimension),
        target_state=ms_bell_target_motion_ground(N_LEVELS),
    )
    step_builder = UnitaryStepBuilder()
    evolution = NominalUnitaryEvolution(step_builder)
    objective = StateTransferFidelity(context.target_state)
    differentiator = GrapeDifferentiator(step_builder)
    problem = ControlProblem(
        system=system,
        pulse=initial_pulse,
        context=context,
        evolution=evolution,
        objective=objective,
        differentiator=differentiator,
    )

    parameterized_problem = ParameterizedControlProblem(problem, parameterization)
    penalty = ParameterSmoothPenalty(
        l1_weight=args.l1_smooth_weight,
        l2_weight=args.l2_smooth_weight,
    )
    penalized_problem = PenalizedParameterizedProblem(parameterized_problem, penalty)
    initial_parameters = penalized_problem.initial_parameters()
    initial_fidelity = problem.value()
    initial_l1_penalty = penalty.l1_value(
        initial_parameters,
        penalized_problem.parameter_shape,
    )
    initial_l2_penalty = penalty.l2_value(
        initial_parameters,
        penalized_problem.parameter_shape,
    )
    initial_objective = penalized_problem.value(initial_parameters)
    masked_initial_pulse = penalized_problem.pulse_from_parameters(initial_parameters)
    generated_at = datetime.now()
    experiment_dir = timestamped_experiment_dir(
        OUTPUT_DIR,
        "spin_boson_lbfgsb",
        generated_at,
    )
    experiment_dir.mkdir(parents=True, exist_ok=True)
    optimizer_options = {"maxiter": args.maxiter, "gtol": 1e-12, "ftol": 1e-15}
    optimizer = ScipyOptimizer(
        method="L-BFGS-B",
        maximize=True,
        options=optimizer_options,
    )
    target_gate = ms_xx_pi_over_2_gate()
    step_log_path = experiment_dir / "step_log.csv"
    step_log = StepLog(step_log_path, print_steps=args.print_step)

    step_counter = {"value": 0}

    def record_step(step, parameters):
        pulse = penalized_problem.pulse_from_parameters(parameters)
        l1_penalty = penalty.l1_value(parameters, penalized_problem.parameter_shape)
        l2_penalty = penalty.l2_value(parameters, penalized_problem.parameter_shape)
        step_log.append(
            step=step,
            close_fidelity=closed_gate_fidelity(system, pulse, motion_resolved_gate_state_pairs(target_gate, N_LEVELS)),
            open_fidelity=noisy_gate_fidelity(noisy_system, pulse, motion_resolved_gate_state_pairs(target_gate, N_LEVELS)),
            cost_function=penalized_problem.value(parameters),
            raw_fidelity=penalized_problem.raw_value(parameters),
            l1_penalty=l1_penalty,
            l2_penalty=l2_penalty,
            gradient_norm=np.linalg.norm(penalized_problem.gradient(parameters)),
        )

    def step_callback(parameters):
        step_counter["value"] += 1
        record_step(step_counter["value"], parameters)

    record_step(0, initial_parameters)
    result = optimizer.optimize_parameters(
        penalized_problem,
        callback=step_callback,
    )
    final_pulse = result.optimized_pulse
    final_parameters = result.x.reshape(penalized_problem.parameter_shape)
    final_fidelity = problem.value(final_pulse)
    final_l1_penalty = penalty.l1_value(
        final_parameters,
        penalized_problem.parameter_shape,
    )
    final_l2_penalty = penalty.l2_value(
        final_parameters,
        penalized_problem.parameter_shape,
    )
    final_objective = penalized_problem.value(final_parameters)

    if not np.allclose(masked_initial_pulse.amplitudes[[0, -1], 1], 0.0):
        raise RuntimeError("Initial alpha2 endpoints are not masked to zero.")
    if not np.allclose(final_pulse.amplitudes[[0, -1], 1], 0.0):
        raise RuntimeError("Optimized alpha2 endpoints are not masked to zero.")

    initial_close_gate_fidelity = closed_gate_fidelity(
        system,
        masked_initial_pulse,
        motion_resolved_gate_state_pairs(target_gate, N_LEVELS),
    )
    final_close_gate_fidelity = closed_gate_fidelity(
        system,
        final_pulse,
        motion_resolved_gate_state_pairs(target_gate, N_LEVELS),
    )
    initial_open_gate_fidelity = noisy_gate_fidelity(
        noisy_system,
        masked_initial_pulse,
        motion_resolved_gate_state_pairs(target_gate, N_LEVELS),
    )
    final_open_gate_fidelity = noisy_gate_fidelity(
        noisy_system,
        final_pulse,
        motion_resolved_gate_state_pairs(target_gate, N_LEVELS),
    )

    lower, upper = parameterization.base._bounds_for(final_pulse.amplitudes.shape)
    if np.any(final_pulse.amplitudes < lower) or np.any(final_pulse.amplitudes > upper):
        raise RuntimeError("Optimized pulse violates amplitude bounds.")

    initial_states = propagate_states(evolution, system, initial_pulse, context)
    final_states = propagate_states(evolution, system, final_pulse, context)
    if not np.allclose(np.sum(np.abs(initial_states) ** 2, axis=1), 1.0, atol=1e-8):
        raise RuntimeError("Initial propagation populations are not normalized.")
    if not np.allclose(np.sum(np.abs(final_states) ** 2, axis=1), 1.0, atol=1e-8):
        raise RuntimeError("Final propagation populations are not normalized.")

    time_us = (np.arange(initial_pulse.n_steps) + 0.5) * initial_pulse.dt * 1e6
    time_edges_us = np.arange(initial_pulse.n_steps + 1) * initial_pulse.dt * 1e6
    pulse_path = experiment_dir / "spin_boson_pulses.png"
    propagation_path = experiment_dir / "spin_boson_state_propagation.png"
    initial_pulse_npz_path, initial_pulse_csv_path = export_pulse_controls(
        masked_initial_pulse,
        experiment_dir / "initial_pulse",
        RAD_S_PER_KHZ,
    )
    final_pulse_npz_path, final_pulse_csv_path = export_pulse_controls(
        final_pulse,
        experiment_dir / "final_pulse",
        RAD_S_PER_KHZ,
    )
    plot_pulses(time_us, initial_pulse, final_pulse, pulse_path)
    plot_population_marginals(time_edges_us, initial_states, final_states, propagation_path)

    for path in (
        pulse_path,
        propagation_path,
        step_log_path,
        initial_pulse_npz_path,
        initial_pulse_csv_path,
        final_pulse_npz_path,
        final_pulse_csv_path,
    ):
        if not path.exists() or path.stat().st_size == 0:
            raise RuntimeError(f"Expected non-empty output at {path}.")

    metrics = {
        "fidelity": (initial_fidelity, final_fidelity),
        "close_gate_fidelity": (initial_close_gate_fidelity, final_close_gate_fidelity),
        "open_gate_fidelity": (initial_open_gate_fidelity, final_open_gate_fidelity),
        "l1_penalty": (initial_l1_penalty, final_l1_penalty),
        "l2_penalty": (initial_l2_penalty, final_l2_penalty),
        "penalized_objective": (initial_objective, final_objective),
    }
    bounds_lower_khz = lower[0] / RAD_S_PER_KHZ
    bounds_upper_khz = upper[0] / RAD_S_PER_KHZ
    report_path = write_experiment_report(
        output_dir=OUTPUT_DIR,
        experiment_slug="spin_boson_lbfgsb",
        title="Spin-Boson L-BFGS-B Optimization",
        configuration=[
            ("objective", "state_transfer_fidelity"),
            ("target_state", "(|00,0>-i|11,0>)/sqrt(2)"),
            ("target_gate", "MS_XX(pi/2)"),
            ("n_levels", N_LEVELS),
            ("n_steps", initial_pulse.n_steps),
            ("dt_s", initial_pulse.dt),
            ("total_time_us", initial_pulse.n_steps * initial_pulse.dt * 1e6),
            ("phi_s", phi_s),
            ("alpha1_cycles", args.alpha1_cycles),
            ("alpha1_bounds_khz", f"{bounds_lower_khz[0]:.12g} to {bounds_upper_khz[0]:.12g}"),
            ("alpha2_bounds_khz", f"{bounds_lower_khz[1]:.12g} to {bounds_upper_khz[1]:.12g}"),
            ("alpha2_endpoint_constraint", "initial and final alpha2 fixed to 0"),
            ("static_fluctuation_count", len(noisy_system.static_fluctuations)),
            ("control_fluctuation_count", len(noisy_system.control_fluctuations)),
            ("l1_smooth_weight", args.l1_smooth_weight),
            ("l2_smooth_weight", args.l2_smooth_weight),
            ("print_step", args.print_step),
            ("step_log", step_log_path.name),
            ("initial_pulse_npz", initial_pulse_npz_path.name),
            ("initial_pulse_csv", initial_pulse_csv_path.name),
            ("final_pulse_npz", final_pulse_npz_path.name),
            ("final_pulse_csv", final_pulse_csv_path.name),
            ("optimizer_method", "L-BFGS-B"),
            ("optimizer_maximize", True),
            ("optimizer_options", optimizer_options),
        ],
        results=[
            ("fidelity", *metrics["fidelity"]),
            ("close_gate_fidelity", *metrics["close_gate_fidelity"]),
            ("open_gate_fidelity", *metrics["open_gate_fidelity"]),
            ("l1_penalty", *metrics["l1_penalty"]),
            ("l2_penalty", *metrics["l2_penalty"]),
            ("penalized_objective", *metrics["penalized_objective"]),
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
        generated_at=generated_at,
    )

    print(f"initial_fidelity={metrics['fidelity'][0]:.12g}")
    print(f"final_fidelity={metrics['fidelity'][1]:.12g}")
    print(f"initial_close_gate_fidelity={metrics['close_gate_fidelity'][0]:.12g}")
    print(f"final_close_gate_fidelity={metrics['close_gate_fidelity'][1]:.12g}")
    print(f"initial_open_gate_fidelity={metrics['open_gate_fidelity'][0]:.12g}")
    print(f"final_open_gate_fidelity={metrics['open_gate_fidelity'][1]:.12g}")
    print(f"initial_l1_penalty={metrics['l1_penalty'][0]:.12g}")
    print(f"final_l1_penalty={metrics['l1_penalty'][1]:.12g}")
    print(f"initial_l2_penalty={metrics['l2_penalty'][0]:.12g}")
    print(f"final_l2_penalty={metrics['l2_penalty'][1]:.12g}")
    print(f"initial_penalized_objective={metrics['penalized_objective'][0]:.12g}")
    print(f"final_penalized_objective={metrics['penalized_objective'][1]:.12g}")
    print(f"l1_smooth_weight={args.l1_smooth_weight:.12g}")
    print(f"l2_smooth_weight={args.l2_smooth_weight:.12g}")
    print(f"alpha1_cycles={args.alpha1_cycles:.12g}")
    print("target_state=(|00,0>-i|11,0>)/sqrt(2)")
    print(f"optimizer_success={result.success}")
    print(f"optimizer_message={result.message}")
    print(f"optimizer_nit={getattr(result, 'nit', 'NA')}")
    print(f"optimizer_nfev={getattr(result, 'nfev', 'NA')}")
    print(f"experiment_dir={experiment_dir}")
    print(f"step_log={step_log_path}")
    print(f"initial_pulse_npz={initial_pulse_npz_path}")
    print(f"initial_pulse_csv={initial_pulse_csv_path}")
    print(f"final_pulse_npz={final_pulse_npz_path}")
    print(f"final_pulse_csv={final_pulse_csv_path}")
    print(f"pulse_plot={pulse_path}")
    print(f"propagation_plot={propagation_path}")
    print(f"markdown_report={report_path}")


if __name__ == "__main__":
    main()
