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

from quantum_control import (
    ControlProblem,
    EvolutionContext,
    GrapeDifferentiator,
    NominalUnitaryEvolution,
    StateTransferFidelity,
    UnitaryStepBuilder,
    spin_boson_control_system,
    spin_boson_initial_pulse,
    spin_boson_parameterization,
)
from quantum_control.optimizers import ScipyOptimizer


N_LEVELS = 6
MAXITER = 50
RAD_S_PER_KHZ = 2.0 * np.pi * 1000.0
DEFAULT_SMOOTH_PENALTY_WEIGHT = 1e-4


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


class SmoothPenaltyProblem:
    def __init__(self, problem, weight):
        self.problem = problem
        self.pulse = problem.pulse
        self.weight = float(weight)

    def value(self, pulse=None):
        pulse = pulse or self.pulse
        return self.raw_fidelity(pulse) - self.weight * smooth_penalty(pulse)

    def gradient(self, pulse=None):
        pulse = pulse or self.pulse
        return self.problem.gradient(pulse) - self.weight * smooth_penalty_gradient(pulse)

    def raw_fidelity(self, pulse=None):
        return self.problem.value(pulse or self.pulse)


def smooth_penalty(pulse):
    amplitudes_khz = pulse.amplitudes / RAD_S_PER_KHZ
    second_difference = np.diff(amplitudes_khz, n=2, axis=0)
    return float(np.sum(second_difference**2))


def smooth_penalty_gradient(pulse):
    amplitudes_khz = pulse.amplitudes / RAD_S_PER_KHZ
    if amplitudes_khz.shape[0] < 3:
        return np.zeros_like(pulse.amplitudes)
    second_difference = np.diff(amplitudes_khz, n=2, axis=0)
    gradient_khz = np.zeros_like(amplitudes_khz)
    gradient_khz[:-2] += 2.0 * second_difference
    gradient_khz[1:-1] -= 4.0 * second_difference
    gradient_khz[2:] += 2.0 * second_difference
    return gradient_khz / RAD_S_PER_KHZ


def parse_args():
    parser = argparse.ArgumentParser(description="Run spin-boson L-BFGS-B experiment.")
    parser.add_argument("--maxiter", type=int, default=MAXITER)
    parser.add_argument("--alpha1-cycles", type=float, default=1.0)
    parser.add_argument(
        "--smooth-penalty-weight",
        type=float,
        default=DEFAULT_SMOOTH_PENALTY_WEIGHT,
    )
    return parser.parse_args()


def pulse_bounds(pulse, parameterization):
    lower = np.broadcast_to(parameterization.lower, pulse.amplitudes.shape)
    upper = np.broadcast_to(parameterization.upper, pulse.amplitudes.shape)
    return list(zip(lower.reshape(-1), upper.reshape(-1), strict=True))


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
    system = spin_boson_control_system(n_levels=N_LEVELS, phi_s=0.0)
    initial_pulse = spin_boson_initial_pulse(alpha1_cycles=args.alpha1_cycles)
    parameterization = spin_boson_parameterization(initial_pulse.n_steps)
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

    penalized_problem = SmoothPenaltyProblem(problem, weight=args.smooth_penalty_weight)
    initial_fidelity = problem.value()
    initial_penalty = smooth_penalty(initial_pulse)
    initial_objective = penalized_problem.value()
    optimizer = ScipyOptimizer(
        method="L-BFGS-B",
        maximize=True,
        options={"maxiter": args.maxiter, "gtol": 1e-12, "ftol": 1e-15},
    )
    result = optimizer.optimize(
        penalized_problem,
        bounds=pulse_bounds(initial_pulse, parameterization),
    )
    final_pulse = result.optimized_pulse
    final_fidelity = problem.value(final_pulse)
    final_penalty = smooth_penalty(final_pulse)
    final_objective = penalized_problem.value(final_pulse)

    lower = np.broadcast_to(parameterization.lower, final_pulse.amplitudes.shape)
    upper = np.broadcast_to(parameterization.upper, final_pulse.amplitudes.shape)
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
    pulse_path = OUTPUT_DIR / "spin_boson_pulses.png"
    propagation_path = OUTPUT_DIR / "spin_boson_state_propagation.png"
    plot_pulses(time_us, initial_pulse, final_pulse, pulse_path)
    plot_population_marginals(time_edges_us, initial_states, final_states, propagation_path)

    for path in (pulse_path, propagation_path):
        if not path.exists() or path.stat().st_size == 0:
            raise RuntimeError(f"Expected non-empty plot at {path}.")

    print(f"initial_fidelity={initial_fidelity:.12g}")
    print(f"final_fidelity={final_fidelity:.12g}")
    print(f"initial_smooth_penalty={initial_penalty:.12g}")
    print(f"final_smooth_penalty={final_penalty:.12g}")
    print(f"initial_penalized_objective={initial_objective:.12g}")
    print(f"final_penalized_objective={final_objective:.12g}")
    print(f"smooth_penalty_weight={args.smooth_penalty_weight:.12g}")
    print(f"alpha1_cycles={args.alpha1_cycles:.12g}")
    print("target_state=(|00,0>-i|11,0>)/sqrt(2)")
    print(f"optimizer_success={result.success}")
    print(f"optimizer_message={result.message}")
    print(f"optimizer_nit={getattr(result, 'nit', 'NA')}")
    print(f"optimizer_nfev={getattr(result, 'nfev', 'NA')}")
    print(f"pulse_plot={pulse_path}")
    print(f"propagation_plot={propagation_path}")


if __name__ == "__main__":
    main()
