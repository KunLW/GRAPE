from __future__ import annotations

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


def basis_state(index, dimension):
    state = np.zeros(dimension, dtype=complex)
    state[index] = 1.0
    return state


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


def plot_state_propagation(time_edges_us, initial_states, final_states, output_path):
    initial_populations = np.abs(initial_states) ** 2
    final_populations = np.abs(final_states) ** 2

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True, sharey=True)
    for basis_index in range(2 * N_LEVELS):
        spin = basis_index // N_LEVELS
        level = basis_index % N_LEVELS
        label = f"|{spin},{level}>"
        axes[0].plot(time_edges_us, initial_populations[:, basis_index], label=label)
        axes[1].plot(time_edges_us, final_populations[:, basis_index], label=label)

    axes[0].set_title("Initial pulse")
    axes[1].set_title("Optimized pulse")
    axes[1].set_xlabel("time (us)")
    for axis in axes:
        axis.set_ylabel("population")
        axis.grid(True, alpha=0.3)
        axis.set_ylim(-0.02, 1.02)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", ncol=1, fontsize="small")
    fig.suptitle("State propagation")
    fig.tight_layout(rect=(0, 0, 0.88, 1))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main():
    system = spin_boson_control_system(n_levels=N_LEVELS, phi_s=0.0)
    initial_pulse = spin_boson_initial_pulse()
    parameterization = spin_boson_parameterization(initial_pulse.n_steps)
    dimension = 2 * N_LEVELS
    context = EvolutionContext(
        initial_state=basis_state(0, dimension),
        target_state=basis_state(N_LEVELS + 1, dimension),
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

    initial_fidelity = problem.value()
    optimizer = ScipyOptimizer(
        method="L-BFGS-B",
        maximize=True,
        options={"maxiter": MAXITER, "gtol": 1e-12, "ftol": 1e-15},
    )
    result = optimizer.optimize(
        problem,
        bounds=pulse_bounds(initial_pulse, parameterization),
    )
    final_pulse = result.optimized_pulse
    final_fidelity = problem.value(final_pulse)

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
    plot_state_propagation(time_edges_us, initial_states, final_states, propagation_path)

    for path in (pulse_path, propagation_path):
        if not path.exists() or path.stat().st_size == 0:
            raise RuntimeError(f"Expected non-empty plot at {path}.")

    print(f"initial_fidelity={initial_fidelity:.12g}")
    print(f"final_fidelity={final_fidelity:.12g}")
    print(f"optimizer_success={result.success}")
    print(f"optimizer_message={result.message}")
    print(f"optimizer_nit={getattr(result, 'nit', 'NA')}")
    print(f"optimizer_nfev={getattr(result, 'nfev', 'NA')}")
    print(f"pulse_plot={pulse_path}")
    print(f"propagation_plot={propagation_path}")


if __name__ == "__main__":
    main()
