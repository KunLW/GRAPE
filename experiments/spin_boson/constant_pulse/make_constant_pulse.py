"""Constant-amplitude MS-gate baseline pulse.

Generates a piecewise-constant pulse with both controls flat over the whole
gate, on the pulse grid (``n_steps``, ``total_time_us``) and Lamb-Dicke
``eta`` taken from the config:

    alpha1 = 2*pi / T           one phase-space loop over the gate time
    alpha2 = 2*pi / (eta * T)   loop closes with entangling phase pi/2

With H = alpha1 (I x n) + alpha2 * eta * S_phi x (a + adag)/2, the detuning
is delta = alpha1 and the effective drive g = alpha2 * eta / 2. The loop
closes when delta * T = 2*pi and accumulates the geometric phase
Theta = 2*pi * (g / delta)^2 per S_phi eigenvalue squared; the maximally
entangling MS gate XX(pi/2) = exp(i Theta S_phi^2) (up to global phase)
needs Theta = pi/2, i.e. g = delta / 2, i.e. alpha2 = alpha1 / eta.

When the config sets ``system.params.alpha2_endpoint_zero: true``, the first
and last alpha2 steps are zeroed to match the endpoint constraint the shared
parameterization then enforces; with ``false`` (this folder's config) the
pulse is flat on every step.

Outputs (npz + csv in the exporter's standard format, plus a plot) land in
``--output-root`` (default: this folder's ``outputs/``). Run from the
repository root:

    .venv/bin/python -m experiments.spin_boson.constant_pulse.make_constant_pulse
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from experiments.driver.reporting import export_pulse_controls
from experiments.driver.run_experiment import (
    _load_base_config,
    build_initial_pulse,
    build_parameterization,
    build_state_pairs,
    build_systems,
    load_custom_initial_parameters,
)
from quantum_control import RAD_S_PER_KHZ, closed_gate_fidelity
from quantum_control.pulses.pulse import PiecewiseConstantPulse

HERE = Path(__file__).resolve().parent


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate the constant-amplitude MS-gate baseline pulse."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=HERE / "config.yaml",
        help="Experiment config supplying n_steps, total_time_us, and eta.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=HERE / "outputs",
        help="Directory receiving the pulse npz/csv/png (default: ./outputs).",
    )
    return parser.parse_args(argv)


def constant_ms_amplitudes(n_steps, total_time_s, eta, alpha2_endpoint_zero):
    """Flat alpha1/alpha2 columns; alpha2 endpoints zeroed only if required."""
    alpha1 = 2.0 * np.pi / total_time_s
    alpha2 = 2.0 * np.pi / (eta * total_time_s)
    amplitudes = np.column_stack(
        [np.full(n_steps, alpha1), np.full(n_steps, alpha2)]
    )
    if alpha2_endpoint_zero:
        amplitudes[[0, -1], 1] = 0.0
    return amplitudes


def plot_pulse(pulse, png_path):
    import matplotlib.pyplot as plt

    time_us = (np.arange(pulse.n_steps) + 0.5) * pulse.dt * 1e6
    fig, ax = plt.subplots(figsize=(8, 4))
    for column, label in enumerate(("alpha1", "alpha2")):
        ax.step(
            time_us,
            pulse.amplitudes[:, column] / RAD_S_PER_KHZ,
            where="mid",
            label=label,
        )
    ax.set_xlabel("time (us)")
    ax.set_ylabel("amplitude (kHz)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="center right")
    ax.set_title("Constant MS-gate pulse")
    fig.tight_layout()
    fig.savefig(png_path, dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    config = _load_base_config(args.config)

    n_steps = config.pulse.n_steps
    total_time_s = config.pulse.total_time_us * 1e-6
    eta = config.system.params.eta
    amplitudes = constant_ms_amplitudes(
        n_steps, total_time_s, eta, config.system.params.alpha2_endpoint_zero
    )
    pulse = PiecewiseConstantPulse(amplitudes=amplitudes, dt=total_time_s / n_steps)

    args.output_root.mkdir(parents=True, exist_ok=True)
    output_stem = args.output_root / "constant_pulse"
    export_pulse_controls(pulse, output_stem, RAD_S_PER_KHZ)
    npz_path = output_stem.with_name(f"constant_pulse_s{n_steps}.npz")
    plot_pulse(pulse, output_stem.with_suffix(".png"))

    # Round-trip through the standard loader so bound violations or endpoint
    # constraint mismatches fail here rather than at evaluation time.
    reference_pulse = build_initial_pulse(config)
    parameterization = build_parameterization(config, reference_pulse)
    load_custom_initial_parameters(npz_path, reference_pulse, parameterization)

    closed_system, _ = build_systems(config)
    fidelity = float(
        closed_gate_fidelity(closed_system, pulse, build_state_pairs(config))
    )

    alpha1_khz = amplitudes[0, 0] / RAD_S_PER_KHZ
    alpha2_khz = amplitudes[1, 1] / RAD_S_PER_KHZ
    endpoint_note = (
        "endpoints zeroed"
        if config.system.params.alpha2_endpoint_zero
        else "flat on all steps"
    )
    print(
        f"n_steps={n_steps}, total_time_us={config.pulse.total_time_us:.6f}, "
        f"eta={eta}\n"
        f"alpha1={alpha1_khz:.6f} kHz (2*pi/T), "
        f"alpha2={alpha2_khz:.6f} kHz (2*pi/(eta*T), {endpoint_note})\n"
        f"closed_gate_fidelity={fidelity:.9f}\n"
        f"pulse_npz={npz_path}"
    )


if __name__ == "__main__":
    main()
