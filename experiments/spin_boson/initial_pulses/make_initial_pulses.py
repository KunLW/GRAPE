"""Generate a set of initial-pulse .npz files for the spin-boson open-gate experiment.

Each pulse is a (n_steps, 2) array of control amplitudes in rad/s:

    column 0 = alpha1 (AC-Stark / number term),   bounds [1, 60] kHz
    column 1 = alpha2 (spin-motion coupling),      bounds [0, 200] kHz

alpha2 must be exactly zero at the first and last step (the endpoint-zero
constraint enforced by ``Alpha2EndpointZeroParameterization``). Files are saved
with a matching ``dt`` so they load through ``run_experiment.py`` without a
dt-mismatch warning.

Usage:
    python -m experiments.spin_boson.initial_pulses.make_initial_pulses            # write all pulses
    python -m experiments.spin_boson.initial_pulses.make_initial_pulses --list     # list names only
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

RAD_S_PER_KHZ = 2.0 * np.pi * 1000.0
ALPHA1_KHZ_BOUNDS = (1.0, 60.0)
ALPHA2_KHZ_BOUNDS = (0.0, 200.0)
DEFAULT_N_STEPS = 200
DEFAULT_TOTAL_TIME_US = 225.8
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "initial_pulses"


def _assemble(alpha1_khz, alpha2_khz, n_steps):
    """Clip to bounds, zero the alpha2 endpoints, and convert kHz -> rad/s."""
    alpha1 = np.clip(np.broadcast_to(alpha1_khz, (n_steps,)).astype(float), *ALPHA1_KHZ_BOUNDS)
    alpha2 = np.clip(np.broadcast_to(alpha2_khz, (n_steps,)).astype(float), *ALPHA2_KHZ_BOUNDS)
    alpha2[[0, -1]] = 0.0
    return np.column_stack([alpha1, alpha2]) * RAD_S_PER_KHZ


def _smooth(values, window=9):
    kernel = np.ones(window) / window
    padded = np.pad(values, window // 2, mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: len(values)]


def build_pulses(n_steps=DEFAULT_N_STEPS):
    """Return an ordered mapping name -> amplitudes (rad/s) for the pulse set."""
    t = np.arange(n_steps) / (n_steps - 1)  # 0..1
    window = np.sin(np.pi * t)              # zero at both endpoints
    rng = np.random.default_rng(7)          # fixed seed -> reproducible "random" pulse
    random_alpha1 = _smooth(rng.uniform(10.0, 50.0, n_steps))
    random_alpha2 = _smooth(rng.uniform(20.0, 180.0, n_steps))

    return {
        # single sine lobe, gentle balanced drive
        "ms_sine_100": _assemble(30.0, 100.0 * window, n_steps),
        # single sine lobe, stronger drive + higher alpha1 offset
        "ms_sine_160": _assemble(45.0, 160.0 * window, n_steps),
        # single sine lobe, low-power start
        "ms_sine_60": _assemble(15.0, 60.0 * window, n_steps),
        # three-lobe sideband with a ramped alpha1
        "ms_multilobe": _assemble(
            20.0 + 20.0 * window, 120.0 * window * np.abs(np.sin(3 * np.pi * t)), n_steps
        ),
        # raised-cosine flat-top alpha2 (near-constant sideband)
        "ms_flattop": _assemble(35.0, 150.0 * np.clip(np.sin(np.pi * t) * 3.0, 0.0, 1.0), n_steps),
        # reproducible smoothed-random guess
        "ms_random_smooth": _assemble(random_alpha1, random_alpha2 * window, n_steps),
    }


def _validate(name, amplitudes, n_steps):
    if amplitudes.shape != (n_steps, 2):
        raise ValueError(f"{name}: expected shape {(n_steps, 2)}, got {amplitudes.shape}.")
    if not np.allclose(amplitudes[[0, -1], 1], 0.0):
        raise ValueError(f"{name}: alpha2 endpoints must be zero.")
    alpha1_khz = amplitudes[:, 0] / RAD_S_PER_KHZ
    alpha2_khz = amplitudes[:, 1] / RAD_S_PER_KHZ
    if alpha1_khz.min() < ALPHA1_KHZ_BOUNDS[0] - 1e-6 or alpha1_khz.max() > ALPHA1_KHZ_BOUNDS[1] + 1e-6:
        raise ValueError(f"{name}: alpha1 out of bounds {ALPHA1_KHZ_BOUNDS} kHz.")
    if alpha2_khz.min() < ALPHA2_KHZ_BOUNDS[0] - 1e-6 or alpha2_khz.max() > ALPHA2_KHZ_BOUNDS[1] + 1e-6:
        raise ValueError(f"{name}: alpha2 out of bounds {ALPHA2_KHZ_BOUNDS} kHz.")


def write_pulses(output_dir=DEFAULT_OUTPUT_DIR, n_steps=DEFAULT_N_STEPS, total_time_us=DEFAULT_TOTAL_TIME_US):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dt = float(total_time_us) * 1e-6 / n_steps
    written = []
    for name, amplitudes in build_pulses(n_steps).items():
        _validate(name, amplitudes, n_steps)
        path = output_dir / f"{name}_s{n_steps}.npz"
        np.savez(path, amplitudes=amplitudes, dt=dt)
        written.append(path)
        alpha1_khz = amplitudes[:, 0] / RAD_S_PER_KHZ
        alpha2_khz = amplitudes[:, 1] / RAD_S_PER_KHZ
        print(
            f"{name:18s} a1[{alpha1_khz.min():5.1f},{alpha1_khz.max():5.1f}]kHz  "
            f"a2[{alpha2_khz.min():5.1f},{alpha2_khz.max():6.1f}]kHz  -> {path.name}"
        )
    print(f"\nwrote {len(written)} pulses to {output_dir}")
    return written


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-steps", type=int, default=DEFAULT_N_STEPS)
    parser.add_argument("--total-time-us", type=float, default=DEFAULT_TOTAL_TIME_US)
    parser.add_argument("--list", action="store_true", help="List pulse names and exit.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.list:
        for name in build_pulses(args.n_steps):
            print(name)
        return
    write_pulses(args.output_dir, args.n_steps, args.total_time_us)


if __name__ == "__main__":
    main()
