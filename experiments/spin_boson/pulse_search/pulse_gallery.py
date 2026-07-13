"""Gallery of named initial pulses for the pulse-search experiment.

Each pulse is a (n_steps, 2) array of spin-boson control amplitudes in rad/s:

    column 0 = alpha1 (AC-Stark / number term)
    column 1 = alpha2 (spin-motion coupling)

alpha2 is forced to zero at the first and last step (the endpoint-zero
constraint enforced by the spin-boson parameterization), and both channels are
clipped to the control bounds before conversion to rad/s.

Adding a pulse = writing one decorated function that returns the two channel
profiles in kHz (arrays broadcastable to length ``n_steps``):

    @pulse("my_shape")
    def _my_shape(t):
        return 30.0, 120.0 * np.sin(np.pi * t)

``t`` is the normalized time axis (0..1, length n_steps). Registration order
defines the stable index used by ``run_search.py --index`` / Slurm array tasks.

Usage:
    python -m experiments.spin_boson.pulse_search.pulse_gallery --list
    python -m experiments.spin_boson.pulse_search.pulse_gallery --write-dir <dir> --n-steps 40 200 400

``--write-dir`` exports every pulse as ``<dir>/n<steps>/<name>_s<steps>.npz``
(one subfolder per ``--n-steps`` value, each ``.npz`` holding ``amplitudes`` +
``dt``), ready for reuse via ``runtime.initial_pulse_npz`` / ``--pulse-npz``
with any config whose ``pulse.n_steps`` matches.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from quantum_control.units import RAD_S_PER_KHZ

ALPHA1_KHZ_BOUNDS = (1.0, 60.0)
ALPHA2_KHZ_BOUNDS = (0.0, 200.0)
DEFAULT_N_STEPS = 400
DEFAULT_TOTAL_TIME_US = 225.8

GALLERY: dict[str, callable] = {}


def pulse(name):
    """Register a gallery entry; the function maps t (0..1) -> (alpha1_khz, alpha2_khz)."""

    def decorator(func):
        if name in GALLERY:
            raise ValueError(f"duplicate gallery pulse name: {name}")
        GALLERY[name] = func
        return func

    return decorator


def _smooth(values, window=9):
    kernel = np.ones(window) / window
    padded = np.pad(values, window // 2, mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: len(values)]


# @pulse("sine_60")
# def _sine_60(t):
#     return 15.0, 60.0 * np.sin(np.pi * t)


@pulse("sine_100_cos_60")
def _sine_100_cos_60(t):
    return 60.0 * np.cos(np.pi * t), 100.0 * np.sin(np.pi * t)

@pulse("sine_100_cos_60_2_lobe")
def _sine_100_cos_60_2_lobe(t):
    return 60.0 * np.cos(2 * np.pi * t), 100.0 * np.sin(np.pi * t)

@pulse("flattopdown")
def _flattop(t):
    return 35.0 * (1- np.clip(np.sin(np.pi * t) * 3.0, 0.0, 1.0)), 150.0 * np.clip(np.sin(np.pi * t) * 3.0, 0.0, 1.0)

@pulse("random")
def _random(t):
    rng = np.random.default_rng(7)  # fixed seed -> reproducible "random" pulse
    window = np.sin(np.pi * t)
    alpha1 = _smooth(rng.uniform(10.0, 50.0, t.size))
    alpha2 = _smooth(rng.uniform(20.0, 180.0, t.size))
    return alpha1, alpha2


# @pulse("sine_160")
# def _sine_160(t):
#     return 45.0, 160.0 * np.sin(np.pi * t)


# @pulse("two_lobe")
# def _two_lobe(t):
#     return 25.0, 140.0 * np.sin(np.pi * t) * np.abs(np.sin(2 * np.pi * t))


# @pulse("multilobe_3")
# def _multilobe_3(t):
#     window = np.sin(np.pi * t)
#     return 20.0 + 20.0 * window, 120.0 * window * np.abs(np.sin(3 * np.pi * t))




# @pulse("gaussian_lobe")
# def _gaussian_lobe(t):
#     envelope = np.exp(-(((t - 0.5) / 0.18) ** 2))
#     return 30.0, 150.0 * envelope * np.sin(np.pi * t) ** 0.25


# @pulse("chirp_ramp")
# def _chirp_ramp(t):
#     return 10.0 + 40.0 * t, 120.0 * np.sin(np.pi * t)


# @pulse("high_alpha1_const")
# def _high_alpha1_const(t):
#     return 55.0, 80.0 * np.sin(np.pi * t)


# @pulse("random_smooth")
# def _random_smooth(t):
#     rng = np.random.default_rng(7)  # fixed seed -> reproducible "random" pulse
#     window = np.sin(np.pi * t)
#     alpha1 = _smooth(rng.uniform(10.0, 50.0, t.size))
#     alpha2 = _smooth(rng.uniform(20.0, 180.0, t.size)) * window
#     return alpha1, alpha2



def pulse_names():
    """Gallery names in registration order (the --index order)."""
    return list(GALLERY)


def build_pulse(
    name,
    n_steps=DEFAULT_N_STEPS,
    alpha1_khz_bounds=ALPHA1_KHZ_BOUNDS,
    alpha2_khz_bounds=ALPHA2_KHZ_BOUNDS,
):
    """Return the named pulse as validated (n_steps, 2) amplitudes in rad/s."""
    if name not in GALLERY:
        raise KeyError(f"unknown gallery pulse {name!r}; choose from {pulse_names()}")
    t = np.arange(n_steps) / (n_steps - 1)
    alpha1_khz, alpha2_khz = GALLERY[name](t)
    alpha1 = np.clip(np.broadcast_to(alpha1_khz, (n_steps,)).astype(float), *alpha1_khz_bounds)
    alpha2 = np.clip(np.broadcast_to(alpha2_khz, (n_steps,)).astype(float), *alpha2_khz_bounds)
    alpha2[[0, -1]] = 0.0
    amplitudes = np.column_stack([alpha1, alpha2]) * RAD_S_PER_KHZ
    _validate(name, amplitudes, n_steps, alpha1_khz_bounds, alpha2_khz_bounds)
    return amplitudes


def _validate(name, amplitudes, n_steps, alpha1_khz_bounds, alpha2_khz_bounds):
    if amplitudes.shape != (n_steps, 2):
        raise ValueError(f"{name}: expected shape {(n_steps, 2)}, got {amplitudes.shape}.")
    if not np.allclose(amplitudes[[0, -1], 1], 0.0):
        raise ValueError(f"{name}: alpha2 endpoints must be zero.")
    khz = amplitudes / RAD_S_PER_KHZ
    for column, bounds, label in ((0, alpha1_khz_bounds, "alpha1"), (1, alpha2_khz_bounds, "alpha2")):
        if khz[:, column].min() < bounds[0] - 1e-6 or khz[:, column].max() > bounds[1] + 1e-6:
            raise ValueError(f"{name}: {label} out of bounds {bounds} kHz.")


def write_pulses(output_dir, n_steps=DEFAULT_N_STEPS, total_time_us=DEFAULT_TOTAL_TIME_US):
    """Export every gallery pulse as <name>.npz (amplitudes + dt) into output_dir."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dt = float(total_time_us) * 1e-6 / n_steps
    written = []
    for name in pulse_names():
        amplitudes = build_pulse(name, n_steps)
        # Repo convention: pulse .npz names carry the step count (see
        # experiments/driver/reporting.py export_pulse_controls).
        path = output_dir / f"{name}_s{n_steps}.npz"
        np.savez(path, amplitudes=amplitudes, dt=dt)
        written.append(path)
        khz = amplitudes / RAD_S_PER_KHZ
        print(
            f"{name:18s} a1[{khz[:, 0].min():5.1f},{khz[:, 0].max():5.1f}]kHz  "
            f"a2[{khz[:, 1].min():5.1f},{khz[:, 1].max():6.1f}]kHz  -> {path.name}"
        )
    print(f"\nwrote {len(written)} pulses to {output_dir}")
    return written


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list", action="store_true", help="List pulse names (index order) and exit.")
    parser.add_argument(
        "--write-dir",
        type=Path,
        help="Export all pulses as <dir>/n<steps>/<name>.npz, one subfolder per --n-steps value.",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        nargs="+",
        default=[DEFAULT_N_STEPS],
        help="One or more grid sizes to export (default: %(default)s).",
    )
    parser.add_argument("--total-time-us", type=float, default=DEFAULT_TOTAL_TIME_US)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.list:
        for index, name in enumerate(pulse_names()):
            print(f"{index:2d}  {name}")
        return
    if args.write_dir is None:
        raise SystemExit("Pass --list or --write-dir <dir>.")
    for n_steps in args.n_steps:
        write_pulses(args.write_dir / f"n{n_steps}", n_steps, args.total_time_us)


if __name__ == "__main__":
    main()
