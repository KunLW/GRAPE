"""Pulse robustness evaluation: open-system gate fidelity vs. noise scale.

Takes any valid experiment config.yaml plus an exported pulse .npz and sweeps
a global scale factor applied to the config's noise strengths — fluctuation
sigmas and decoherence rates are both multiplied by the scale, relative to
the values in the config. Noise types disabled in the config stay disabled
at every scale. At each scale the pulse is evaluated with
``noisy_gate_fidelity`` (perturbative expansion + first-order Lindblad
correction); ``--faithful`` additionally evaluates the exact
``faithful_gate_fidelity`` (full Lindblad propagation with Gauss-Hermite
averaging, cost ``hermite_points ** n_fluctuation_terms`` nodes per scale).

The ``closed_gate_fidelity`` with all noise disabled is the scale -> 0
reference; the log-scaled x-axis cannot show x = 0, so it is drawn as a
horizontal dotted line. Note the perturbative curve is a second-order
expansion in the noise strength and loses validity at large scales — the
gap to the faithful curve measures the truncation error.

``--close-grape-pulse-npz`` evaluates a second pulse (e.g. one optimized
with closed-system GRAPE) on the same scale grid and draws its curves in
the same figure, labeled "close-grape", for a robustness comparison. The
close-grape pulse keeps its own time grid from the npz (step count from
the amplitudes shape, dt from the npz; the config's dt is only a fallback
when the npz has none), so a pulse with a different duration is evaluated
as designed rather than stretched onto the config grid — both pulses face
the identical scaled noise model, but each over its own gate time.

Results land in a timestamped directory next to the pulse .npz (override
with ``--output-root``): ``robustness.csv``, ``robustness.png``,
``report.md``, and a resolved config snapshot.

Run from the repository root:

    .venv/bin/python -m experiments.robustness_eval.run_robustness_eval \
        --config <config.yaml> --pulse-npz <pulse.npz>
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from experiments.driver.run_experiment import (
    _load_base_config,
    build_initial_pulse,
    build_parameterization,
    build_state_pairs,
    build_systems,
    load_custom_initial_parameters,
)
from experiments.driver.config_io import write_config_snapshot
from experiments.driver.reporting import timestamped_experiment_dir
from quantum_control import (
    closed_gate_fidelity,
    faithful_gate_fidelity,
    noisy_gate_fidelity,
)
from quantum_control.pulses.pulse import PiecewiseConstantPulse

CSV_COLUMNS = (
    "scale",
    "noisy_gate_fidelity",
    "faithful_gate_fidelity",
    "close_grape_noisy_gate_fidelity",
    "close_grape_faithful_gate_fidelity",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a pulse's open-system fidelity while scaling the "
            "config's noise strengths (fluctuation sigmas and decoherence "
            "rates) by a global factor."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="YAML experiment configuration defining system, noise, and pulse grid.",
    )
    parser.add_argument(
        "--pulse-npz",
        type=Path,
        required=True,
        help="Pulse .npz with an amplitudes array matching the config's pulse grid.",
    )
    parser.add_argument(
        "--close-grape-pulse-npz",
        type=Path,
        default=None,
        help=(
            "Optional second pulse .npz (same grid) evaluated on the same "
            "scales and drawn in the same figure, labeled 'close-grape', "
            "for comparison."
        ),
    )
    parser.add_argument(
        "--scale-min",
        type=float,
        default=0.01,
        help="Smallest noise scale of the log-spaced grid (default: 0.01).",
    )
    parser.add_argument(
        "--scale-max",
        type=float,
        default=100.0,
        help="Largest noise scale of the log-spaced grid (default: 100).",
    )
    parser.add_argument(
        "--n-scales",
        type=int,
        default=13,
        help="Number of log-spaced scale points (default: 13).",
    )
    parser.add_argument(
        "--scales",
        type=str,
        default=None,
        help=(
            "Explicit comma-separated scale list (e.g. 0.01,0.1,1,10,100); "
            "overrides --scale-min/--scale-max/--n-scales."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Override runtime.workers for noisy_gate_fidelity state-pair averaging.",
    )
    parser.add_argument(
        "--faithful",
        action="store_true",
        help=(
            "Also compute faithful_gate_fidelity at every scale: exact Lindblad "
            "propagation with Gauss-Hermite averaging over the fluctuations "
            "(cost: hermite_points ** n_fluctuation_terms nodes per scale)."
        ),
    )
    parser.add_argument(
        "--hermite-points",
        type=int,
        default=5,
        help="Gauss-Hermite nodes per fluctuation dimension for --faithful (default: 5).",
    )
    parser.add_argument(
        "--y-scale",
        choices=("fidelity", "infidelity"),
        default="fidelity",
        help=(
            "Plot y-axis: 'fidelity' (linear F) or 'infidelity' (1 - F on a "
            "log axis; points with F >= 1 cannot be drawn and are dropped)."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Directory that receives the timestamped run directory "
            "(default: the folder containing --pulse-npz)."
        ),
    )
    return parser.parse_args(argv)


def resolve_scales(args):
    if args.scales is not None:
        scales = tuple(float(token) for token in args.scales.split(",") if token.strip())
        if not scales:
            raise ValueError("--scales must contain at least one value.")
    else:
        if args.scale_min <= 0.0 or args.scale_max <= 0.0:
            raise ValueError("--scale-min and --scale-max must be positive (log grid).")
        if args.n_scales < 1:
            raise ValueError("--n-scales must be at least 1.")
        scales = tuple(np.geomspace(args.scale_min, args.scale_max, args.n_scales))
    if any(scale <= 0.0 for scale in scales):
        raise ValueError("All noise scales must be positive (log-scaled axis).")
    return scales


def scale_noise_config(noise, scale):
    """Scale every fluctuation sigma and decoherence rate by ``scale``.

    ``enabled`` flags are untouched, so the sweep is strictly relative to the
    input config: noise types disabled there stay disabled at every scale.
    """
    fluctuations = replace(
        noise.fluctuations,
        **{name: scale * value for name, value in noise.fluctuations.sigmas.items()},
    )
    decoherence = replace(
        noise.decoherence,
        **{name: scale * value for name, value in noise.decoherence.rates.items()},
    )
    return replace(noise, fluctuations=fluctuations, decoherence=decoherence)


def scaled_config(config, scale):
    return replace(
        config,
        system=replace(
            config.system, noise=scale_noise_config(config.system.noise, scale)
        ),
    )


def load_pulse(npz_path, reference_pulse, parameterization):
    parameters, metadata = load_custom_initial_parameters(
        npz_path, reference_pulse, parameterization
    )
    for warning in metadata["warnings"]:
        print(warning, file=sys.stderr, flush=True)
    pulse = PiecewiseConstantPulse(
        amplitudes=parameterization.to_physical(parameters),
        dt=reference_pulse.dt,
    )
    return pulse, metadata


def load_comparison_pulse(npz_path, config):
    """Load the close-grape pulse on its own time grid.

    The grid comes from the npz itself: step count from the amplitudes
    shape, dt from the npz (the config's dt is only a fallback when the npz
    has none). Bounds and parameterization constraints are validated on
    that grid, so a pulse with a different duration is evaluated as
    designed rather than stretched onto the config grid.
    """
    npz_path = Path(npz_path)
    with np.load(npz_path) as data:
        if "amplitudes" not in data.files:
            raise ValueError(f"{npz_path} does not contain required 'amplitudes'.")
        n_steps = int(np.asarray(data["amplitudes"]).shape[0])
        dt = float(np.asarray(data["dt"]).reshape(())) if "dt" in data.files else None
    if dt is None:
        dt = config.pulse.total_time_us * 1e-6 / config.pulse.n_steps
    grid_config = replace(
        config,
        pulse=replace(config.pulse, n_steps=n_steps, total_time_us=n_steps * dt * 1e6),
    )
    reference_pulse = build_initial_pulse(grid_config)
    parameterization = build_parameterization(grid_config, reference_pulse)
    return load_pulse(npz_path, reference_pulse, parameterization)


def write_rows(csv_path, rows):
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    column: "" if row.get(column) is None else f"{row[column]:.12g}"
                    for column in CSV_COLUMNS
                }
            )


def plot_robustness(
    rows,
    closed_fidelity,
    output_path,
    *,
    faithful,
    hermite_points,
    y_scale,
    close_grape_closed_fidelity=None,
    close_grape_time_us=None,
):
    import matplotlib.pyplot as plt

    infidelity = y_scale == "infidelity"

    def series(column):
        """(scales, y-values) for the plot; infidelity mode drops F >= 1."""
        pairs = [(row["scale"], row[column]) for row in rows if row.get(column) is not None]
        if infidelity:
            dropped = sum(1 for _, value in pairs if value >= 1.0)
            if dropped:
                print(
                    f"warning: {dropped} {column} point(s) with F >= 1 cannot "
                    "be drawn on the log infidelity axis and were dropped.",
                    file=sys.stderr,
                    flush=True,
                )
            pairs = [(scale, 1.0 - value) for scale, value in pairs if value < 1.0]
        return [pair[0] for pair in pairs], [pair[1] for pair in pairs]

    # Color encodes the pulse, marker/linestyle the metric: solid + circle or
    # square = perturbative, dashed + diamond = faithful, dotted = closed ref.
    pulse_specs = [("", "", "tab:blue", "o", closed_fidelity)]
    if close_grape_closed_fidelity is not None:
        pulse_specs.append(
            ("close_grape_", "close-grape ", "tab:orange", "s", close_grape_closed_fidelity)
        )

    fig, ax = plt.subplots(figsize=(8, 5))
    for prefix, label_prefix, color, marker, closed in pulse_specs:
        scales, values = series(f"{prefix}noisy_gate_fidelity")
        if prefix:
            noisy_suffix = (
                "" if close_grape_time_us is None else f" (T={close_grape_time_us:.6g} us)"
            )
        else:
            noisy_suffix = " (perturbative + Lindblad correction)"
        ax.plot(
            scales,
            values,
            marker=marker,
            linestyle="-",
            color=color,
            label=f"{label_prefix}noisy_gate_fidelity{noisy_suffix}",
            markersize=5,
        )
        if faithful:
            scales, values = series(f"{prefix}faithful_gate_fidelity")
            ax.plot(
                scales,
                values,
                marker="D",
                linestyle="--",
                color=color,
                label=f"{label_prefix}faithful_gate_fidelity (hermite={hermite_points})",
                markersize=5,
            )
        closed_reference = 1.0 - closed if infidelity else closed
        if not infidelity or closed_reference > 0.0:
            ax.axhline(
                closed_reference,
                color=color,
                linestyle=":",
                linewidth=1.5,
                label=(
                    f"{label_prefix}closed_gate_fidelity, all noise disabled "
                    f"({closed_reference:.6g})"
                ),
            )
    ax.axvline(1.0, color="gray", linestyle="-.", linewidth=1.0, label="nominal (scale = 1)")
    ax.set_xscale("log")
    if infidelity:
        ax.set_yscale("log")
    ax.set_xlabel("noise scale (x config values)")
    ax.set_ylabel("gate infidelity (1 - F)" if infidelity else "gate fidelity")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="best")
    ax.set_title(
        "Pulse robustness: open "
        + ("infidelity" if infidelity else "fidelity")
        + " vs. noise scale"
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_report(
    report_path,
    *,
    generated_at,
    config,
    args,
    scales,
    closed_fidelity,
    rows,
    pulse_metadata,
    n_fluctuation_terms,
    n_decoherence_channels,
    wall_s,
    close_grape_closed_fidelity=None,
    close_grape_time_us=None,
):
    def fmt(value):
        return "not computed" if value is None else f"{value:.12g}"

    compare = close_grape_closed_fidelity is not None

    lines = [
        "# Robustness Evaluation: Open Fidelity vs. Noise Scale",
        "",
        f"Generated at: {generated_at.isoformat(timespec='seconds')}",
        "",
        "All fluctuation sigmas and decoherence rates from the config are "
        "multiplied by each scale factor (noise types disabled in the config "
        "stay disabled); the pulse is then evaluated with "
        "`noisy_gate_fidelity`. `closed_gate_fidelity` (all noise disabled) "
        "is the scale -> 0 reference. The perturbative expansion is "
        "second-order in the noise strength, so values at large scales are "
        "qualitative at best.",
        "",
        "## Run Summary",
        "",
        "| Parameter | Value |",
        "| --- | --- |",
        f"| config | {args.config} |",
        f"| pulse_npz | {pulse_metadata['source_npz']} |",
        f"| close_grape_pulse_npz | {args.close_grape_pulse_npz or 'disabled'} |",
        f"| system_type | {config.system.type} |",
        f"| n_steps | {config.pulse.n_steps} |",
        f"| total_time_us | {config.pulse.total_time_us} |",
        *(
            [f"| close_grape_total_time_us | {close_grape_time_us:.6g} |"]
            if compare and close_grape_time_us is not None
            else []
        ),
        f"| n_fluctuation_terms (scale=1) | {n_fluctuation_terms} |",
        f"| n_decoherence_channels (scale=1) | {n_decoherence_channels} |",
        f"| scales | {', '.join(f'{scale:.6g}' for scale in scales)} |",
        f"| faithful | {args.faithful} |",
        f"| hermite_points | {args.hermite_points if args.faithful else 'NA'} |",
        f"| y_scale | {args.y_scale} |",
        f"| workers | {config.runtime.workers} |",
        f"| closed_gate_fidelity | {closed_fidelity:.12g} |",
        *(
            [f"| close_grape_closed_gate_fidelity | {close_grape_closed_fidelity:.12g} |"]
            if compare
            else []
        ),
        f"| wall_s | {wall_s:.1f} |",
        "",
        "## Fidelity vs. Noise Scale",
        "",
    ]
    columns = ["noisy_gate_fidelity", "faithful_gate_fidelity"]
    if compare:
        columns += ["close_grape_noisy_gate_fidelity", "close_grape_faithful_gate_fidelity"]
    lines += [
        "| scale | " + " | ".join(columns) + " |",
        "| --- |" + " --- |" * len(columns),
    ]
    for row in rows:
        lines.append(
            f"| {row['scale']:.6g} | "
            + " | ".join(fmt(row.get(column)) for column in columns)
            + " |"
        )
    lines += [
        "",
        "## Figure",
        "",
        "![robustness](robustness.png)",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    scales = resolve_scales(args)
    config = _load_base_config(args.config)
    if args.workers is not None:
        config = replace(config, runtime=replace(config.runtime, workers=args.workers))

    reference_pulse = build_initial_pulse(config)
    parameterization = build_parameterization(config, reference_pulse)
    pulse, pulse_metadata = load_pulse(args.pulse_npz, reference_pulse, parameterization)
    close_grape_pulse = None
    close_grape_time_us = None
    if args.close_grape_pulse_npz is not None:
        close_grape_pulse, _ = load_comparison_pulse(args.close_grape_pulse_npz, config)
        close_grape_time_us = close_grape_pulse.n_steps * close_grape_pulse.dt * 1e6
    state_pairs = build_state_pairs(config)

    closed_system, nominal_open_system = build_systems(config)
    if not nominal_open_system.noise_terms:
        print(
            "warning: the config enables no noise terms; every scale point "
            "will equal the closed fidelity.",
            file=sys.stderr,
            flush=True,
        )
    closed_fidelity = float(closed_gate_fidelity(closed_system, pulse, state_pairs))
    close_grape_closed_fidelity = None
    if close_grape_pulse is not None:
        close_grape_closed_fidelity = float(
            closed_gate_fidelity(closed_system, close_grape_pulse, state_pairs)
        )

    generated_at = datetime.now()
    output_root = (
        args.output_root if args.output_root is not None else args.pulse_npz.resolve().parent
    )
    run_dir = timestamped_experiment_dir(output_root, "robustness_eval", generated_at)
    run_dir.mkdir(parents=True, exist_ok=True)
    write_config_snapshot(config, run_dir / "config.yaml")
    csv_path = run_dir / "robustness.csv"

    n_fluctuation_terms = len(nominal_open_system.fluctuation_terms)
    n_decoherence_channels = len(nominal_open_system.collapse_operators)
    print(
        f"run_dir={run_dir}\n"
        f"system={config.system.type}, n_steps={config.pulse.n_steps}, "
        f"n_state_pairs={len(state_pairs)}, "
        f"n_fluctuation_terms={n_fluctuation_terms}, "
        f"n_decoherence_channels={n_decoherence_channels}\n"
        f"closed_gate_fidelity={closed_fidelity:.9f} (all noise disabled)"
        + (
            ""
            if close_grape_closed_fidelity is None
            else f"\nclose-grape closed_gate_fidelity={close_grape_closed_fidelity:.9f}"
        ),
        flush=True,
    )
    if args.faithful and n_fluctuation_terms:
        n_nodes = args.hermite_points**n_fluctuation_terms
        print(
            f"faithful check: hermite_points={args.hermite_points} -> "
            f"{n_nodes} nodes per scale point",
            flush=True,
        )

    def evaluate(open_system, evaluated_pulse):
        noisy = float(
            noisy_gate_fidelity(
                open_system,
                evaluated_pulse,
                state_pairs,
                collapse_operators=open_system.collapse_operators,
                n_workers=config.runtime.workers,
            )
        )
        faithful = None
        if args.faithful:
            faithful = float(
                faithful_gate_fidelity(
                    open_system,
                    evaluated_pulse,
                    state_pairs,
                    hermite_points=args.hermite_points,
                )
            )
        return noisy, faithful

    def progress_note(noisy, faithful):
        return f"noisy={noisy:.9f}" + ("" if faithful is None else f" faithful={faithful:.9f}")

    rows = []
    start = time.perf_counter()
    for scale in scales:
        _, open_system = build_systems(scaled_config(config, scale))
        noisy, faithful = evaluate(open_system, pulse)
        row = {
            "scale": float(scale),
            "noisy_gate_fidelity": noisy,
            "faithful_gate_fidelity": faithful,
            "close_grape_noisy_gate_fidelity": None,
            "close_grape_faithful_gate_fidelity": None,
        }
        note = progress_note(noisy, faithful)
        if close_grape_pulse is not None:
            noisy, faithful = evaluate(open_system, close_grape_pulse)
            row["close_grape_noisy_gate_fidelity"] = noisy
            row["close_grape_faithful_gate_fidelity"] = faithful
            note += f" | close-grape {progress_note(noisy, faithful)}"
        rows.append(row)
        write_rows(csv_path, rows)
        print(f"scale {scale:>10.6g}: {note}", flush=True)
    wall_s = time.perf_counter() - start

    plot_path = run_dir / "robustness.png"
    plot_robustness(
        rows,
        closed_fidelity,
        plot_path,
        faithful=args.faithful,
        hermite_points=args.hermite_points,
        y_scale=args.y_scale,
        close_grape_closed_fidelity=close_grape_closed_fidelity,
        close_grape_time_us=close_grape_time_us,
    )
    report_path = run_dir / "report.md"
    write_report(
        report_path,
        generated_at=generated_at,
        config=config,
        args=args,
        scales=scales,
        closed_fidelity=closed_fidelity,
        rows=rows,
        pulse_metadata=pulse_metadata,
        n_fluctuation_terms=n_fluctuation_terms,
        n_decoherence_channels=n_decoherence_channels,
        wall_s=wall_s,
        close_grape_closed_fidelity=close_grape_closed_fidelity,
        close_grape_time_us=close_grape_time_us,
    )
    for path in (csv_path, plot_path, report_path):
        if not path.exists() or path.stat().st_size == 0:
            raise RuntimeError(f"Expected non-empty output at {path}.")
    print(f"robustness_csv={csv_path}")
    print(f"robustness_plot={plot_path}")
    print(f"report={report_path}")


if __name__ == "__main__":
    main()
