"""Preview every gallery initial pulse: shape, fidelities, and regularity.

Evaluates each pulse in ``pulse_gallery`` under the given config (no
optimization, no per-pulse run directories). ``--output-dir`` is the GROUP
folder for this setup — the same one run_search.py writes into — and the
preview lands in a subfolder, keeping everything for one config together:

    <output-dir>/config.yaml           snapshot of the config (written if absent)
    <output-dir>/preview/preview.png   one subplot per pulse, both control channels
    <output-dir>/preview/preview.md    table of close/open gate fidelity + L1/L2
    <output-dir>/preview/preview.csv   same table, machine-readable

The L1/L2 values are the weighted smoothness penalties the optimizer
subtracts from the fidelity (computed on the optimizer parameters, same
convention as initial_l1_penalty/initial_l2_penalty in run reports).

Usage:
    python -m experiments.spin_boson.pulse_search.preview_pulses \\
        --config experiments/spin_boson/pulse_search/smoke.yaml \\
        --output-dir experiments/spin_boson/pulse_search/outputs/smoke
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

from experiments.driver.config_io import write_config_snapshot
from experiments.spin_boson.pulse_search.pulse_gallery import build_pulse, pulse_names
from experiments.driver.run_experiment import (
    _load_base_config,
    build_initial_pulse,
    build_parameterization,
    build_state_pairs,
    build_systems,
    system_definition,
)
from quantum_control.evaluation import closed_gate_fidelity, noisy_gate_fidelity
from quantum_control.problems.penalties import ParameterSmoothPenalty
from quantum_control.pulses.pulse import PiecewiseConstantPulse

import matplotlib.pyplot as plt  # backend configured by run_experiment import
import numpy as np

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = PACKAGE_DIR / "search.yaml"
DEFAULT_OUTPUT_DIR = PACKAGE_DIR / "outputs"

HEADERS = ("pulse", "close_gate", "open_gate", "l1_penalty", "l2_penalty")


def preview_gallery(config, workers=None):
    """Return one metrics row per gallery pulse plus the pulse objects for plotting."""
    params = config.system.params
    system, open_system = build_systems(config)
    state_pairs = build_state_pairs(config)
    reference = build_initial_pulse(config)
    parameterization = build_parameterization(config, reference)
    penalty = ParameterSmoothPenalty(
        l1_weight=config.penalty.l1_smooth_weight,
        l2_weight=config.penalty.l2_smooth_weight,
    )
    n_workers = workers if workers is not None else config.runtime.workers

    rows = []
    pulses = {}
    for name in pulse_names():
        amplitudes = build_pulse(
            name,
            n_steps=config.pulse.n_steps,
            alpha1_khz_bounds=tuple(getattr(params, "alpha1_khz_bounds", (1.0, 60.0))),
            alpha2_khz_bounds=tuple(getattr(params, "alpha2_khz_bounds", (0.0, 200.0))),
        )
        pulse = PiecewiseConstantPulse(amplitudes=amplitudes, dt=reference.dt)
        pulses[name] = pulse
        parameters = np.asarray(parameterization.to_parameters(amplitudes))
        rows.append(
            {
                "pulse": name,
                "close_gate": closed_gate_fidelity(system, pulse, state_pairs),
                "open_gate": noisy_gate_fidelity(
                    open_system,
                    pulse,
                    state_pairs,
                    collapse_operators=open_system.collapse_operators,
                    n_workers=n_workers,
                ),
                "l1_penalty": penalty.l1_value(parameters, parameters.shape),
                "l2_penalty": penalty.l2_value(parameters, parameters.shape),
            }
        )
        print(
            f"{name:18s} close={rows[-1]['close_gate']:.6f}  open={rows[-1]['open_gate']:.6f}  "
            f"l1={rows[-1]['l1_penalty']:.3g}  l2={rows[-1]['l2_penalty']:.3g}",
            flush=True,
        )
    rows.sort(key=lambda row: row["open_gate"], reverse=True)
    return rows, pulses


def plot_preview(rows, pulses, channels, output_path, n_cols=2):
    """Grid figure, one subplot per pulse (best open fidelity first)."""
    n_rows = math.ceil(len(rows) / n_cols)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(6 * n_cols, 2.4 * n_rows), sharex=True, squeeze=False
    )
    for slot, row in enumerate(rows):
        axis = axes[slot // n_cols, slot % n_cols]
        pulse = pulses[row["pulse"]]
        time_us = np.arange(pulse.n_steps) * pulse.dt * 1e6
        for index, channel in enumerate(channels):
            axis.plot(
                time_us,
                pulse.amplitudes[:, index] * channel.display_scale,
                label=f"{channel.label} ({channel.display_unit})",
                linewidth=1.5,
            )
        axis.set_title(
            f"{row['pulse']}   close={row['close_gate']:.4f}  open={row['open_gate']:.4f}\n"
            f"l1={row['l1_penalty']:.3g}  l2={row['l2_penalty']:.3g}",
            fontsize=9,
        )
        axis.grid(True, alpha=0.3)
    for slot in range(len(rows), n_rows * n_cols):
        axes[slot // n_cols, slot % n_cols].set_visible(False)
    for column in range(n_cols):
        axes[-1, column].set_xlabel("time (us)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=len(labels))
    fig.suptitle("Gallery initial pulses (best open-gate fidelity first)")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_tables(rows, output_dir):
    csv_path = output_dir / "preview.csv"
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADERS)
        writer.writeheader()
        writer.writerows(rows)

    md_path = output_dir / "preview.md"
    with open(md_path, "w") as handle:
        handle.write("# Gallery pulse preview (best open-gate fidelity first)\n\n")
        handle.write("| " + " | ".join(HEADERS) + " |\n")
        handle.write("|" + "|".join(" --- " for _ in HEADERS) + "|\n")
        for row in rows:
            cells = [row["pulse"]] + [f"{row[key]:.6g}" for key in HEADERS[1:]]
            handle.write("| " + " | ".join(cells) + " |\n")
    return csv_path, md_path


def print_table(rows):
    print(f"\n=== Gallery pulse preview (best open-gate first) ===")
    print(f"{'pulse':<18}  {'close_gate':>10}  {'open_gate':>10}  {'l1_penalty':>10}  {'l2_penalty':>10}")
    for row in rows:
        print(
            f"{row['pulse']:<18}  {row['close_gate']:>10.6f}  {row['open_gate']:>10.6f}  "
            f"{row['l1_penalty']:>10.4g}  {row['l2_penalty']:>10.4g}"
        )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Experiment YAML.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Group folder for this setup (same as run_search.py); preview goes to <output-dir>/preview.",
    )
    parser.add_argument("--workers", type=int, help="Override runtime.workers for the fidelity evaluation.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    config = _load_base_config(args.config)
    group_dir = Path(args.output_dir)
    output_dir = group_dir / "preview"
    output_dir.mkdir(parents=True, exist_ok=True)
    group_config = group_dir / "config.yaml"
    if not group_config.exists():
        write_config_snapshot(config, group_config)

    rows, pulses = preview_gallery(config, workers=args.workers)
    channels = system_definition(config).control_channels(config.system.params)
    plot_path = output_dir / "preview.png"
    plot_preview(rows, pulses, channels, plot_path)
    csv_path, md_path = write_tables(rows, output_dir)
    print_table(rows)
    print(f"\npreview_png={plot_path}")
    print(f"preview_csv={csv_path}")
    print(f"preview_md={md_path}")


if __name__ == "__main__":
    main()
