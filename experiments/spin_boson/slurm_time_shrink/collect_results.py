"""Collect time-shrink results into a per-pulse summary table and plot.

Scans ``<output-dir>/*/result.json`` (written by run_time_shrink.py), prints
a table grouped by pulse (grid order: orig, 300us, 170us per pulse), and
writes ``summary.csv``, ``summary.md``, and ``best_time_vs_fidelity.png``
into the output directory. Grid tasks without a result.json are listed as
missing, so this works mid-run for standings.

Usage:
    python -m experiments.spin_boson.slurm_time_shrink.collect_results
    python -m experiments.spin_boson.slurm_time_shrink.collect_results --output-dir <dir>
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.spin_boson.slurm_time_shrink.run_time_shrink import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PULSES_DIR,
    task_dir_name,
    task_grid,
)

HEADERS = (
    "pulse",
    "start",
    "start_time_us",
    "best_time_us",
    "best_noisy",
    "best_close",
    "rounds",
    "stop",
    "status",
)


def load_records(output_dir, pulses_dir):
    """One record per grid task (grid order) plus any extras on disk."""
    output_dir = Path(output_dir)
    on_disk = {}
    for result_path in sorted(output_dir.glob("*/result.json")):
        with open(result_path) as handle:
            on_disk[result_path.parent.name] = json.load(handle)

    records = []
    for pulse, label, _, _ in task_grid(pulses_dir):
        name = task_dir_name(pulse, label)
        records.append(
            on_disk.pop(name, {"pulse": pulse, "start_label": label, "status": "missing"})
        )
    records.extend(on_disk.values())  # tasks no longer in the grid
    return records


def _short_stop_reason(record):
    return str(record.get("stop_reason", record.get("error", ""))).split(" (")[0]


def build_rows(records):
    rows = []
    for record in records:
        metrics = record.get("metrics", {})
        rows.append(
            [
                record.get("pulse", ""),
                record.get("start_label", ""),
                record.get("start_total_time_us", ""),
                metrics.get("best_total_time_us", ""),
                metrics.get("best_noisy_gate_fidelity", ""),
                metrics.get("best_close_gate_fidelity", ""),
                metrics.get("rounds", ""),
                _short_stop_reason(record),
                record.get("status", "missing"),
            ]
        )
    return rows


def _format_cell(value):
    return f"{value:.6f}" if isinstance(value, float) else str(value)


def write_summary(output_dir, rows):
    output_dir = Path(output_dir)
    csv_path = output_dir / "summary.csv"
    with open(csv_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(HEADERS)
        writer.writerows(rows)

    md_path = output_dir / "summary.md"
    formatted = [[_format_cell(value) for value in row] for row in rows]
    with open(md_path, "w") as handle:
        handle.write("# Time-shrink summary (per pulse: orig, 300us, 170us)\n\n")
        handle.write("| " + " | ".join(HEADERS) + " |\n")
        handle.write("|" + "|".join(" --- " for _ in HEADERS) + "|\n")
        for row in formatted:
            handle.write("| " + " | ".join(row) + " |\n")
    return csv_path, md_path


def plot_best_time_vs_fidelity(output_dir, records):
    """Best noisy fidelity vs best total time, one marker per finished task."""
    by_label = {}
    for record in records:
        metrics = record.get("metrics", {})
        if "best_total_time_us" not in metrics:
            continue
        by_label.setdefault(record.get("start_label", "?"), []).append(
            (
                metrics["best_total_time_us"],
                metrics["best_noisy_gate_fidelity"],
                record.get("pulse", ""),
            )
        )
    if not by_label:
        return None
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    markers = {"orig": "o", "300us": "s", "170us": "^"}
    for label, points in by_label.items():
        times, fidelities, pulses = zip(*points)
        ax.scatter(times, fidelities, label=f"from {label}", marker=markers.get(label, "x"))
        for time, fidelity, pulse in points:
            ax.annotate(pulse, (time, fidelity), fontsize=6, alpha=0.7)
    ax.set_xlabel("best total pulse time (us)")
    ax.set_ylabel("best noisy gate fidelity")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = Path(output_dir) / "best_time_vs_fidelity.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def print_table(rows):
    formatted = [[_format_cell(value) for value in row] for row in rows]
    widths = [
        max(len(HEADERS[i]), *(len(row[i]) for row in formatted))
        for i in range(len(HEADERS))
    ]
    print("  ".join(header.ljust(width) for header, width in zip(HEADERS, widths)))
    for row in formatted:
        print("  ".join(cell.ljust(width) for cell, width in zip(row, widths)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--pulses-dir", type=Path, default=DEFAULT_PULSES_DIR)
    args = parser.parse_args(argv)

    records = load_records(args.output_dir, args.pulses_dir)
    rows = build_rows(records)
    print("=== Time-shrink summary (per pulse: orig, 300us, 170us) ===")
    print_table(rows)
    csv_path, md_path = write_summary(args.output_dir, rows)
    plot_path = plot_best_time_vs_fidelity(args.output_dir, records)
    print(f"\nsummary_csv={csv_path}")
    print(f"summary_md={md_path}")
    if plot_path:
        print(f"plot={plot_path}")


if __name__ == "__main__":
    main()
