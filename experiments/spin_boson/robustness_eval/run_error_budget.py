"""Per-noise-source error budget: infidelity contribution of each noise term.

Takes any valid experiment config.yaml plus an exported pulse .npz, builds the
nominal open system, then evaluates ``noisy_gate_fidelity`` once with the full
noise model and once per noise term with *only that term* active (the term
list of an ``OpenSystem`` is the selection of what noise acts on it). The
infidelity attributable to source ``i`` is

    eps_i = F_closed - F(only term i),

and because the perturbative expansion is second order in independent
zero-mean fluctuations (and the Lindblad correction first order in the
rates), the contributions are additive at leading order: each output also
states ``eps_i`` as a percentage of the all-terms total ``F_closed - F_all``,
and the summary reports the residual ``(F_closed - F_all) - sum_i eps_i``
as a consistency check.

``--faithful`` additionally computes the exact ``faithful_gate_fidelity``
per source (full Lindblad propagation with Gauss-Hermite averaging; with a
single fluctuation term active this costs only ``hermite_points`` nodes
instead of ``hermite_points ** n_terms``), so the per-source budget gets an
expansion-free cross-check. ``--validity`` appends the perturbative-method
validity budget (W unitarity, dW gradient consistency, V insertion error,
(sigma T)^2 truncation estimates) via ``quantum_control.evaluate_error_budget``
into a ``validity/`` subfolder.

``--close-grape-pulse-npz`` evaluates a second pulse (on its own time grid
from the npz, as in ``run_robustness_eval``) against the same per-source
budget for comparison.

Results land in a timestamped directory next to the pulse .npz (override
with ``--output-root``): ``error_budget.csv``, ``error_budget.png``,
``report.md``, and a resolved config snapshot.

Run from the repository root:

    .venv/bin/python -m experiments.spin_boson.robustness_eval.run_error_budget \
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
)
from experiments.driver.config_io import write_config_snapshot
from experiments.driver.reporting import timestamped_experiment_dir
from experiments.spin_boson.robustness_eval.run_robustness_eval import (
    load_comparison_pulse,
    load_pulse,
)
from quantum_control import (
    ErrorBudgetConfig,
    closed_gate_fidelity,
    evaluate_error_budget,
    faithful_gate_fidelity,
    noisy_gate_fidelity,
    write_error_budget_report,
)
from quantum_control.systems.noise import DecoherenceChannel, FluctuationTerm

CSV_COLUMNS = (
    "source",
    "kind",
    "strength",
    "scaled_spectral_norm",
    "noisy_gate_fidelity",
    "infidelity_contribution",
    "pct_of_total",
    "faithful_gate_fidelity",
    "faithful_infidelity_contribution",
    "faithful_pct_of_total",
    "close_grape_noisy_gate_fidelity",
    "close_grape_infidelity_contribution",
    "close_grape_pct_of_total",
    "close_grape_faithful_gate_fidelity",
    "close_grape_faithful_infidelity_contribution",
    "close_grape_faithful_pct_of_total",
)

# Contribution column -> the column holding its share of the all-terms total.
PCT_COLUMNS = {
    "infidelity_contribution": "pct_of_total",
    "faithful_infidelity_contribution": "faithful_pct_of_total",
    "close_grape_infidelity_contribution": "close_grape_pct_of_total",
    "close_grape_faithful_infidelity_contribution": "close_grape_faithful_pct_of_total",
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the per-noise-source infidelity budget of a pulse: "
            "open-system gate fidelity with each noise term active alone."
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
            "Optional second pulse .npz evaluated against the same per-source "
            "budget and drawn in the same figure, labeled 'close-grape'."
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
            "Also compute faithful_gate_fidelity per source (exact Lindblad "
            "propagation with Gauss-Hermite averaging; hermite_points nodes "
            "per single-fluctuation-term source)."
        ),
    )
    parser.add_argument(
        "--hermite-points",
        type=int,
        default=5,
        help="Gauss-Hermite nodes per fluctuation dimension for --faithful (default: 5).",
    )
    parser.add_argument(
        "--validity",
        action="store_true",
        help=(
            "Also run the perturbative-method validity budget (W/dW/V/"
            "truncation) into a validity/ subfolder."
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


def noise_sources(open_system):
    """(label, single-term OpenSystem) per noise term of the nominal system."""
    sources = []
    for term in open_system.noise_terms:
        if isinstance(term, FluctuationTerm):
            kind = f"fluctuation-{term.kind}"
            strength = term.coefficient
        elif isinstance(term, DecoherenceChannel):
            kind = "decoherence"
            strength = term.rate
        else:
            kind = type(term).__name__
            strength = float("nan")
        sources.append(
            {
                "name": term.name,
                "kind": kind,
                "strength": float(strength),
                "scaled_spectral_norm": float(np.linalg.norm(term.matrix, ord=2)),
                "system": replace(open_system, noise_terms=(term,)),
            }
        )
    return sources


def evaluate_budget(open_system, sources, pulse, state_pairs, closed_fidelity, args, workers):
    """One budget row per source plus the all-terms total, for one pulse."""

    def evaluate(system):
        noisy = float(
            noisy_gate_fidelity(
                system,
                pulse,
                state_pairs,
                collapse_operators=system.collapse_operators,
                n_workers=workers,
            )
        )
        faithful = None
        if args.faithful:
            faithful = float(
                faithful_gate_fidelity(
                    system, pulse, state_pairs, hermite_points=args.hermite_points
                )
            )
        return noisy, faithful

    rows = []
    for source in sources:
        noisy, faithful = evaluate(source["system"])
        rows.append(
            {
                "source": source["name"],
                "kind": source["kind"],
                "strength": source["strength"],
                "scaled_spectral_norm": source["scaled_spectral_norm"],
                "noisy_gate_fidelity": noisy,
                "infidelity_contribution": closed_fidelity - noisy,
                "faithful_gate_fidelity": faithful,
                "faithful_infidelity_contribution": (
                    None if faithful is None else closed_fidelity - faithful
                ),
            }
        )
        print(
            f"  {source['name']:>14}: noisy={noisy:.12f}"
            + ("" if faithful is None else f" faithful={faithful:.12f}"),
            flush=True,
        )
    noisy, faithful = evaluate(open_system)
    rows.append(
        {
            "source": "all",
            "kind": "total",
            "strength": None,
            "scaled_spectral_norm": None,
            "noisy_gate_fidelity": noisy,
            "infidelity_contribution": closed_fidelity - noisy,
            "faithful_gate_fidelity": faithful,
            "faithful_infidelity_contribution": (
                None if faithful is None else closed_fidelity - faithful
            ),
        }
    )
    print(
        f"  {'all':>14}: noisy={noisy:.12f}"
        + ("" if faithful is None else f" faithful={faithful:.12f}"),
        flush=True,
    )
    return rows


def add_percentages(rows):
    """Store each contribution's share of the all-terms total in every row."""
    total = next(row for row in rows if row["source"] == "all")
    for contribution_column, pct_column in PCT_COLUMNS.items():
        denominator = total.get(contribution_column)
        for row in rows:
            value = row.get(contribution_column)
            row[pct_column] = (
                None
                if value is None or not denominator
                else 100.0 * value / denominator
            )
    return rows


def additivity_residual(rows):
    """(F_closed - F_all) - sum of per-source contributions."""
    total = next(row for row in rows if row["source"] == "all")
    per_source = [row for row in rows if row["source"] != "all"]
    return total["infidelity_contribution"] - sum(
        row["infidelity_contribution"] for row in per_source
    )


def merge_comparison(rows, close_grape_rows):
    for row, comparison in zip(rows, close_grape_rows):
        row["close_grape_noisy_gate_fidelity"] = comparison["noisy_gate_fidelity"]
        row["close_grape_infidelity_contribution"] = comparison["infidelity_contribution"]
        row["close_grape_faithful_gate_fidelity"] = comparison["faithful_gate_fidelity"]
        row["close_grape_faithful_infidelity_contribution"] = comparison[
            "faithful_infidelity_contribution"
        ]
    return rows


def write_rows(csv_path, rows):
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    column: (
                        ""
                        if row.get(column) is None
                        else (
                            row[column]
                            if isinstance(row[column], str)
                            else f"{row[column]:.12g}"
                        )
                    )
                    for column in CSV_COLUMNS
                }
            )


def plot_budget(rows, output_path, *, compare, close_grape_time_us, pulse_time_us):
    import matplotlib.pyplot as plt

    def positive(values):
        dropped = sum(1 for value in values if value is not None and value <= 0.0)
        if dropped:
            print(
                f"warning: {dropped} non-positive contribution(s) cannot be "
                "drawn on the log axis and were dropped.",
                file=sys.stderr,
                flush=True,
            )
        return [
            (index, value)
            for index, value in enumerate(values)
            if value is not None and value > 0.0
        ]

    # Rows ordered largest contribution first, with the all-terms total on top.
    ordered = sorted(
        (row for row in rows if row["source"] != "all"),
        key=lambda row: row["infidelity_contribution"],
    ) + [next(row for row in rows if row["source"] == "all")]
    labels = [row["source"] for row in ordered]
    y = np.arange(len(ordered))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    # Color encodes the pulse (as in run_robustness_eval's figure); a 2px-ish
    # offset gap separates the two series' bars.
    series = [
        (
            "infidelity_contribution",
            f"noisy budget (T={pulse_time_us:.4g} us)",
            "tab:blue",
            0.19 if compare else 0.0,
        )
    ]
    if compare:
        series.append(
            (
                "close_grape_infidelity_contribution",
                f"close-grape budget (T={close_grape_time_us:.4g} us)",
                "tab:orange",
                -0.19,
            )
        )
    height = 0.36 if compare else 0.6
    for column, label, color, offset in series:
        points = positive([row.get(column) for row in ordered])
        ax.barh(
            [y[index] + offset for index, _ in points],
            [value for _, value in points],
            height=height,
            color=color,
            label=label,
        )
        for index, value in points:
            pct = ordered[index].get(PCT_COLUMNS[column])
            ax.annotate(
                f"{value:.2e}" + ("" if pct is None else f" ({pct:.3g}%)"),
                (value, y[index] + offset),
                xytext=(4, 0),
                textcoords="offset points",
                va="center",
                fontsize=8,
                color="dimgray",
            )
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xscale("log")
    ax.set_xlabel("infidelity contribution (F_closed - F_only_source)")
    ax.grid(True, alpha=0.3, axis="x", which="both")
    ax.legend(loc="lower right")
    ax.set_title("Per-noise-source error budget at nominal strengths")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_report(
    report_path,
    *,
    generated_at,
    config,
    args,
    rows,
    closed_fidelity,
    close_grape_closed_fidelity,
    close_grape_time_us,
    validity_outputs,
    wall_s,
):
    def fmt(column, value):
        if value is None:
            return "not computed"
        if column in PCT_COLUMNS.values():
            return f"{value:.4g}%"
        return f"{value:.12g}"

    compare = close_grape_closed_fidelity is not None
    lines = [
        "# Error Budget: Per-Noise-Source Infidelity at Nominal Strengths",
        "",
        f"Generated at: {generated_at.isoformat(timespec='seconds')}",
        "",
        "Each row evaluates the pulse with only that noise term active "
        "(`noisy_gate_fidelity`; strengths as configured). The contribution "
        "is `F_closed - F_only_source`; at leading order the contributions "
        "add up to the all-terms row, and the residual below measures how "
        "well they do. `pct_of_total` states each contribution as a "
        "percentage of the all-terms total `F_closed - F_all`, so the "
        "per-source shares sum to 100% minus that residual.",
        "",
        "## Run Summary",
        "",
        "| Parameter | Value |",
        "| --- | --- |",
        f"| config | {args.config} |",
        f"| pulse_npz | {args.pulse_npz} |",
        f"| close_grape_pulse_npz | {args.close_grape_pulse_npz or 'disabled'} |",
        f"| system_type | {config.system.type} |",
        f"| n_steps | {config.pulse.n_steps} |",
        f"| total_time_us | {config.pulse.total_time_us} |",
        *(
            [f"| close_grape_total_time_us | {close_grape_time_us:.6g} |"]
            if compare and close_grape_time_us is not None
            else []
        ),
        f"| faithful | {args.faithful} |",
        f"| hermite_points | {args.hermite_points if args.faithful else 'NA'} |",
        f"| closed_gate_fidelity | {closed_fidelity:.12g} |",
        *(
            [f"| close_grape_closed_gate_fidelity | {close_grape_closed_fidelity:.12g} |"]
            if compare
            else []
        ),
        f"| wall_s | {wall_s:.1f} |",
        "",
        "## Budget",
        "",
    ]
    columns = [
        "strength",
        "scaled_spectral_norm",
        "noisy_gate_fidelity",
        "infidelity_contribution",
        "pct_of_total",
    ]
    if args.faithful:
        columns += [
            "faithful_gate_fidelity",
            "faithful_infidelity_contribution",
            "faithful_pct_of_total",
        ]
    if compare:
        columns += [
            "close_grape_noisy_gate_fidelity",
            "close_grape_infidelity_contribution",
            "close_grape_pct_of_total",
        ]
        if args.faithful:
            columns += [
                "close_grape_faithful_gate_fidelity",
                "close_grape_faithful_infidelity_contribution",
                "close_grape_faithful_pct_of_total",
            ]
    lines += [
        "| source | kind | " + " | ".join(columns) + " |",
        "| --- | --- |" + " --- |" * len(columns),
    ]
    for row in rows:
        lines.append(
            f"| {row['source']} | {row['kind']} | "
            + " | ".join(fmt(column, row.get(column)) for column in columns)
            + " |"
        )
    lines += [
        "",
        "## Additivity",
        "",
        "| pulse | (F_closed - F_all) - sum of contributions |",
        "| --- | --- |",
        f"| pulse_npz | {additivity_residual(rows):.6g} |",
    ]
    if compare:
        comparison_rows = [
            {
                "source": row["source"],
                "infidelity_contribution": row["close_grape_infidelity_contribution"],
            }
            for row in rows
        ]
        lines.append(
            f"| close_grape_pulse_npz | {additivity_residual(comparison_rows):.6g} |"
        )
    if validity_outputs is not None:
        lines += [
            "",
            "## Validity Budget",
            "",
            f"Perturbative-method validity metrics: [{validity_outputs['markdown'].name}]"
            f"(validity/{validity_outputs['markdown'].name})",
        ]
    lines += [
        "",
        "## Figure",
        "",
        "![error_budget](error_budget.png)",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    config = _load_base_config(args.config)
    if args.workers is not None:
        config = replace(config, runtime=replace(config.runtime, workers=args.workers))
    workers = config.runtime.workers

    reference_pulse = build_initial_pulse(config)
    parameterization = build_parameterization(config, reference_pulse)
    pulse, _ = load_pulse(args.pulse_npz, reference_pulse, parameterization)
    close_grape_pulse = None
    close_grape_time_us = None
    if args.close_grape_pulse_npz is not None:
        close_grape_pulse, _ = load_comparison_pulse(args.close_grape_pulse_npz, config)
        close_grape_time_us = close_grape_pulse.n_steps * close_grape_pulse.dt * 1e6
    state_pairs = build_state_pairs(config)

    closed_system, open_system = build_systems(config)
    if not open_system.noise_terms:
        raise SystemExit("the config enables no noise terms; there is no budget to evaluate.")
    sources = noise_sources(open_system)

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
    run_dir = timestamped_experiment_dir(output_root, "error_budget", generated_at)
    run_dir.mkdir(parents=True, exist_ok=True)
    write_config_snapshot(config, run_dir / "config.yaml")

    print(
        f"run_dir={run_dir}\n"
        f"system={config.system.type}, n_steps={config.pulse.n_steps}, "
        f"n_state_pairs={len(state_pairs)}, n_noise_terms={len(sources)}\n"
        f"closed_gate_fidelity={closed_fidelity:.12f}"
        + (
            ""
            if close_grape_closed_fidelity is None
            else f"\nclose-grape closed_gate_fidelity={close_grape_closed_fidelity:.12f}"
        ),
        flush=True,
    )

    start = time.perf_counter()
    print("pulse budget:", flush=True)
    rows = evaluate_budget(
        open_system, sources, pulse, state_pairs, closed_fidelity, args, workers
    )
    if close_grape_pulse is not None:
        print("close-grape budget:", flush=True)
        close_grape_rows = evaluate_budget(
            open_system,
            sources,
            close_grape_pulse,
            state_pairs,
            close_grape_closed_fidelity,
            args,
            workers,
        )
        rows = merge_comparison(rows, close_grape_rows)
    rows = add_percentages(rows)

    validity_outputs = None
    if args.validity:
        print("validity budget (W/dW/V/truncation):", flush=True)
        validity_report = evaluate_error_budget(
            open_system, pulse, state_pairs, ErrorBudgetConfig()
        )
        validity_outputs = write_error_budget_report(validity_report, run_dir / "validity")
        print(f"  validity_md={validity_outputs['markdown']}", flush=True)
    wall_s = time.perf_counter() - start

    csv_path = run_dir / "error_budget.csv"
    write_rows(csv_path, rows)
    plot_path = run_dir / "error_budget.png"
    plot_budget(
        rows,
        plot_path,
        compare=close_grape_pulse is not None,
        close_grape_time_us=close_grape_time_us,
        pulse_time_us=config.pulse.total_time_us,
    )
    report_path = run_dir / "report.md"
    write_report(
        report_path,
        generated_at=generated_at,
        config=config,
        args=args,
        rows=rows,
        closed_fidelity=closed_fidelity,
        close_grape_closed_fidelity=close_grape_closed_fidelity,
        close_grape_time_us=close_grape_time_us,
        validity_outputs=validity_outputs,
        wall_s=wall_s,
    )
    for path in (csv_path, plot_path, report_path):
        if not path.exists() or path.stat().st_size == 0:
            raise RuntimeError(f"Expected non-empty output at {path}.")
    print(f"error_budget_csv={csv_path}")
    print(f"error_budget_plot={plot_path}")
    print(f"report={report_path}")


if __name__ == "__main__":
    main()
