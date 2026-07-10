"""Four-way fidelity comparison along a spin-boson optimization trajectory.

Runs the optimization defined by ``spin-boson-origin.yaml`` and, every
``--eval-interval`` optimizer iterations (default 20), evaluates the current
pulse with the four fidelity calculations supported by the code base:

1. ``grape_raw_objective``  – the optimizer's own raw objective
   (``PenalizedParameterizedProblem.raw_value``, i.e. without penalties);
2. ``noisy_gate_fidelity``  – ``quantum_control.evaluation`` perturbative
   expansion plus first-order Lindblad correction;
3. ``closed_gate_fidelity`` – ``quantum_control.evaluation`` noise-free
   propagation;
4. ``faithful_gate_fidelity`` – ``quantum_control.evaluation`` exact
   propagation with Gauss-Hermite averaging over the fluctuations.

The first three are cheap and are computed inside the optimizer callback.
``faithful_gate_fidelity`` costs ``hermite_points ** n_fluctuation_terms``
superoperator propagations per pulse (~8 s per node for the origin config),
so it is evaluated once, on the final pulse only. Checkpoint pulses are
still exported every interval so a run can be resumed via
``--initial-pulse-npz``.

Results land in a timestamped directory under ``outputs/`` next to this
script: ``evaluations.csv``, ``evaluations.png``, ``report.md``, checkpoint
pulses, and a resolved config snapshot.

Run from the repository root:

    .venv/bin/python experiments/spin_boson/evaluation_check/run_evaluation_check.py \
        --config experiments/spin_boson/evaluation_check/spin-boson-origin.yaml
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
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np

from experiments.driver.run_experiment import (
    OptimizationProgressBar,
    _load_base_config,
    build_initial_pulse,
    build_objective_problem,
    build_optimizer,
    build_parameterization,
    build_state_pairs,
    build_systems,
    load_custom_initial_parameters,
    system_definition,
)
from experiments.driver.config_io import write_config_snapshot
from experiments.driver.reporting import export_pulse_controls, timestamped_experiment_dir
from quantum_control import (
    ParameterSmoothPenalty,
    ParameterizedControlProblem,
    PenalizedParameterizedProblem,
    closed_gate_fidelity,
    faithful_gate_fidelity,
    noisy_gate_fidelity,
)

CSV_COLUMNS = (
    "step",
    "grape_raw_objective",
    "noisy_gate_fidelity",
    "closed_gate_fidelity",
    "faithful_gate_fidelity",
    "penalized_objective",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Compare the four fidelity calculations along an optimization run."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=HERE / "spin-boson-origin.yaml",
        help="YAML experiment configuration (default: spin-boson-origin.yaml).",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=20,
        help="Evaluate the four fidelities every this many optimizer iterations.",
    )
    parser.add_argument(
        "--hermite-points",
        type=int,
        default=3,
        help=(
            "Gauss-Hermite nodes per fluctuation dimension for the final "
            "faithful_gate_fidelity check (cost: points ** n_fluctuation_terms "
            "nodes, ~8 s per node for the origin config)."
        ),
    )
    parser.add_argument(
        "--initial-pulse-npz",
        type=Path,
        default=None,
        help=(
            "Resume optimization from a pulse .npz with an amplitudes array "
            "(e.g. a checkpoints/pulse_step_XXXX.npz from a previous run)."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=HERE / "outputs",
        help="Directory that receives the timestamped run directory.",
    )
    parser.add_argument(
        "--maxiter",
        type=int,
        default=None,
        help="Override optimizer.maxiter (for quick smoke tests).",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=None,
        help="Override pulse.n_steps (for quick smoke tests).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Override runtime.workers (also used for the faithful phase).",
    )
    return parser.parse_args(argv)


def load_config(args):
    config = _load_base_config(args.config)
    if args.maxiter is not None:
        config = replace(config, optimizer=replace(config.optimizer, maxiter=args.maxiter))
    if args.n_steps is not None:
        config = replace(config, pulse=replace(config.pulse, n_steps=args.n_steps))
    if args.workers is not None:
        config = replace(config, runtime=replace(config.runtime, workers=args.workers))
    return config


def write_rows(csv_path, rows):
    """Rewrite the evaluation CSV; pending faithful values stay blank."""
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    column: ("" if row.get(column) is None else f"{row[column]:.12g}")
                    if column != "step"
                    else row["step"]
                    for column in CSV_COLUMNS
                }
            )


def plot_evaluations(rows, output_path, hermite_points):
    import matplotlib.pyplot as plt

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    series = (
        ("grape_raw_objective", "GRAPE raw objective", "o-"),
        ("noisy_gate_fidelity", "noisy_gate_fidelity", "s--"),
        ("closed_gate_fidelity", "closed_gate_fidelity", "^-"),
        ("faithful_gate_fidelity", f"faithful_gate_fidelity (hermite={hermite_points})", "D"),
    )

    def present(column, transform=lambda row: row[column]):
        pairs = [(row["step"], transform(row)) for row in rows if row.get(column) is not None]
        return [pair[0] for pair in pairs], [pair[1] for pair in pairs]

    for column, label, style in series:
        steps, values = present(column)
        if steps:
            top.plot(steps, values, style, label=label, markersize=5)
    top.set_ylabel("fidelity")
    top.legend(loc="best")
    top.grid(True, alpha=0.3)
    top.set_title("Four fidelity calculations along the optimization")

    reference = "noisy_gate_fidelity"
    for column, label, style in series:
        if column == reference:
            continue
        steps, deltas = present(
            column, lambda row, column=column: abs(row[column] - row[reference])
        )
        if steps:
            bottom.semilogy(steps, deltas, style, label=f"|{column} - {reference}|", markersize=5)
    bottom.set_xlabel("optimizer iteration")
    bottom.set_ylabel(f"absolute difference to {reference}")
    bottom.legend(loc="best")
    bottom.grid(True, alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_report(report_path, *, generated_at, config, args, rows, result, durations):
    def fmt(value):
        return "pending" if value is None else f"{value:.12g}"

    lines = [
        "# Evaluation Check: Four Fidelity Calculations",
        "",
        f"Generated at: {generated_at.isoformat(timespec='seconds')}",
        "",
        "Optimization from `spin-boson-origin.yaml`; every "
        f"{args.eval_interval} optimizer iterations the current pulse is "
        "evaluated with the GRAPE raw objective, `noisy_gate_fidelity`, and "
        "`closed_gate_fidelity`; the final pulse is additionally checked with "
        f"`faithful_gate_fidelity` (hermite_points={args.hermite_points}).",
        "",
        "## Run Summary",
        "",
        "| Parameter | Value |",
        "| --- | --- |",
        f"| system_type | {config.system.type} |",
        f"| initial_pulse | {args.initial_pulse_npz or 'generated (built-in)'} |",
        f"| n_steps | {config.pulse.n_steps} |",
        f"| total_time_us | {config.pulse.total_time_us} |",
        f"| maxiter | {config.optimizer.maxiter} |",
        f"| eval_interval | {args.eval_interval} |",
        f"| hermite_points | {args.hermite_points} |",
        f"| workers | {config.runtime.workers} |",
        f"| optimizer_success | {getattr(result, 'success', 'NA')} |",
        f"| optimizer_message | {getattr(result, 'message', 'NA')} |",
        f"| nit | {getattr(result, 'nit', 'NA')} |",
        f"| optimization_wall_s | {durations['optimization']:.1f} |",
        f"| faithful_phase_wall_s | {durations['faithful']:.1f} |",
        "",
        "## Evaluations",
        "",
        "| step | grape_raw_objective | noisy_gate_fidelity | closed_gate_fidelity | faithful_gate_fidelity | penalized_objective |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["step"]),
                    fmt(row.get("grape_raw_objective")),
                    fmt(row.get("noisy_gate_fidelity")),
                    fmt(row.get("closed_gate_fidelity")),
                    fmt(row.get("faithful_gate_fidelity")),
                    fmt(row.get("penalized_objective")),
                ]
            )
            + " |"
        )
    agreement = [
        abs(row["grape_raw_objective"] - row["noisy_gate_fidelity"])
        for row in rows
        if row.get("grape_raw_objective") is not None and row.get("noisy_gate_fidelity") is not None
    ]
    faithful_gaps = [
        abs(row["faithful_gate_fidelity"] - row["noisy_gate_fidelity"])
        for row in rows
        if row.get("faithful_gate_fidelity") is not None and row.get("noisy_gate_fidelity") is not None
    ]
    lines += [
        "",
        "## Agreement",
        "",
        "| Check | Max absolute difference |",
        "| --- | --- |",
    ]
    if agreement:
        lines.append(f"| grape_raw_objective vs noisy_gate_fidelity | {max(agreement):.3g} |")
    if faithful_gaps:
        lines.append(f"| faithful_gate_fidelity vs noisy_gate_fidelity | {max(faithful_gaps):.3g} |")
    lines += [
        "",
        "The GRAPE raw objective and `noisy_gate_fidelity` compute the same "
        "quantity through different code paths, so their difference should be "
        "at numerical noise level. `closed_gate_fidelity` ignores the noise "
        "model, and `faithful_gate_fidelity` is the exact reference; its gap "
        "to `noisy_gate_fidelity` measures the perturbative truncation error.",
        "",
        "## Figure",
        "",
        "![evaluations](evaluations.png)",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    config = load_config(args)

    definition = system_definition(config)
    channels = definition.control_channels(config.system.params)
    channel_names = tuple(channel.label for channel in channels)
    export_unit_divisor = 1.0 / channels[0].display_scale

    system, open_system = build_systems(config)
    initial_pulse = build_initial_pulse(config)
    parameterization = build_parameterization(config, initial_pulse)
    state_pairs = build_state_pairs(config)
    collapse_operators = open_system.collapse_operators

    optimization_problem = build_objective_problem(config, open_system, initial_pulse, state_pairs)
    parameterized_problem = ParameterizedControlProblem(optimization_problem, parameterization)
    penalty = ParameterSmoothPenalty(
        l1_weight=config.penalty.l1_smooth_weight,
        l2_weight=config.penalty.l2_smooth_weight,
    )
    penalized_problem = PenalizedParameterizedProblem(parameterized_problem, penalty)
    if args.initial_pulse_npz is not None:
        loaded_parameters, load_metadata = load_custom_initial_parameters(
            args.initial_pulse_npz, initial_pulse, parameterization
        )
        for warning in load_metadata["warnings"]:
            print(warning, file=sys.stderr, flush=True)
        initial_parameters = np.asarray(loaded_parameters, dtype=float).reshape(
            penalized_problem.parameter_shape
        )
    else:
        initial_parameters = penalized_problem.initial_parameters()

    generated_at = datetime.now()
    run_dir = timestamped_experiment_dir(args.output_root, "evaluation-check", generated_at)
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    write_config_snapshot(config, run_dir / "config.yaml")
    csv_path = run_dir / "evaluations.csv"

    rows = []
    checkpoints = []

    def evaluate_cheap(step, parameters):
        parameters = np.asarray(parameters, dtype=float).reshape(
            penalized_problem.parameter_shape
        )
        pulse = penalized_problem.pulse_from_parameters(parameters)
        row = {
            "step": int(step),
            "grape_raw_objective": float(penalized_problem.raw_value(parameters)),
            "noisy_gate_fidelity": float(
                noisy_gate_fidelity(
                    open_system,
                    pulse,
                    state_pairs,
                    collapse_operators=collapse_operators,
                    n_workers=config.runtime.workers,
                )
            ),
            "closed_gate_fidelity": float(closed_gate_fidelity(system, pulse, state_pairs)),
            "faithful_gate_fidelity": None,
            "penalized_objective": float(penalized_problem.value(parameters)),
        }
        rows.append(row)
        checkpoints.append((int(step), pulse))
        export_pulse_controls(
            pulse,
            checkpoint_dir / f"pulse_step_{int(step):04d}",
            export_unit_divisor,
            channel_names=channel_names,
        )
        write_rows(csv_path, rows)
        print(
            f"step {step:>4}: raw={row['grape_raw_objective']:.9f} "
            f"noisy={row['noisy_gate_fidelity']:.9f} "
            f"closed={row['closed_gate_fidelity']:.9f} (faithful pending)",
            flush=True,
        )
        return row

    step_counter = {"value": 0}
    latest_parameters = {"value": np.asarray(initial_parameters, dtype=float)}

    def step_callback(parameters):
        step_counter["value"] += 1
        latest_parameters["value"] = np.asarray(parameters, dtype=float)
        if step_counter["value"] % args.eval_interval == 0:
            evaluate_cheap(step_counter["value"], parameters)

    progress = None if config.runtime.no_progress else OptimizationProgressBar(config.optimizer.maxiter)

    def callback(parameters):
        if progress is not None:
            progress(parameters)
        step_callback(parameters)

    print(
        f"run_dir={run_dir}\n"
        f"n_steps={config.pulse.n_steps}, maxiter={config.optimizer.maxiter}, "
        f"eval_interval={args.eval_interval}, hermite_points={args.hermite_points}, "
        f"n_state_pairs={len(state_pairs)}, "
        f"n_fluctuation_terms={len(open_system.fluctuation_terms)}",
        flush=True,
    )

    evaluate_cheap(0, initial_parameters)
    optimizer = build_optimizer(config)
    if progress is not None:
        progress.start()
    optimization_start = time.perf_counter()
    try:
        result = optimizer.optimize_parameters(
            penalized_problem,
            initial_parameters=initial_parameters,
            callback=callback,
        )
        final_parameters = np.asarray(result.x, dtype=float)
        final_step = int(step_counter["value"])
    finally:
        optimization_problem.shutdown()
    optimization_wall = time.perf_counter() - optimization_start
    if progress is not None:
        progress.finish(result)
    if final_step % args.eval_interval != 0 or not rows or rows[-1]["step"] != final_step:
        evaluate_cheap(final_step, final_parameters)

    n_nodes = args.hermite_points ** len(open_system.fluctuation_terms)
    print(
        f"optimization done in {optimization_wall:.1f} s "
        f"(nit={getattr(result, 'nit', 'NA')}); running the faithful check on "
        f"the final pulse with hermite_points={args.hermite_points} "
        f"({n_nodes} nodes, ~{n_nodes * 8 / 60:.0f} min)...",
        flush=True,
    )

    faithful_start = time.perf_counter()
    final_step, final_pulse = checkpoints[-1]
    final_row = rows[-1]
    assert final_row["step"] == final_step
    final_row["faithful_gate_fidelity"] = float(
        faithful_gate_fidelity(
            open_system, final_pulse, state_pairs, hermite_points=args.hermite_points
        )
    )
    write_rows(csv_path, rows)
    faithful_wall = time.perf_counter() - faithful_start
    print(
        f"step {final_step:>4}: faithful={final_row['faithful_gate_fidelity']:.9f} "
        f"(noisy={final_row['noisy_gate_fidelity']:.9f}) in {faithful_wall:.1f} s",
        flush=True,
    )

    plot_evaluations(rows, run_dir / "evaluations.png", args.hermite_points)
    write_report(
        run_dir / "report.md",
        generated_at=generated_at,
        config=config,
        args=args,
        rows=rows,
        result=result,
        durations={"optimization": optimization_wall, "faithful": faithful_wall},
    )
    for path in (csv_path, run_dir / "evaluations.png", run_dir / "report.md"):
        if not path.exists() or path.stat().st_size == 0:
            raise RuntimeError(f"Expected non-empty output at {path}.")
    print(f"evaluations_csv={csv_path}")
    print(f"evaluations_plot={run_dir / 'evaluations.png'}")
    print(f"report={run_dir / 'report.md'}")


if __name__ == "__main__":
    main()
