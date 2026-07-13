"""Recursively shrink ``pulse.total_time_us`` and re-optimize each round.

Starting from any experiment YAML (e.g. ``experiments/spin_boson/configs/example.yaml``),
each round multiplies the total pulse time by ``--shrink-factor`` and re-runs
the optimization warm-started from the previous round's optimized pulse
(``n_steps`` stays fixed, so only ``dt`` shrinks and the amplitudes carry over).
The loop stops when the open (noisy) gate fidelity can no longer increase,
i.e. the first round whose ``final_noisy_gate_fidelity`` falls more than
``--fidelity-drop-tolerance`` below the previous round's, or after
``--max-rounds`` rounds.

Run with:
  python -m experiments.spin_boson.time_collapsing.run_time_collapsing \
      --config experiments/spin_boson/configs/example.yaml
"""

import argparse
import csv
from dataclasses import replace
from pathlib import Path

from experiments.driver.reporting import _markdown_table, timestamped_experiment_dir
from experiments.driver.run_experiment import (
    _load_base_config,
    resolved_output_root,
    run_perturbative_experiment,
)

import matplotlib.pyplot as plt

SUMMARY_FIELDS = (
    "round",
    "total_time_us",
    "final_noisy_gate_fidelity",
    "final_close_gate_fidelity",
    "final_penalized_objective",
    "optimizer_iterations",
    "optimizer_success",
    "round_dir",
    "final_pulse_npz",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Recursively shrink total_time_us, re-optimizing the pulse each "
            "round until the noisy gate fidelity stops increasing."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="YAML experiment configuration file (any registered system).",
    )
    parser.add_argument(
        "--shrink-factor",
        type=float,
        default=0.9,
        help="Multiply total_time_us by this factor each round; in (0, 1).",
    )
    parser.add_argument(
        "--fidelity-drop-tolerance",
        type=float,
        default=1e-4,
        help=(
            "Stop once a round's final noisy gate fidelity falls more than "
            "this below the previous round's."
        ),
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=20,
        help="Safety cap on the number of shrink rounds.",
    )
    parser.add_argument(
        "--maxiter",
        type=int,
        default=None,
        help="Override optimizer.maxiter from the base config.",
    )
    args = parser.parse_args(argv)
    if not 0.0 < args.shrink_factor < 1.0:
        parser.error("--shrink-factor must be strictly between 0 and 1.")
    if args.max_rounds < 1:
        parser.error("--max-rounds must be at least 1.")
    if args.fidelity_drop_tolerance < 0.0:
        parser.error("--fidelity-drop-tolerance must be non-negative.")
    return args


def _round_config(base, total_time_us, warm_start_npz):
    return replace(
        base,
        pulse=replace(base.pulse, total_time_us=total_time_us),
        runtime=replace(base.runtime, initial_pulse_npz=warm_start_npz),
    )


def _write_summary_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_summary_markdown(path, configuration, rows, best_row, stop_reason):
    table_rows = [
        (
            row["round"],
            f"{row['total_time_us']:.4f}",
            f"{row['final_noisy_gate_fidelity']:.8f}",
            f"{row['final_close_gate_fidelity']:.8f}",
            f"{row['final_penalized_objective']:.8f}",
            row["optimizer_iterations"],
            row["optimizer_success"],
            row["round_dir"],
        )
        for row in rows
    ]
    lines = [
        "# Time-collapsing optimization summary",
        "",
        "## Configuration",
        "",
        _markdown_table(("Parameter", "Value"), configuration),
        "",
        "## Rounds",
        "",
        _markdown_table(
            (
                "Round",
                "Total time (us)",
                "Noisy gate fidelity",
                "Closed gate fidelity",
                "Penalized objective",
                "Iterations",
                "Converged",
                "Run directory",
            ),
            table_rows,
        ),
        "",
        "## Outcome",
        "",
        f"Stop reason: {stop_reason}",
        "",
        f"Best round: {best_row['round']} at "
        f"{best_row['total_time_us']:.4f} us with noisy gate fidelity "
        f"{best_row['final_noisy_gate_fidelity']:.8f}",
        "",
        f"Best pulse: {best_row['final_pulse_npz']}",
        "",
    ]
    path.write_text("\n".join(lines))
    return path


def _plot_fidelity_vs_time(path, rows, best_row):
    times = [row["total_time_us"] for row in rows]
    noisy = [row["final_noisy_gate_fidelity"] for row in rows]
    closed = [row["final_close_gate_fidelity"] for row in rows]
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.plot(times, noisy, marker="o", label="noisy gate fidelity")
    ax.plot(times, closed, marker="s", label="closed gate fidelity")
    ax.axvline(
        best_row["total_time_us"],
        color="tab:gray",
        linestyle="--",
        label=f"best round ({best_row['total_time_us']:.2f} us)",
    )
    ax.set_xlabel("total pulse time (us)")
    ax.set_ylabel("gate fidelity")
    ax.invert_xaxis()
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def run_time_collapsing(
    config_path,
    shrink_factor=0.9,
    fidelity_drop_tolerance=1e-4,
    max_rounds=20,
    maxiter=None,
    total_time_us=None,
    initial_pulse_npz=None,
    top_dir=None,
):
    """Shrink-and-reoptimize loop; see the module docstring.

    ``total_time_us`` / ``initial_pulse_npz`` override the base config's
    starting time and warm-start pulse; ``top_dir`` replaces the timestamped
    output directory with a fixed one (idempotent reruns, e.g. Slurm tasks).
    """
    base = _load_base_config(config_path)
    if maxiter is not None:
        base = replace(base, optimizer=replace(base.optimizer, maxiter=maxiter))
    if total_time_us is not None:
        base = replace(base, pulse=replace(base.pulse, total_time_us=total_time_us))
    if initial_pulse_npz is not None:
        base = replace(
            base, runtime=replace(base.runtime, initial_pulse_npz=initial_pulse_npz)
        )
    if top_dir is None:
        top_dir = timestamped_experiment_dir(
            resolved_output_root(base), "time_collapsing"
        )
    else:
        top_dir = Path(top_dir)
    top_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    previous_fidelity = None
    warm_start_npz = base.runtime.initial_pulse_npz
    total_time_us = base.pulse.total_time_us
    stop_reason = f"max rounds ({max_rounds}) reached"
    for round_index in range(max_rounds):
        round_dir = top_dir / f"round_{round_index:02d}_T{total_time_us:.2f}us"
        print(
            f"round {round_index}: optimizing at total_time_us={total_time_us:.4f}",
            flush=True,
        )
        outcome = run_perturbative_experiment(
            _round_config(base, total_time_us, warm_start_npz),
            experiment_dir=round_dir,
            print_report=False,
        )
        fidelity = outcome["metrics"]["final_noisy_gate_fidelity"]
        rows.append(
            {
                "round": round_index,
                "total_time_us": total_time_us,
                "final_noisy_gate_fidelity": fidelity,
                "final_close_gate_fidelity": outcome["metrics"][
                    "final_close_gate_fidelity"
                ],
                "final_penalized_objective": outcome["metrics"][
                    "final_penalized_objective"
                ],
                "optimizer_iterations": outcome["result"].nit,
                "optimizer_success": outcome["result"].success,
                "round_dir": outcome["experiment_dir"],
                "final_pulse_npz": outcome["outputs"]["final_pulse_npz"],
            }
        )
        print(
            f"round {round_index}: final_noisy_gate_fidelity={fidelity:.8f}",
            flush=True,
        )
        if outcome["interrupted"]:
            stop_reason = f"optimization interrupted during round {round_index}"
            break
        if (
            previous_fidelity is not None
            and fidelity < previous_fidelity - fidelity_drop_tolerance
        ):
            stop_reason = (
                f"noisy gate fidelity stopped increasing at round {round_index} "
                f"({fidelity:.8f} < {previous_fidelity:.8f} - "
                f"{fidelity_drop_tolerance})"
            )
            break
        previous_fidelity = fidelity
        warm_start_npz = outcome["outputs"]["final_pulse_npz"]
        total_time_us *= shrink_factor

    best_row = max(rows, key=lambda row: row["final_noisy_gate_fidelity"])
    configuration = (
        ("base_config", config_path),
        ("shrink_factor", shrink_factor),
        ("fidelity_drop_tolerance", fidelity_drop_tolerance),
        ("max_rounds", max_rounds),
        ("n_steps", base.pulse.n_steps),
        ("initial_total_time_us", base.pulse.total_time_us),
        ("optimizer_maxiter", base.optimizer.maxiter),
    )
    summary_csv = _write_summary_csv(top_dir / "summary.csv", rows)
    summary_md = _write_summary_markdown(
        top_dir / "summary.md", configuration, rows, best_row, stop_reason
    )
    fidelity_plot = _plot_fidelity_vs_time(
        top_dir / "fidelity_vs_time.png", rows, best_row
    )
    print(f"stop_reason={stop_reason}")
    print(
        f"best_round={best_row['round']} "
        f"total_time_us={best_row['total_time_us']:.4f} "
        f"noisy_gate_fidelity={best_row['final_noisy_gate_fidelity']:.8f}"
    )
    print(f"best_pulse_npz={best_row['final_pulse_npz']}")
    print(f"summary_csv={summary_csv}")
    print(f"summary_md={summary_md}")
    print(f"fidelity_plot={fidelity_plot}")
    return {
        "experiment_dir": top_dir,
        "rounds": rows,
        "best_round": best_row,
        "stop_reason": stop_reason,
        "summary_csv": summary_csv,
        "summary_md": summary_md,
        "fidelity_plot": fidelity_plot,
    }


def main(argv=None):
    args = parse_args(argv)
    run_time_collapsing(
        args.config,
        shrink_factor=args.shrink_factor,
        fidelity_drop_tolerance=args.fidelity_drop_tolerance,
        max_rounds=args.max_rounds,
        maxiter=args.maxiter,
    )


if __name__ == "__main__":
    main()
