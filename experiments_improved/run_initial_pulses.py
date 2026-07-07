"""Run every initial-pulse .npz in a directory through the spin-boson experiment.

By default each pulse is *evaluated* (fast: no optimization) and a summary table
of close/noisy-gate fidelities is printed, sorted best-first. Pass ``--optimize``
to instead run the full L-BFGS-B optimization for each pulse and compare the
initial vs. final noisy-gate fidelity.

Usage:
    python experiments_improved/run_initial_pulses.py                 # evaluate all
    python experiments_improved/run_initial_pulses.py --optimize      # optimize all
    python experiments_improved/run_initial_pulses.py --workers 4 --maxiter 40
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from run_experiment import (
    OUTPUT_DIR,
    default_experiment_config,
    evaluate_pulse,
    run_perturbative_experiment,
)

DEFAULT_PULSE_DIR = OUTPUT_DIR / "initial_pulses"


def _config_for(pulse_npz, workers, maxiter):
    base = default_experiment_config()
    return replace(
        base,
        optimizer=replace(base.optimizer, maxiter=maxiter),
        runtime=replace(
            base.runtime,
            workers=workers,
            initial_pulse_npz=Path(pulse_npz),
            no_progress=True,
        ),
    )


def run_all(pulse_dir=DEFAULT_PULSE_DIR, workers=1, optimize=False, maxiter=40):
    pulse_dir = Path(pulse_dir)
    pulses = sorted(pulse_dir.glob("*.npz"))
    if not pulses:
        raise SystemExit(f"No .npz pulses found in {pulse_dir}. Run make_initial_pulses.py first.")

    rows = []
    for pulse_npz in pulses:
        config = _config_for(pulse_npz, workers, maxiter)
        print(f"[{'optimize' if optimize else 'evaluate'}] {pulse_npz.name} ...", flush=True)
        if optimize:
            result = run_perturbative_experiment(config, print_report=False)
            metrics = result["metrics"]
            rows.append(
                (
                    pulse_npz.name,
                    metrics["initial_noisy_gate_fidelity"],
                    metrics["final_noisy_gate_fidelity"],
                    result["experiment_dir"].name,
                )
            )
        else:
            result = evaluate_pulse(config, print_report=False)
            metrics = result["metrics"]
            rows.append(
                (
                    pulse_npz.name,
                    metrics["close_gate_fidelity"],
                    metrics["noisy_gate_fidelity"],
                    result["experiment_dir"].name,
                )
            )

    _print_summary(rows, optimize)
    return rows


def _print_summary(rows, optimize):
    if optimize:
        headers = ("pulse", "initial_noisy", "final_noisy", "experiment_dir")
        rows = sorted(rows, key=lambda r: r[2], reverse=True)
    else:
        headers = ("pulse", "close_gate", "noisy_gate", "experiment_dir")
        rows = sorted(rows, key=lambda r: r[2], reverse=True)
    name_w = max(len(headers[0]), *(len(r[0]) for r in rows))
    dir_w = max(len(headers[3]), *(len(r[3]) for r in rows))
    print(f"\n=== Summary ({'optimized' if optimize else 'evaluated'}, best first) ===")
    print(f"{headers[0]:<{name_w}}  {headers[1]:>13}  {headers[2]:>13}  {headers[3]:<{dir_w}}")
    for name, a, b, exp_dir in rows:
        print(f"{name:<{name_w}}  {a:>13.6f}  {b:>13.6f}  {exp_dir:<{dir_w}}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pulse-dir", type=Path, default=DEFAULT_PULSE_DIR)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--optimize", action="store_true", help="Optimize each pulse instead of only evaluating.")
    parser.add_argument("--maxiter", type=int, default=40, help="Optimizer iterations (only with --optimize).")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    run_all(args.pulse_dir, workers=args.workers, optimize=args.optimize, maxiter=args.maxiter)


if __name__ == "__main__":
    main()
