from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.reporting import timestamped_experiment_dir
from experiments.spin_boson_perturbative_lbfgsb import (
    DEFAULT_L1_SMOOTH_WEIGHT,
    DEFAULT_L2_SMOOTH_WEIGHT,
    MAXITER,
    N_STEPS,
    OUTPUT_DIR,
    Alpha2EndpointZeroParameterization,
    run_perturbative_experiment,
)
from quantum_control import PiecewiseConstantPulse, spin_boson_initial_pulse, spin_boson_parameterization


SUMMARY_FIELDS = (
    "mode",
    "run_index",
    "seed",
    "source_npz",
    "source_dt",
    "experiment_dt",
    "dt_missing",
    "dt_mismatch",
    "interrupted",
    "success",
    "nit",
    "nfev",
    "initial_open_gate_fidelity",
    "final_open_gate_fidelity",
    "initial_close_gate_fidelity",
    "final_close_gate_fidelity",
    "initial_cost",
    "final_cost",
    "initial_l1_penalty",
    "final_l1_penalty",
    "initial_l2_penalty",
    "final_l2_penalty",
    "experiment_dir",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run perturbative open-gate optimizations from multiple initial pulses."
    )
    parser.add_argument(
        "--initial-mode",
        choices=("noise", "random", "custom", "both", "all"),
        default="all",
    )
    parser.add_argument("--n-runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--noise-scale", type=float, default=0.3)
    parser.add_argument(
        "--initial-pulse-npz",
        action="append",
        default=[],
        help="Path to a custom initial pulse .npz. May be repeated.",
    )
    parser.add_argument("--maxiter", type=int, default=MAXITER)
    parser.add_argument("--n-steps", type=int, default=N_STEPS)
    parser.add_argument("--alpha1-cycles", type=float, default=1.0)
    parser.add_argument("--l1-smooth-weight", type=float, default=DEFAULT_L1_SMOOTH_WEIGHT)
    parser.add_argument("--l2-smooth-weight", type=float, default=DEFAULT_L2_SMOOTH_WEIGHT)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--sweep-workers",
        type=int,
        default=1,
        help="Number of initial-pulse runs to execute in parallel.",
    )
    parser.add_argument("--print-step", action="store_true")
    parser.add_argument(
        "--print-fidelity-terms",
        action="store_true",
        help="Print and save per-step perturbative fidelity term diagnostics.",
    )
    parser.add_argument("--no-progress", action="store_true", default=True)
    return parser.parse_args()


def build_run_args(args):
    return SimpleNamespace(
        maxiter=args.maxiter,
        n_steps=args.n_steps,
        alpha1_cycles=args.alpha1_cycles,
        l1_smooth_weight=args.l1_smooth_weight,
        l2_smooth_weight=args.l2_smooth_weight,
        workers=args.workers,
        print_step=args.print_step,
        print_fidelity_terms=args.print_fidelity_terms,
        save_fidelity_terms=args.print_fidelity_terms,
        no_progress=args.no_progress,
        initial_pulse_npz=None,
    )


def reference_pulse_and_parameterization(n_steps, alpha1_cycles):
    pulse = spin_boson_initial_pulse(n_steps=n_steps, alpha1_cycles=alpha1_cycles)
    parameterization = Alpha2EndpointZeroParameterization(
        spin_boson_parameterization(pulse.n_steps)
    )
    return pulse, parameterization


def noise_initial_parameters(base_parameters, rng, noise_scale):
    return np.clip(
        np.asarray(base_parameters, dtype=float)
        + rng.normal(0.0, float(noise_scale), np.shape(base_parameters)),
        -1.0,
        1.0,
    )


def random_initial_parameters(shape, rng):
    return rng.uniform(-1.0, 1.0, size=shape)


def load_custom_initial_parameters(npz_path, reference_pulse, parameterization, atol=1e-9):
    npz_path = Path(npz_path)
    with np.load(npz_path) as data:
        if "amplitudes" not in data.files:
            raise ValueError(f"{npz_path} does not contain required 'amplitudes'.")
        amplitudes = np.asarray(data["amplitudes"], dtype=float)
        source_dt = float(np.asarray(data["dt"]).reshape(())) if "dt" in data.files else None

    if amplitudes.shape != reference_pulse.amplitudes.shape:
        raise ValueError(
            f"{npz_path} amplitudes shape {amplitudes.shape} does not match "
            f"expected {reference_pulse.amplitudes.shape}."
        )
    if not np.allclose(amplitudes[[0, -1], 1], 0.0, atol=atol):
        raise ValueError(f"{npz_path} alpha2 endpoints must be zero.")

    parameters = parameterization.to_parameters(amplitudes)
    if np.any(parameters < -1.0 - atol) or np.any(parameters > 1.0 + atol):
        raise ValueError(f"{npz_path} amplitudes exceed the configured parameter bounds.")
    parameters = np.clip(parameters, -1.0, 1.0)

    experiment_dt = float(reference_pulse.dt)
    dt_missing = source_dt is None
    dt_mismatch = False if source_dt is None else not np.isclose(source_dt, experiment_dt)
    warnings = []
    if dt_missing:
        warnings.append(f"warning: {npz_path} has no dt; using experiment dt {experiment_dt:.12g}.")
    elif dt_mismatch:
        warnings.append(
            f"warning: {npz_path} dt={source_dt:.12g} differs from experiment "
            f"dt={experiment_dt:.12g}; using experiment dt."
        )

    return parameters, {
        "source_npz": str(npz_path),
        "source_dt": "NA" if source_dt is None else source_dt,
        "experiment_dt": experiment_dt,
        "dt_missing": dt_missing,
        "dt_mismatch": dt_mismatch,
        "warnings": warnings,
    }


def modes_to_run(initial_mode, custom_paths):
    if initial_mode == "both":
        modes = ("noise", "random")
    elif initial_mode == "all":
        modes = ("noise", "random", "custom")
    else:
        modes = (initial_mode,)
    if "custom" in modes and not custom_paths:
        if initial_mode == "custom":
            raise ValueError("--initial-mode custom requires at least one --initial-pulse-npz.")
        modes = tuple(mode for mode in modes if mode != "custom")
    return modes


def safe_label(value):
    label = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return label or "pulse"


def summary_row(mode, run_index, seed, source_metadata, run_result):
    metrics = run_result["metrics"]
    result = run_result["result"]
    source_metadata = source_metadata or {}
    return {
        "mode": mode,
        "run_index": run_index,
        "seed": "NA" if seed is None else seed,
        "source_npz": source_metadata.get("source_npz", "NA"),
        "source_dt": source_metadata.get("source_dt", "NA"),
        "experiment_dt": source_metadata.get("experiment_dt", "NA"),
        "dt_missing": source_metadata.get("dt_missing", False),
        "dt_mismatch": source_metadata.get("dt_mismatch", False),
        "interrupted": run_result.get("interrupted", False),
        "success": result.success,
        "nit": getattr(result, "nit", "NA"),
        "nfev": getattr(result, "nfev", "NA"),
        "initial_open_gate_fidelity": metrics["initial_open_gate_fidelity"],
        "final_open_gate_fidelity": metrics["final_open_gate_fidelity"],
        "initial_close_gate_fidelity": metrics["initial_close_gate_fidelity"],
        "final_close_gate_fidelity": metrics["final_close_gate_fidelity"],
        "initial_cost": metrics["initial_penalized_objective"],
        "final_cost": metrics["final_penalized_objective"],
        "initial_l1_penalty": metrics["initial_l1_penalty"],
        "final_l1_penalty": metrics["final_l1_penalty"],
        "initial_l2_penalty": metrics["initial_l2_penalty"],
        "final_l2_penalty": metrics["final_l2_penalty"],
        "experiment_dir": run_result["experiment_dir"],
    }


def build_sweep_run_specs(args, sweep_dir, generated_at):
    reference_pulse, parameterization = reference_pulse_and_parameterization(
        args.n_steps,
        args.alpha1_cycles,
    )
    base_parameters = parameterization.to_parameters(reference_pulse.amplitudes)
    specs = []
    run_index = 0

    for mode in modes_to_run(args.initial_mode, args.initial_pulse_npz):
        if mode in {"noise", "random"}:
            for index in range(args.n_runs):
                seed = args.seed + index + (args.n_runs if mode == "random" else 0)
                rng = np.random.default_rng(seed)
                parameters = (
                    noise_initial_parameters(base_parameters, rng, args.noise_scale)
                    if mode == "noise"
                    else random_initial_parameters(base_parameters.shape, rng)
                )
                run_index += 1
                label = f"{mode}_seed_{seed}"
                specs.append(
                    {
                        "mode": mode,
                        "run_index": run_index,
                        "seed": seed,
                        "parameters": parameters,
                        "label": label,
                        "experiment_dir": sweep_dir / label,
                        "generated_at": generated_at,
                        "source_metadata": None,
                        "extra_configuration": [
                            ("sweep_mode", mode),
                            ("sweep_seed", seed),
                            ("noise_scale", args.noise_scale if mode == "noise" else "NA"),
                        ],
                    }
                )
        elif mode == "custom":
            for index, npz_path in enumerate(args.initial_pulse_npz, start=1):
                parameters, metadata = load_custom_initial_parameters(
                    npz_path,
                    reference_pulse,
                    parameterization,
                )
                for warning in metadata["warnings"]:
                    print(warning, file=sys.stderr, flush=True)
                run_index += 1
                label = f"custom_{index}_{safe_label(Path(npz_path).stem)}"
                specs.append(
                    {
                        "mode": mode,
                        "run_index": run_index,
                        "seed": None,
                        "parameters": parameters,
                        "label": label,
                        "experiment_dir": sweep_dir / label,
                        "generated_at": generated_at,
                        "source_metadata": metadata,
                        "extra_configuration": [
                            ("sweep_mode", mode),
                            ("source_npz", metadata["source_npz"]),
                            ("source_dt", metadata["source_dt"]),
                            ("dt_missing", metadata["dt_missing"]),
                            ("dt_mismatch", metadata["dt_mismatch"]),
                        ],
                    }
                )
    return specs


def _execute_sweep_run(run_args, spec):
    run_result = run_perturbative_experiment(
        run_args,
        initial_parameters=spec["parameters"],
        run_label=spec["label"],
        experiment_dir=spec["experiment_dir"],
        generated_at=spec["generated_at"],
        extra_configuration=spec["extra_configuration"],
        print_report=False,
    )
    return summary_row(
        spec["mode"],
        spec["run_index"],
        spec["seed"],
        spec["source_metadata"],
        run_result,
    )


def _write_summaries(sweep_dir, rows):
    rows = sorted(rows, key=lambda row: int(row["run_index"]))
    write_summary_csv(sweep_dir / "summary.csv", rows)
    write_summary_markdown(sweep_dir / "summary.md", rows)
    return rows


def write_summary_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "NA") for field in SUMMARY_FIELDS})


def write_summary_markdown(path, rows):
    path = Path(path)
    lines = ["# Perturbative Initial-Pulse Sweep", ""]
    lines.extend(_summary_table("Top Runs by Final Cost", rows, "final_cost"))
    lines.extend([""])
    lines.extend(_summary_table("Top Runs by Final Open Gate Fidelity", rows, "final_open_gate_fidelity"))
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _summary_table(title, rows, sort_field, limit=10):
    lines = [f"## {title}", ""]
    headers = tuple(
        dict.fromkeys(
            (
                "mode",
                "run_index",
                "seed",
                sort_field,
                "final_cost",
                "final_open_gate_fidelity",
                "experiment_dir",
            )
        )
    )
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in sorted(rows, key=lambda item: float(item[sort_field]), reverse=True)[:limit]:
        lines.append("| " + " | ".join(str(row.get(header, "NA")) for header in headers) + " |")
    return lines


def run_sweep(args):
    if args.n_runs < 1:
        raise ValueError("--n-runs must be at least 1.")
    if args.noise_scale < 0.0:
        raise ValueError("--noise-scale must be non-negative.")
    if args.sweep_workers < 1:
        raise ValueError("--sweep-workers must be at least 1.")

    run_args = build_run_args(args)
    generated_at = datetime.now()
    sweep_dir = timestamped_experiment_dir(
        OUTPUT_DIR,
        "spin_boson_perturbative_sweep",
        generated_at,
    )
    sweep_dir.mkdir(parents=True, exist_ok=True)
    specs = build_sweep_run_specs(args, sweep_dir, generated_at)

    if args.sweep_workers > 1 and args.print_step:
        print(
            "warning: --print-step console tables are disabled while "
            "--sweep-workers > 1; per-step logs are still saved in each run directory.",
            file=sys.stderr,
            flush=True,
        )
        run_args = SimpleNamespace(**vars(run_args))
        run_args.print_step = False
    if args.sweep_workers > 1 and args.print_fidelity_terms:
        print(
            "warning: --print-fidelity-terms console tables are disabled while "
            "--sweep-workers > 1; fidelity diagnostics are still saved in each run directory.",
            file=sys.stderr,
            flush=True,
        )
        run_args = SimpleNamespace(**vars(run_args))
        run_args.print_fidelity_terms = False
        run_args.save_fidelity_terms = True

    rows = []
    try:
        if args.sweep_workers == 1:
            for spec in specs:
                print(f"running {spec['label']}", flush=True)
                row = _execute_sweep_run(run_args, spec)
                rows.append(row)
                if row.get("interrupted", False):
                    print(
                        "sweep interrupted; summary/report written with latest pulse.",
                        file=sys.stderr,
                        flush=True,
                    )
                    break
        else:
            print(
                f"running {len(specs)} runs with sweep_workers={args.sweep_workers}",
                flush=True,
            )
            with ProcessPoolExecutor(max_workers=args.sweep_workers) as executor:
                futures = {
                    executor.submit(_execute_sweep_run, run_args, spec): spec
                    for spec in specs
                }
                for future in as_completed(futures):
                    spec = futures[future]
                    row = future.result()
                    rows.append(row)
                    print(
                        f"completed {spec['label']} "
                        f"final_cost={float(row['final_cost']):.12g} "
                        f"final_open={float(row['final_open_gate_fidelity']):.12g}",
                        flush=True,
                    )
    except KeyboardInterrupt:
        _write_summaries(sweep_dir, rows)
        print("\ninterrupted=True", file=sys.stderr, flush=True)
        print(f"sweep_dir={sweep_dir}", file=sys.stderr, flush=True)
        print(
            "latest accepted pulse is saved in the current run directory as "
            "latest_pulse.npz/latest_pulse.csv",
            file=sys.stderr,
            flush=True,
        )
        raise

    rows = _write_summaries(sweep_dir, rows)
    print(f"sweep_dir={sweep_dir}")
    print(f"summary_csv={sweep_dir / 'summary.csv'}")
    print(f"summary_md={sweep_dir / 'summary.md'}")
    return {"sweep_dir": sweep_dir, "rows": rows}


def main():
    run_sweep(parse_args())


if __name__ == "__main__":
    main()
