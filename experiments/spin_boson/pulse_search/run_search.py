"""Optimize (or evaluate) every gallery initial pulse as independent runs.

Each gallery pulse gets its own fixed output directory ``<output-dir>/<name>/``
containing the standard run artifacts plus a ``result.json`` summary, so runs
are idempotent and safe to launch in parallel (no timestamp collisions).

Selection modes (one per invocation):
    --index N     run gallery pulse N            (Slurm: $SLURM_ARRAY_TASK_ID)
    --pulse NAME  run one pulse by name
    (default)     run all pulses; --parallel K fans out K subprocesses

Usage:
    python -m experiments.spin_boson.pulse_search.run_search --index 0
    python -m experiments.spin_boson.pulse_search.run_search --parallel 2 \\
        --config experiments/spin_boson/pulse_search/smoke.yaml --evaluate-only \\
        --output-dir experiments/spin_boson/pulse_search/outputs/smoke
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import traceback
from dataclasses import replace
from pathlib import Path

import numpy as np

from experiments.driver.config_io import write_config_snapshot
from experiments.spin_boson.pulse_search.pulse_gallery import build_pulse, pulse_names
from experiments.driver.run_experiment import _load_base_config, evaluate_pulse, run_perturbative_experiment

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = PACKAGE_DIR / "search.yaml"
DEFAULT_OUTPUT_DIR = PACKAGE_DIR / "outputs"


def _write_initial_pulse(config, pulse_name, run_dir):
    """Build the gallery pulse at the config's grid and save it for the driver."""
    params = config.system.params
    amplitudes = build_pulse(
        pulse_name,
        n_steps=config.pulse.n_steps,
        alpha1_khz_bounds=tuple(getattr(params, "alpha1_khz_bounds", (1.0, 60.0))),
        alpha2_khz_bounds=tuple(getattr(params, "alpha2_khz_bounds", (0.0, 200.0))),
    )
    dt = float(config.pulse.total_time_us) * 1e-6 / config.pulse.n_steps
    npz_path = run_dir / "gallery_initial_pulse.npz"
    np.savez(npz_path, amplitudes=amplitudes, dt=dt)
    return npz_path


def _json_safe(value):
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return str(value)


def _snapshot_group_config(config, output_dir):
    """Keep one config.yaml at the top of the group folder (idempotent)."""
    group_config = Path(output_dir) / "config.yaml"
    if not group_config.exists():
        write_config_snapshot(config, group_config)


def _metrics_from_step_log(run_dir):
    """Best-effort gate fidelities from step_log.csv rows (first = initial)."""
    step_log = Path(run_dir) / "step_log.csv"
    if not step_log.exists():
        return None
    with open(step_log, newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return None
    first, last = rows[0], rows[-1]
    try:
        return {
            "initial_noisy_gate_fidelity": float(first["open_fidelity"]),
            "final_noisy_gate_fidelity": float(last["open_fidelity"]),
            "initial_close_gate_fidelity": float(first["close_fidelity"]),
            "final_close_gate_fidelity": float(last["close_fidelity"]),
            "last_step": int(last["step"]),
        }
    except (KeyError, ValueError):
        return None


def run_one(pulse_name, config_path, output_dir, evaluate_only=False):
    """Run a single gallery pulse; write result.json and return its contents.

    SIGTERM (scancel / Slurm timeout) is handled in two stages: a summary
    result.json built from step_log.csv is persisted immediately (well inside
    Slurm's ~30 s SIGTERM-to-SIGKILL grace), then the driver's
    KeyboardInterrupt path finalizes the run from the latest accepted pulse
    and overwrites it with full metrics if the grace period allows.
    """
    config = _load_base_config(config_path)
    run_dir = Path(output_dir) / pulse_name
    run_dir.mkdir(parents=True, exist_ok=True)
    _snapshot_group_config(config, output_dir)
    npz_path = _write_initial_pulse(config, pulse_name, run_dir)
    config = replace(
        config,
        runtime=replace(config.runtime, initial_pulse_npz=npz_path, no_progress=True),
    )

    record = {
        "pulse": pulse_name,
        "mode": "evaluate" if evaluate_only else "optimize",
        "config": str(config_path) if config_path else "defaults",
        "experiment_dir": str(run_dir),
    }

    def _write_record():
        with open(run_dir / "result.json", "w") as handle:
            json.dump(record, handle, indent=2)

    main_pid = os.getpid()
    sigterm_seen = {"value": False}

    def _handle_sigterm(signum, frame):
        # State-pair worker processes inherit this handler; only the main
        # process owns result.json.
        if os.getpid() != main_pid:
            raise SystemExit(128 + signum)
        sigterm_seen["value"] = True
        record["status"] = "interrupted"
        record["latest_pulse_npz"] = str(
            run_dir / f"latest_pulse_s{config.pulse.n_steps}.npz"
        )
        metrics = _metrics_from_step_log(run_dir)
        if metrics:
            record["metrics"] = metrics
        _write_record()
        raise KeyboardInterrupt

    previous_handler = signal.signal(signal.SIGTERM, _handle_sigterm)
    try:
        try:
            if evaluate_only:
                result = evaluate_pulse(config, experiment_dir=run_dir, print_report=False)
            else:
                result = run_perturbative_experiment(config, experiment_dir=run_dir, print_report=False)
            record["status"] = "interrupted" if result.get("interrupted") else "ok"
            record["metrics"] = {key: _json_safe(value) for key, value in result["metrics"].items()}
        except Exception as error:
            # After SIGTERM the driver's wrap-up may fail (workers are dying);
            # keep the interrupted record instead of downgrading it to failed.
            record["status"] = "interrupted" if sigterm_seen["value"] else "failed"
            record["error"] = f"{type(error).__name__}: {error}"
            traceback.print_exc()
        _write_record()
    finally:
        signal.signal(signal.SIGTERM, previous_handler)
    print(f"[{record['mode']}] {pulse_name}: {record['status']}", flush=True)
    if record["status"] == "failed":
        raise SystemExit(1)
    return record


def _child_command(pulse_name, args):
    command = [
        sys.executable,
        "-m",
        "experiments.spin_boson.pulse_search.run_search",
        "--pulse",
        pulse_name,
        "--output-dir",
        str(args.output_dir),
    ]
    if args.config is not None:
        command += ["--config", str(args.config)]
    if args.evaluate_only:
        command.append("--evaluate-only")
    return command


def run_all(args):
    """Run every gallery pulse, fanning out --parallel subprocesses.

    Subprocesses (rather than an in-process pool) keep the code path identical
    to a Slurm array task and leave the driver's own state-pair worker
    processes undisturbed.
    """
    names = pulse_names()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pending = list(names)
    running = {}
    failures = []
    while pending or running:
        while pending and len(running) < max(1, args.parallel):
            name = pending.pop(0)
            log_path = output_dir / f"{name}.log"
            log = open(log_path, "w")
            process = subprocess.Popen(
                _child_command(name, args), stdout=log, stderr=subprocess.STDOUT
            )
            running[name] = (process, log)
            print(f"launched {name} (pid {process.pid}, log {log_path})", flush=True)
        for name, (process, log) in list(running.items()):
            if process.poll() is None:
                continue
            log.close()
            del running[name]
            status = "ok" if process.returncode == 0 else f"failed (exit {process.returncode})"
            if process.returncode != 0:
                failures.append(name)
            print(f"finished {name}: {status}", flush=True)
        if running:
            try:
                next(iter(running.values()))[0].wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass

    print(f"\n{len(names) - len(failures)}/{len(names)} pulses succeeded.")
    if failures:
        print(f"failed: {', '.join(failures)}")
        raise SystemExit(1)
    print("Run collect_results.py for the summary table:")
    print(f"  python -m experiments.spin_boson.pulse_search.collect_results --output-dir {output_dir}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Experiment YAML.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    selector = parser.add_mutually_exclusive_group()
    selector.add_argument("--index", type=int, help="Gallery index to run (Slurm array task id).")
    selector.add_argument("--pulse", help="Gallery pulse name to run.")
    parser.add_argument("--parallel", type=int, default=1, help="Concurrent runs when running all pulses.")
    parser.add_argument("--evaluate-only", action="store_true", help="Evaluate instead of optimizing.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    names = pulse_names()
    if args.index is not None:
        if not 0 <= args.index < len(names):
            raise SystemExit(f"--index {args.index} out of range; gallery has {len(names)} pulses.")
        run_one(names[args.index], args.config, args.output_dir, args.evaluate_only)
    elif args.pulse is not None:
        run_one(args.pulse, args.config, args.output_dir, args.evaluate_only)
    else:
        run_all(args)


if __name__ == "__main__":
    main()
