"""Shrink every copied pulse-search final pulse from three starting times.

The task grid is every pulse in ``pulses/`` x three starting total times —
``orig`` (the config's ``pulse.total_time_us``), ``300us``, and ``170us`` —
pulse-major, so with 10 pulses index 0..2 is the first pulse's orig/300us/170us
and so on (``--list`` prints the full map). Each task runs the time-collapsing
loop (``experiments.spin_boson.time_collapsing``): warm-start from the copied
pulse at the starting time, multiply the total time by ``--shrink-factor``
each round, re-optimize, and stop once the noisy gate fidelity drops.

Every task writes to a fixed directory ``<output-dir>/<pulse>__from_<label>/``
(round dirs + summary + ``result.json``), so runs are idempotent and safe to
launch in parallel — locally via subprocesses or as a Slurm job array.

Selection modes (one per invocation):
    --index N              run task N               (Slurm: $SLURM_ARRAY_TASK_ID)
    --task PULSE:LABEL     run one task by name     (e.g. gaussian_lobe:300us)
    (default)              run all tasks; --parallel K fans out K subprocesses

Usage:
    python -m experiments.spin_boson.slurm_time_shrink.run_time_shrink --list
    python -m experiments.spin_boson.slurm_time_shrink.run_time_shrink --index 0
    python -m experiments.spin_boson.slurm_time_shrink.run_time_shrink \\
        --config experiments/spin_boson/slurm_time_shrink/smoke.yaml \\
        --max-rounds 2 --parallel 2 \\
        --output-dir experiments/spin_boson/slurm_time_shrink/outputs/smoke
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import traceback
from pathlib import Path

from experiments.driver.config_io import write_config_snapshot
from experiments.driver.run_experiment import _load_base_config
from experiments.spin_boson.time_collapsing.run_time_collapsing import (
    run_time_collapsing,
)

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = PACKAGE_DIR / "shrink.yaml"
DEFAULT_OUTPUT_DIR = PACKAGE_DIR / "outputs"
DEFAULT_PULSES_DIR = PACKAGE_DIR / "pulses"

# (label, starting total_time_us); None = keep the config's total_time_us.
START_TIMES = (("orig", None), ("300us", 300.0), ("170us", 170.0))
START_LABELS = tuple(label for label, _ in START_TIMES)


def pulse_files(pulses_dir=DEFAULT_PULSES_DIR):
    """Map pulse name -> npz path from ``<pulses-dir>/<name>_s<steps>.npz``."""
    pulses_dir = Path(pulses_dir)
    files = {}
    for path in sorted(pulses_dir.glob("*_s*.npz")):
        match = re.fullmatch(r"(.+)_s\d+", path.stem)
        if match:
            files[match.group(1)] = path
    if not files:
        raise SystemExit(
            f"no <name>_s<steps>.npz pulses in {pulses_dir}; "
            "run copy_final_pulses.py first."
        )
    return files


def task_grid(pulses_dir=DEFAULT_PULSES_DIR):
    """Return the (pulse, label, start_time_us, npz) list in index order."""
    return [
        (name, label, start_time_us, npz)
        for name, npz in pulse_files(pulses_dir).items()
        for label, start_time_us in START_TIMES
    ]


def task_dir_name(pulse_name, label):
    return f"{pulse_name}__from_{label}"


def _snapshot_group_config(config_path, output_dir):
    """Keep one config.yaml at the top of the group folder (idempotent)."""
    group_config = Path(output_dir) / "config.yaml"
    if not group_config.exists():
        write_config_snapshot(_load_base_config(config_path), group_config)


def run_one(pulse_name, label, start_time_us, pulse_npz, args):
    """Run one shrink task; write result.json and return its contents.

    SIGTERM (scancel / Slurm timeout) is converted to KeyboardInterrupt after
    persisting an ``interrupted`` result.json: the driver finalizes the
    in-flight round from its latest checkpoint, the shrink loop stops and
    still writes its summary, and the record is then overwritten with the
    best-so-far metrics if the ~30 s grace period allows.
    """
    task_dir = Path(args.output_dir) / task_dir_name(pulse_name, label)
    task_dir.mkdir(parents=True, exist_ok=True)
    _snapshot_group_config(args.config, args.output_dir)

    record = {
        "task": f"{pulse_name}:{label}",
        "pulse": pulse_name,
        "start_label": label,
        "start_total_time_us": start_time_us,
        "mode": "shrink",
        "config": str(args.config),
        "source_pulse_npz": str(pulse_npz),
        "experiment_dir": str(task_dir),
        "shrink_factor": args.shrink_factor,
        "fidelity_drop_tolerance": args.fidelity_drop_tolerance,
        "max_rounds": args.max_rounds,
    }
    if start_time_us is None:
        record["start_total_time_us"] = float(
            _load_base_config(args.config).pulse.total_time_us
        )

    def _write_record():
        with open(task_dir / "result.json", "w") as handle:
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
        _write_record()
        raise KeyboardInterrupt

    previous_handler = signal.signal(signal.SIGTERM, _handle_sigterm)
    try:
        try:
            outcome = run_time_collapsing(
                args.config,
                shrink_factor=args.shrink_factor,
                fidelity_drop_tolerance=args.fidelity_drop_tolerance,
                max_rounds=args.max_rounds,
                maxiter=args.maxiter,
                total_time_us=start_time_us,
                initial_pulse_npz=pulse_npz,
                top_dir=task_dir,
            )
            best = outcome["best_round"]
            interrupted = sigterm_seen["value"] or "interrupted" in outcome["stop_reason"]
            record["status"] = "interrupted" if interrupted else "ok"
            record["stop_reason"] = outcome["stop_reason"]
            record["metrics"] = {
                "rounds": len(outcome["rounds"]),
                "best_round": int(best["round"]),
                "best_total_time_us": float(best["total_time_us"]),
                "best_noisy_gate_fidelity": float(best["final_noisy_gate_fidelity"]),
                "best_close_gate_fidelity": float(best["final_close_gate_fidelity"]),
                "best_pulse_npz": str(best["final_pulse_npz"]),
            }
        except Exception as error:
            # After SIGTERM the loop's wrap-up may fail (workers are dying);
            # keep the interrupted record instead of downgrading it to failed.
            record["status"] = "interrupted" if sigterm_seen["value"] else "failed"
            record["error"] = f"{type(error).__name__}: {error}"
            traceback.print_exc()
        _write_record()
    finally:
        signal.signal(signal.SIGTERM, previous_handler)
    print(f"[shrink] {pulse_name}:{label}: {record['status']}", flush=True)
    if record["status"] == "failed":
        raise SystemExit(1)
    return record


def _child_command(pulse_name, label, args):
    command = [
        sys.executable,
        "-m",
        "experiments.spin_boson.slurm_time_shrink.run_time_shrink",
        "--task",
        f"{pulse_name}:{label}",
        "--config",
        str(args.config),
        "--output-dir",
        str(args.output_dir),
        "--pulses-dir",
        str(args.pulses_dir),
        "--shrink-factor",
        str(args.shrink_factor),
        "--fidelity-drop-tolerance",
        str(args.fidelity_drop_tolerance),
        "--max-rounds",
        str(args.max_rounds),
    ]
    if args.maxiter is not None:
        command += ["--maxiter", str(args.maxiter)]
    return command


def run_all(args):
    """Run every task, fanning out --parallel subprocesses.

    Subprocesses (rather than an in-process pool) keep the code path identical
    to a Slurm array task and leave the driver's own state-pair worker
    processes undisturbed.
    """
    tasks = task_grid(args.pulses_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pending = [(pulse, label) for pulse, label, _, _ in tasks]
    running = {}
    failures = []
    while pending or running:
        while pending and len(running) < max(1, args.parallel):
            pulse, label = pending.pop(0)
            name = task_dir_name(pulse, label)
            log_path = output_dir / f"{name}.log"
            log = open(log_path, "w")
            process = subprocess.Popen(
                _child_command(pulse, label, args), stdout=log, stderr=subprocess.STDOUT
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

    print(f"\n{len(tasks) - len(failures)}/{len(tasks)} tasks succeeded.")
    if failures:
        print(f"failed: {', '.join(failures)}")
        raise SystemExit(1)
    print("Run collect_results.py for the summary table:")
    print(
        "  python -m experiments.spin_boson.slurm_time_shrink.collect_results "
        f"--output-dir {output_dir}"
    )


def list_tasks(pulses_dir):
    print("index  task                              start_time_us")
    for index, (pulse, label, start_time_us, _) in enumerate(task_grid(pulses_dir)):
        start = "config" if start_time_us is None else f"{start_time_us:g}"
        print(f"{index:>5}  {f'{pulse}:{label}':<32}  {start}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Experiment YAML.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--pulses-dir", type=Path, default=DEFAULT_PULSES_DIR)
    selector = parser.add_mutually_exclusive_group()
    selector.add_argument("--index", type=int, help="Task index to run (Slurm array task id).")
    selector.add_argument("--task", help="Task to run, as <pulse>:<label> (labels: orig, 300us, 170us).")
    selector.add_argument("--list", action="store_true", help="Print the index -> task map and exit.")
    parser.add_argument("--parallel", type=int, default=1, help="Concurrent runs when running all tasks.")
    parser.add_argument("--shrink-factor", type=float, default=0.95, help="Total-time multiplier per round.")
    parser.add_argument("--fidelity-drop-tolerance", type=float, default=1e-4)
    parser.add_argument("--max-rounds", type=int, default=30, help="Safety cap on shrink rounds per task.")
    parser.add_argument("--maxiter", type=int, default=None, help="Override optimizer.maxiter from the config.")
    args = parser.parse_args(argv)
    if not 0.0 < args.shrink_factor < 1.0:
        parser.error("--shrink-factor must be strictly between 0 and 1.")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.list:
        list_tasks(args.pulses_dir)
        return
    tasks = task_grid(args.pulses_dir)
    if args.index is not None:
        if not 0 <= args.index < len(tasks):
            raise SystemExit(f"--index {args.index} out of range; the grid has {len(tasks)} tasks.")
        run_one(*tasks[args.index], args)
    elif args.task is not None:
        by_name = {f"{pulse}:{label}": task for task in tasks for pulse, label, _, _ in [task]}
        if args.task not in by_name:
            raise SystemExit(f"unknown task {args.task!r}; see --list.")
        run_one(*by_name[args.task], args)
    else:
        run_all(args)


if __name__ == "__main__":
    main()
