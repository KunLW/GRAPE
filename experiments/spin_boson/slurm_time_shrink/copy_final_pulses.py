"""Copy pulse-search final pulses into this experiment's ``pulses/`` folder.

Scans ``<source-dir>/<pulse>/result.json`` (a pulse-search group folder, one
subfolder per gallery pulse), keeps the pulses whose optimization finished
``ok``, and copies each ``final_pulse_s<steps>.npz`` to
``pulses/<pulse>_s<steps>.npz``. A ``pulses/manifest.json`` records the
provenance (source run, per-pulse fidelities) so the copies stay traceable
after the git-ignored source outputs are gone.

Run once locally and commit the result (outputs/ never reaches the cluster,
the tracked pulses/ folder does):

    python -m experiments.spin_boson.slurm_time_shrink.copy_final_pulses
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE_DIR = (
    PACKAGE_DIR.parent / "pulse_search" / "outputs" / "pulse_search_260709"
)
DEFAULT_PULSES_DIR = PACKAGE_DIR / "pulses"


def copy_final_pulses(source_dir, pulses_dir):
    source_dir = Path(source_dir)
    pulses_dir = Path(pulses_dir)
    if not source_dir.is_dir():
        raise SystemExit(f"source dir {source_dir} does not exist.")
    pulses_dir.mkdir(parents=True, exist_ok=True)

    entries = {}
    for result_path in sorted(source_dir.glob("*/result.json")):
        with open(result_path) as handle:
            record = json.load(handle)
        pulse = record.get("pulse", result_path.parent.name)
        if record.get("mode") != "optimize" or record.get("status") != "ok":
            print(
                f"skipping {pulse}: mode={record.get('mode')} "
                f"status={record.get('status')}"
            )
            continue
        # Current driver exports under <run>/pulse/, older runs at the top level.
        candidates = sorted(result_path.parent.glob("pulse/final_pulse_s*.npz")) or sorted(
            result_path.parent.glob("final_pulse_s*.npz")
        )
        if len(candidates) != 1:
            raise SystemExit(
                f"{result_path.parent} has {len(candidates)} final_pulse_s*.npz "
                "files; expected exactly one."
            )
        destination = pulses_dir / f"{pulse}_{candidates[0].stem.split('_')[-1]}.npz"
        shutil.copy2(candidates[0], destination)
        metrics = record.get("metrics", {})
        entries[pulse] = {
            "file": destination.name,
            "source": str(candidates[0]),
            "final_noisy_gate_fidelity": metrics.get("final_noisy_gate_fidelity"),
            "final_close_gate_fidelity": metrics.get("final_close_gate_fidelity"),
        }
        print(f"copied {pulse} -> {destination}")

    if not entries:
        raise SystemExit(f"no ok optimize results found in {source_dir}.")
    manifest = {
        "source_dir": str(source_dir),
        "copied_at": datetime.now().isoformat(timespec="seconds"),
        "pulses": entries,
    }
    manifest_path = pulses_dir / "manifest.json"
    with open(manifest_path, "w") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"\n{len(entries)} pulses copied; manifest at {manifest_path}")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--pulses-dir", type=Path, default=DEFAULT_PULSES_DIR)
    args = parser.parse_args(argv)
    copy_final_pulses(args.source_dir, args.pulses_dir)


if __name__ == "__main__":
    main()
