"""Collect pulse-search results into a best-first summary table.

Scans ``<output-dir>/*/result.json`` (written by run_search.py), prints a
table sorted by the final noisy-gate fidelity (evaluate mode: noisy-gate
fidelity), and writes ``summary.csv`` and ``summary.md`` into the output
directory. Gallery pulses without a result.json are listed as missing.

Usage:
    python -m experiments.spin_boson.pulse_search.collect_results
    python -m experiments.spin_boson.pulse_search.collect_results --output-dir <dir>
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from experiments.spin_boson.pulse_search.pulse_gallery import pulse_names
from experiments.spin_boson.pulse_search.run_search import DEFAULT_OUTPUT_DIR

COLUMNS = {
    "optimize": (
        ("initial_noisy", "initial_noisy_gate_fidelity"),
        ("final_noisy", "final_noisy_gate_fidelity"),
        ("initial_close", "initial_close_gate_fidelity"),
        ("final_close", "final_close_gate_fidelity"),
    ),
    "evaluate": (
        ("close_gate", "close_gate_fidelity"),
        ("noisy_gate", "noisy_gate_fidelity"),
    ),
}
SORT_KEY = {"optimize": "final_noisy_gate_fidelity", "evaluate": "noisy_gate_fidelity"}


def load_records(output_dir):
    """Return (records, mode); one record per gallery pulse plus any extras on disk."""
    output_dir = Path(output_dir)
    records = {}
    for result_path in sorted(output_dir.glob("*/result.json")):
        with open(result_path) as handle:
            records[result_path.parent.name] = json.load(handle)
    for name in pulse_names():
        records.setdefault(name, {"pulse": name, "status": "missing"})
    modes = {record.get("mode") for record in records.values() if record.get("mode")}
    if len(modes) > 1:
        raise SystemExit(f"Mixed run modes {sorted(modes)} in {output_dir}; collect them separately.")
    mode = modes.pop() if modes else "optimize"
    return list(records.values()), mode


def build_rows(records, mode):
    """Return (headers, rows) sorted best-first; failed/missing runs sink to the bottom."""
    headers = ["pulse", *(label for label, _ in COLUMNS[mode]), "status"]
    sort_key = SORT_KEY[mode]

    def row_for(record):
        metrics = record.get("metrics", {})
        return [
            record["pulse"],
            *(metrics.get(key, "") for _, key in COLUMNS[mode]),
            record.get("status", "missing"),
        ]

    def sort_value(record):
        value = record.get("metrics", {}).get(sort_key)
        return value if isinstance(value, (int, float)) else float("-inf")

    ordered = sorted(records, key=sort_value, reverse=True)
    return headers, [row_for(record) for record in ordered]


def _format_cell(value):
    return f"{value:.6f}" if isinstance(value, float) else str(value)


def write_summary(output_dir, headers, rows, mode):
    output_dir = Path(output_dir)
    csv_path = output_dir / "summary.csv"
    with open(csv_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)

    md_path = output_dir / "summary.md"
    formatted = [[_format_cell(value) for value in row] for row in rows]
    with open(md_path, "w") as handle:
        handle.write(f"# Pulse-search summary ({mode}, best first)\n\n")
        handle.write("| " + " | ".join(headers) + " |\n")
        handle.write("|" + "|".join(" --- " for _ in headers) + "|\n")
        for row in formatted:
            handle.write("| " + " | ".join(row) + " |\n")
    return csv_path, md_path


def print_table(headers, rows):
    formatted = [[_format_cell(value) for value in row] for row in rows]
    widths = [max(len(headers[i]), *(len(row[i]) for row in formatted)) for i in range(len(headers))]
    print("  ".join(header.ljust(width) for header, width in zip(headers, widths)))
    for row in formatted:
        print("  ".join(cell.ljust(width) for cell, width in zip(row, widths)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)

    records, mode = load_records(args.output_dir)
    headers, rows = build_rows(records, mode)
    print(f"=== Pulse-search summary ({mode}, best first) ===")
    print_table(headers, rows)
    csv_path, md_path = write_summary(args.output_dir, headers, rows, mode)
    print(f"\nsummary_csv={csv_path}")
    print(f"summary_md={md_path}")


if __name__ == "__main__":
    main()
