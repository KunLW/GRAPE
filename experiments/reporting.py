from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
import re

import numpy as np

STEP_LOG_FIELDS = (
    "step",
    "close_fidelity",
    "open_fidelity",
    "raw_fidelity",
    "l1_penalty",
    "l2_penalty",
    "cost_function",
    "gradient_norm",
)

FIDELITY_TERMS_FIELDS = (
    "step",
    "closed_term",
    "first_order_sq",
    "second_order_cross",
    "perturbative_open",
    "correction",
    "excess_over_1",
    "max_pair_open",
    "min_pair_open",
)

FIDELITY_PAIR_TERMS_FIELDS = (
    "step",
    "pair_index",
    "weight",
    "a0_real",
    "a0_imag",
    "a1_real",
    "a1_imag",
    "a2_real",
    "a2_imag",
    "closed_term",
    "first_order_sq",
    "second_order_cross",
    "perturbative_open",
    "dropped_order1_cross",
)


def timestamped_report_path(output_dir, experiment_slug, generated_at=None):
    return timestamped_experiment_dir(output_dir, experiment_slug, generated_at) / "report.md"


def timestamped_experiment_dir(output_dir, experiment_slug, generated_at=None):
    output_dir = Path(output_dir)
    timestamp = (generated_at or datetime.now()).strftime("%Y%m%d_%H%M%S")
    safe_slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", experiment_slug).strip("._")
    if not safe_slug:
        raise ValueError("experiment_slug must contain at least one safe character.")
    return output_dir / f"{safe_slug}_{timestamp}"


def write_experiment_report(
    output_dir,
    experiment_slug,
    title,
    configuration,
    results,
    optimizer,
    figures,
    generated_at=None,
):
    generated_at = generated_at or datetime.now()
    report_path = timestamped_report_path(output_dir, experiment_slug, generated_at)
    return write_experiment_report_at(
        report_path,
        title=title,
        configuration=configuration,
        results=results,
        optimizer=optimizer,
        figures=figures,
        generated_at=generated_at,
    )


def write_experiment_report_at(
    report_path,
    title,
    configuration,
    results,
    optimizer,
    figures,
    generated_at=None,
):
    generated_at = generated_at or datetime.now()
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# {title}",
        "",
        f"Generated at: {generated_at.isoformat(timespec='seconds')}",
        "",
        "## Configuration",
        "",
        _markdown_table(("Parameter", "Value"), configuration),
        "",
        "## Results",
        "",
        _markdown_table(("Metric", "Initial", "Final", "Delta"), _result_rows(results)),
        "",
        "## Optimizer",
        "",
        _markdown_table(("Parameter", "Value"), optimizer),
        "",
        "## Figures",
        "",
    ]
    for label, figure_path in figures:
        relative_path = Path(figure_path).relative_to(report_path.parent)
        lines.extend([f"### {_format_cell(label)}", "", f"![{label}]({relative_path.as_posix()})", ""])

    report_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return report_path


class StepLog:
    def __init__(self, output_path, print_steps=False, stream=None):
        self.output_path = Path(output_path)
        self.print_steps = print_steps
        self.stream = stream
        self.rows = []
        self._printed_header = False

    def append(
        self,
        step,
        close_fidelity,
        open_fidelity,
        cost_function,
        raw_fidelity=None,
        l1_penalty=0.0,
        l2_penalty=0.0,
        gradient_norm=None,
    ):
        if raw_fidelity is None:
            raw_fidelity = cost_function + l1_penalty + l2_penalty
        if gradient_norm is None:
            gradient_norm = float("nan")
        row = {
            "step": int(step),
            "close_fidelity": float(close_fidelity),
            "open_fidelity": float(open_fidelity),
            "raw_fidelity": float(raw_fidelity),
            "l1_penalty": float(l1_penalty),
            "l2_penalty": float(l2_penalty),
            "cost_function": float(cost_function),
            "gradient_norm": float(gradient_norm),
        }
        self.rows.append(row)
        if self.print_steps:
            if not self._printed_header:
                print(format_step_table_header(), file=self.stream, flush=True)
                self._printed_header = True
            print(format_step_table_row(row), file=self.stream, flush=True)
        self.write()

    def write(self):
        write_step_log_csv(self.output_path, self.rows)


class FidelityTermsLog:
    def __init__(self, summary_path, pair_path, print_steps=False, stream=None):
        self.summary_path = Path(summary_path)
        self.pair_path = Path(pair_path)
        self.print_steps = print_steps
        self.stream = stream
        self.summary_rows = []
        self.pair_rows = []
        self._printed_header = False

    def append(self, summary_row, pair_rows):
        row = {field: summary_row[field] for field in FIDELITY_TERMS_FIELDS}
        rows = [{field: pair_row[field] for field in FIDELITY_PAIR_TERMS_FIELDS} for pair_row in pair_rows]
        self.summary_rows.append(row)
        self.pair_rows.extend(rows)
        if self.print_steps:
            if not self._printed_header:
                print(format_fidelity_terms_table_header(), file=self.stream, flush=True)
                self._printed_header = True
            print(format_fidelity_terms_table_row(row), file=self.stream, flush=True)
        self.write()

    def write(self):
        write_fidelity_terms_csv(self.summary_path, self.summary_rows)
        write_fidelity_pair_terms_csv(self.pair_path, self.pair_rows)


def write_step_log_csv(output_path, rows):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=STEP_LOG_FIELDS,
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(_step_log_csv_row(row))


def write_fidelity_terms_csv(output_path, rows):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIDELITY_TERMS_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(_generic_csv_row(row, FIDELITY_TERMS_FIELDS, integer_fields={"step"}))


def write_fidelity_pair_terms_csv(output_path, rows):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIDELITY_PAIR_TERMS_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                _generic_csv_row(row, FIDELITY_PAIR_TERMS_FIELDS, integer_fields={"step", "pair_index"})
            )


def format_step_table_header():
    return (
        f"{'step':<4s}  "
        f"{'close fidelity':<16s}  "
        f"{'open fidelity':<16s}  "
        f"{'raw fidelity':<16s}  "
        f"{'l1 penalty':<13s}  "
        f"{'l2 penalty':<13s}  "
        f"{'cost function':<16s}  "
        f"{'gradient norm':<16s}"
    )


def format_step_table_row(row):
    return (
        f"{int(row['step']):<4d}  "
        f"{float(row['close_fidelity']):<16.12g}  "
        f"{float(row['open_fidelity']):<16.12g}  "
        f"{float(row['raw_fidelity']):<16.12g}  "
        f"{float(row['l1_penalty']):<13.6g}  "
        f"{float(row['l2_penalty']):<13.6g}  "
        f"{float(row['cost_function']):<16.12g}  "
        f"{float(row['gradient_norm']):<16.12g}"
    )


def format_fidelity_terms_table_header():
    return (
        f"{'step':<4s}  "
        f"{'closed term':<16s}  "
        f"{'first order sq':<16s}  "
        f"{'second order cross':<20s}  "
        f"{'perturbative open':<18s}  "
        f"{'correction':<16s}  "
        f"{'excess over 1':<16s}  "
        f"{'max pair open':<16s}  "
        f"{'min pair open':<16s}"
    )


def format_fidelity_terms_table_row(row):
    return (
        f"{int(row['step']):<4d}  "
        f"{float(row['closed_term']):<16.12g}  "
        f"{float(row['first_order_sq']):<16.12g}  "
        f"{float(row['second_order_cross']):<20.12g}  "
        f"{float(row['perturbative_open']):<18.12g}  "
        f"{float(row['correction']):<16.12g}  "
        f"{float(row['excess_over_1']):<16.12g}  "
        f"{float(row['max_pair_open']):<16.12g}  "
        f"{float(row['min_pair_open']):<16.12g}"
    )


def export_pulse_controls(
    pulse,
    output_stem,
    rad_s_per_khz,
    channel_names=("alpha1", "alpha2"),
):
    output_stem = Path(output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    amplitudes = np.asarray(pulse.amplitudes, dtype=float)
    channel_names = np.asarray(channel_names)
    if amplitudes.shape[1] != len(channel_names):
        raise ValueError("channel_names must match the number of pulse controls.")

    step_index = np.arange(pulse.n_steps)
    time_s = (step_index + 0.5) * pulse.dt
    time_us = time_s * 1e6
    npz_path = output_stem.with_suffix(".npz")
    csv_path = output_stem.with_suffix(".csv")

    np.savez(
        npz_path,
        amplitudes=amplitudes,
        dt=float(pulse.dt),
        time_s=time_s,
        time_us=time_us,
        rad_s_per_khz=float(rad_s_per_khz),
        channel_names=channel_names,
    )

    # Column layout matches the historical alpha1/alpha2 format: rad/s
    # columns for every channel, then converted (rad_s_per_khz-divided)
    # columns for every channel.
    names = channel_names.tolist()
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "step_index",
                "time_s",
                "time_us",
                *(f"{name}_rad_s" for name in names),
                *(f"{name}_khz" for name in names),
            ]
        )
        for index in range(pulse.n_steps):
            writer.writerow(
                [
                    int(step_index[index]),
                    _format_csv_float(time_s[index]),
                    _format_csv_float(time_us[index]),
                    *(
                        _format_csv_float(amplitudes[index, channel])
                        for channel in range(len(names))
                    ),
                    *(
                        _format_csv_float(amplitudes[index, channel] / rad_s_per_khz)
                        for channel in range(len(names))
                    ),
                ]
            )
    return npz_path, csv_path


def _result_rows(results):
    rows = []
    for name, initial, final in results:
        rows.append((name, _format_value(initial), _format_value(final), _format_value(final - initial)))
    return rows


def _markdown_table(headers, rows):
    table = [
        "| " + " | ".join(_format_cell(header) for header in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        table.append("| " + " | ".join(_format_cell(value) for value in row) + " |")
    return "\n".join(table)


def _format_cell(value):
    return str(_format_value(value)).replace("|", "\\|").replace("\n", "<br>")


def _format_value(value):
    if isinstance(value, float):
        return f"{value:.12g}"
    return value


def _format_csv_float(value):
    return f"{float(value):.17g}"


def _step_log_csv_row(row):
    complete_row = {}
    for field in STEP_LOG_FIELDS:
        if field == "step":
            complete_row[field] = int(row[field])
        elif field == "raw_fidelity" and field not in row:
            complete_row[field] = _format_csv_float(
                row["cost_function"]
                + row.get("l1_penalty", 0.0)
                + row.get("l2_penalty", 0.0)
            )
        elif field == "gradient_norm" and field not in row:
            complete_row[field] = _format_csv_float(float("nan"))
        else:
            complete_row[field] = _format_csv_float(row.get(field, 0.0))
    return complete_row


def _generic_csv_row(row, fields, integer_fields=()):
    integer_fields = set(integer_fields)
    complete_row = {}
    for field in fields:
        if field in integer_fields:
            complete_row[field] = int(row[field])
        else:
            complete_row[field] = _format_csv_float(row.get(field, 0.0))
    return complete_row
