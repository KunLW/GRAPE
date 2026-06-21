from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re


def timestamped_report_path(output_dir, experiment_slug, generated_at=None):
    output_dir = Path(output_dir)
    timestamp = (generated_at or datetime.now()).strftime("%Y%m%d_%H%M%S")
    safe_slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", experiment_slug).strip("._")
    if not safe_slug:
        raise ValueError("experiment_slug must contain at least one safe character.")
    return output_dir / f"{safe_slug}_{timestamp}.md"


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
