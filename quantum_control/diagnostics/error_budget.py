from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from quantum_control.problems.context import EvolutionContext
from quantum_control.differentiators.expansion_differentiator import (
    PerturbativeExpansionDifferentiator,
)
from quantum_control.evolution.expansion_evolution import PerturbativeExpansionEvolution
from quantum_control.objectives.expansion_fidelity import ExpansionFidelity
from quantum_control.pulses.pulse import PiecewiseConstantPulse
from quantum_control.problems.state_average import StatePair
from quantum_control.steps.perturbative_step import PerturbativeStepBuilder


@dataclass(frozen=True)
class ErrorBudgetConfig:
    finite_difference_epsilon: float = 1e-6
    gradient_samples: int = 16
    fluctuation_scales: tuple[float, ...] = (0.25, 0.5, 1.0)
    random_seed: int = 12345
    max_order: int = 2
    normalize_weights: bool = False
    v_derivative_epsilon: float = 1e-7


@dataclass(frozen=True)
class ErrorBudgetReport:
    rows: tuple[dict, ...]
    metadata: dict


@dataclass(frozen=True)
class _WeightedPair:
    initial_state: np.ndarray
    target_state: np.ndarray
    weight: float


class _ScaledFluctuationSystem:
    def __init__(self, base_system, scale):
        self.base_system = base_system
        self.scale = float(scale)
        self.static_fluctuations = tuple(
            self.scale * np.asarray(item, dtype=complex)
            for item in getattr(base_system, "static_fluctuations", ())
        )
        self.control_fluctuations = tuple(
            self.scale * np.asarray(item, dtype=complex)
            for item in getattr(base_system, "control_fluctuations", ())
        )

    def nominal_hamiltonian(self, controls, t=None):
        return self.base_system.nominal_hamiltonian(controls, t=t)

    def control_hamiltonian(self, control_index, controls=None, t=None):
        return self.base_system.control_hamiltonian(control_index, controls=controls, t=t)

    def fluctuation_hamiltonian(self, controls, t=None):
        hamiltonian = np.zeros_like(
            self.base_system.nominal_hamiltonian(controls, t=t),
            dtype=complex,
        )
        for fluctuation_h in self.static_fluctuations:
            hamiltonian = hamiltonian + fluctuation_h
        for amplitude, fluctuation_h in zip(controls, self.control_fluctuations):
            hamiltonian = hamiltonian + amplitude * fluctuation_h
        return hamiltonian

    def fluctuation_control_derivative(self, control_index, controls=None, t=None):
        if control_index >= len(self.control_fluctuations):
            return np.zeros_like(
                self.base_system.nominal_hamiltonian(controls, t=t),
                dtype=complex,
            )
        return self.control_fluctuations[control_index]


class _FluctuationCompatibleSystem:
    def __init__(self, base_system):
        self.base_system = base_system
        self.static_fluctuations = tuple(getattr(base_system, "static_fluctuations", ()))
        self.control_fluctuations = tuple(getattr(base_system, "control_fluctuations", ()))

    def nominal_hamiltonian(self, controls, t=None):
        return self.base_system.nominal_hamiltonian(controls, t=t)

    def control_hamiltonian(self, control_index, controls=None, t=None):
        return self.base_system.control_hamiltonian(control_index, controls=controls, t=t)

    def fluctuation_hamiltonian(self, controls, t=None):
        if hasattr(self.base_system, "fluctuation_hamiltonian"):
            return self.base_system.fluctuation_hamiltonian(controls, t=t)
        return np.zeros_like(self.nominal_hamiltonian(controls, t=t), dtype=complex)

    def fluctuation_control_derivative(self, control_index, controls=None, t=None):
        if hasattr(self.base_system, "fluctuation_control_derivative"):
            return self.base_system.fluctuation_control_derivative(
                control_index,
                controls=controls,
                t=t,
            )
        return np.zeros_like(self.nominal_hamiltonian(controls, t=t), dtype=complex)


def load_pulse_npz(path, fallback_dt=None):
    path = Path(path)
    with np.load(path) as data:
        if "amplitudes" not in data.files:
            raise ValueError(f"{path} does not contain required 'amplitudes'.")
        amplitudes = np.asarray(data["amplitudes"], dtype=float)
        if "dt" in data.files:
            dt = float(np.asarray(data["dt"]).reshape(()))
        elif fallback_dt is not None:
            dt = float(fallback_dt)
        else:
            raise ValueError(f"{path} does not contain 'dt'; provide fallback_dt.")
    return PiecewiseConstantPulse(amplitudes=amplitudes, dt=dt)


def evaluate_error_budget(system, pulse, state_pairs, config=None):
    config = config or ErrorBudgetConfig()
    pairs = _weighted_pairs(state_pairs, normalize=config.normalize_weights)
    diagnostic_system = _FluctuationCompatibleSystem(system)
    rows = []

    rows.extend(_w_error_rows(diagnostic_system, pulse))
    rows.extend(_gradient_error_rows(diagnostic_system, pulse, pairs, config))
    rows.extend(_v_insertion_rows(diagnostic_system, pulse, pairs, config))
    rows.extend(_truncation_rows(system, pulse, pairs, config))

    metadata = {
        "n_steps": pulse.n_steps,
        "n_controls": pulse.n_controls,
        "dt": pulse.dt,
        "n_state_pairs": len(pairs),
        "gradient_samples": config.gradient_samples,
        "fluctuation_scales": config.fluctuation_scales,
        "random_seed": config.random_seed,
    }
    return ErrorBudgetReport(rows=tuple(rows), metadata=metadata)


def write_error_budget_report(report, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "error_budget.csv"
    md_path = output_dir / "error_budget.md"

    fieldnames = ["category", "metric", "scale", "value", "available", "notes"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in report.rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})

    lines = ["# Error Budget", "", "## Metadata", ""]
    for key, value in report.metadata.items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Metrics", "", "| category | metric | scale | value | available | notes |"])
    lines.append("|---|---:|---:|---:|---|---|")
    for row in report.rows:
        lines.append(
            "| {category} | {metric} | {scale} | {value} | {available} | {notes} |".format(
                category=row.get("category", ""),
                metric=row.get("metric", ""),
                scale=row.get("scale", ""),
                value=_format_value(row.get("value", "")),
                available=row.get("available", ""),
                notes=row.get("notes", ""),
            )
        )
    summary_rows = _summary_rows(report.rows)
    if summary_rows:
        lines.extend(["", "## Summary", "", "| item | value | source metric |"])
        lines.append("|---|---:|---|")
        for item, value, metric in summary_rows:
            lines.append(f"| {item} | {_format_value(value)} | `{metric}` |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"markdown": md_path, "csv": csv_path}


def _weighted_pairs(state_pairs, normalize):
    pairs = tuple(_coerce_pair(pair) for pair in state_pairs)
    if not pairs:
        raise ValueError("state_pairs must contain at least one pair.")
    weights = np.asarray([pair.weight for pair in pairs], dtype=float)
    if np.any(weights < 0.0):
        raise ValueError("state pair weights must be non-negative.")
    total = float(np.sum(weights))
    if total <= 0.0:
        raise ValueError("state pair weights must have positive total weight.")
    if normalize:
        weights = weights / total
    return tuple(
        _WeightedPair(pair.initial_state, pair.target_state, float(weight))
        for pair, weight in zip(pairs, weights)
    )


def _coerce_pair(pair):
    if isinstance(pair, StatePair):
        return _WeightedPair(
            np.asarray(pair.initial_state, dtype=complex),
            np.asarray(pair.target_state, dtype=complex),
            float(pair.weight),
        )
    if len(pair) == 2:
        initial_state, target_state = pair
        weight = 1.0
    elif len(pair) == 3:
        initial_state, target_state, weight = pair
    else:
        raise ValueError("state pairs must be StatePair, (initial, target), or (initial, target, weight).")
    return _WeightedPair(
        np.asarray(initial_state, dtype=complex),
        np.asarray(target_state, dtype=complex),
        float(weight),
    )


def _gradient_error_rows(system, pulse, pairs, config):
    coordinates = _sample_coordinates(
        pulse.amplitudes.shape,
        config.gradient_samples,
        config.random_seed,
    )
    first_builder = PerturbativeStepBuilder(dW_method="first_order")
    frechet_builder = PerturbativeStepBuilder(dW_method="frechet")
    first_gradient = _average_gradient(system, pulse, pairs, first_builder, config)
    frechet_gradient = _average_gradient(system, pulse, pairs, frechet_builder, config)
    fd_values = _finite_difference_values(system, pulse, pairs, coordinates, config)

    first_values = np.asarray([first_gradient[index] for index in coordinates], dtype=float)
    frechet_values = np.asarray([frechet_gradient[index] for index in coordinates], dtype=float)
    fd_values = np.asarray(fd_values, dtype=float)

    return [
        _row("dW", "sample_count", "", len(coordinates), True, "sampled gradient coordinates"),
        _row("dW", "norm_first_minus_fd", "", _norm(first_values - fd_values), True),
        _row("dW", "norm_frechet_minus_fd", "", _norm(frechet_values - fd_values), True),
        _row("dW", "norm_first_minus_frechet", "", _norm(first_values - frechet_values), True),
        _row("dW", "relative_first_minus_fd", "", _relative_norm(first_values, fd_values), True),
        _row("dW", "relative_frechet_minus_fd", "", _relative_norm(frechet_values, fd_values), True),
    ]


def _w_error_rows(system, pulse):
    step_builder = PerturbativeStepBuilder()
    residuals = []
    for step_index in range(pulse.n_steps):
        step = step_builder.build_step(
            system,
            pulse.controls_at(step_index),
            pulse.dt,
            t=step_index * pulse.dt,
        )
        identity = np.eye(step.W.shape[0], dtype=complex)
        residuals.append(float(np.linalg.norm(step.W.conj().T @ step.W - identity, ord="fro")))
    return [
        _row(
            "W",
            "unitarity_fro_mean",
            "",
            float(np.mean(residuals)),
            True,
            "nominal expm W^dagger W - I",
        ),
        _row(
            "W",
            "unitarity_fro_max",
            "",
            float(np.max(residuals)),
            True,
            "nominal expm W^dagger W - I",
        ),
    ]


def _v_insertion_rows(system, pulse, pairs, config):
    leading_builder = PerturbativeStepBuilder(V_method="leading")
    frechet_builder = PerturbativeStepBuilder(
        V_method="frechet",
        v_derivative_epsilon=config.v_derivative_epsilon,
    )
    fro_errors = []
    spectral_errors = []
    for step_index in range(pulse.n_steps):
        controls = pulse.controls_at(step_index)
        t = step_index * pulse.dt
        leading = leading_builder.build_step(system, controls, pulse.dt, t=t).V
        frechet = frechet_builder.build_step(system, controls, pulse.dt, t=t).V
        fro_errors.append(_relative_matrix_error(leading, frechet, ord=None))
        spectral_errors.append(_relative_matrix_error(leading, frechet, ord=2))

    leading_value = _perturbative_value(system, pulse, pairs, leading_builder, config)
    frechet_value = _perturbative_value(system, pulse, pairs, frechet_builder, config)

    return [
        _row("V", "relative_fro_mean", "", float(np.mean(fro_errors)), True),
        _row("V", "relative_fro_median", "", float(np.median(fro_errors)), True),
        _row("V", "relative_fro_max", "", float(np.max(fro_errors)), True),
        _row("V", "relative_spectral_mean", "", float(np.mean(spectral_errors)), True),
        _row("V", "relative_spectral_max", "", float(np.max(spectral_errors)), True),
        _row("V", "fidelity_leading", "", leading_value, True),
        _row("V", "fidelity_frechet", "", frechet_value, True),
        _row("V", "fidelity_leading_minus_frechet", "", leading_value - frechet_value, True),
    ]


def _truncation_rows(system, pulse, pairs, config):
    if not _has_enumerable_fluctuations(system):
        return [
            _row(
                "truncation",
                "sigmaT_estimate_available",
                "",
                "",
                False,
                "system lacks static_fluctuations/control_fluctuations sequences",
            )
        ]

    rows = []
    for scale in config.fluctuation_scales:
        scaled_system = _ScaledFluctuationSystem(system, scale)
        perturbative = _perturbative_value(
            scaled_system,
            pulse,
            pairs,
            PerturbativeStepBuilder(),
            config,
        )
        sigma_estimate = _sigma_total_time_estimate(system, pulse, scale)
        rows.extend(
            [
                _row("truncation", "perturbative_fidelity", scale, perturbative, True),
                _row("truncation", "sigma_rms_spectral", scale, sigma_estimate["sigma_rms"], True),
                _row("truncation", "sigma_mean_spectral", scale, sigma_estimate["sigma_mean"], True),
                _row("truncation", "sigma_max_spectral", scale, sigma_estimate["sigma_max"], True),
                _row("truncation", "total_time", scale, sigma_estimate["total_time"], True),
                _row(
                    "truncation",
                    "sigmaT_squared_estimate",
                    scale,
                    sigma_estimate["sigmaT_squared"],
                    True,
                    "(sigma_rms_spectral * total_time)^2",
                ),
            ]
        )
    return rows


def _average_gradient(system, pulse, pairs, step_builder, config):
    objective = ExpansionFidelity(max_order=config.max_order, drop_odd_average=True)
    evolution = PerturbativeExpansionEvolution(step_builder, max_order=config.max_order)
    differentiator = PerturbativeExpansionDifferentiator(step_builder, objective)
    gradient = np.zeros_like(pulse.amplitudes)
    for pair in pairs:
        context = EvolutionContext(pair.initial_state, pair.target_state)
        result = evolution.evolve(system, pulse, context)
        gradient = gradient + pair.weight * differentiator.gradient(system, pulse, context, result)
    return gradient


def _finite_difference_values(system, pulse, pairs, coordinates, config):
    values = []
    for step_index, control_index in coordinates:
        plus = np.array(pulse.amplitudes, copy=True)
        minus = np.array(pulse.amplitudes, copy=True)
        plus[step_index, control_index] += config.finite_difference_epsilon
        minus[step_index, control_index] -= config.finite_difference_epsilon
        plus_pulse = pulse.with_amplitudes(plus)
        minus_pulse = pulse.with_amplitudes(minus)
        plus_value = _perturbative_value(system, plus_pulse, pairs, PerturbativeStepBuilder(), config)
        minus_value = _perturbative_value(system, minus_pulse, pairs, PerturbativeStepBuilder(), config)
        values.append((plus_value - minus_value) / (2.0 * config.finite_difference_epsilon))
    return values


def _perturbative_value(system, pulse, pairs, step_builder, config):
    objective = ExpansionFidelity(max_order=config.max_order, drop_odd_average=True)
    evolution = PerturbativeExpansionEvolution(step_builder, max_order=config.max_order)
    value = 0.0
    for pair in pairs:
        context = EvolutionContext(pair.initial_state, pair.target_state)
        result = evolution.evolve(system, pulse, context)
        value = value + pair.weight * objective.evaluate(result)
    return float(value)


def _sigma_total_time_estimate(system, pulse, scale):
    static_norms = [
        float(np.linalg.norm(scale * np.asarray(fluctuation_h, dtype=complex), ord=2))
        for fluctuation_h in getattr(system, "static_fluctuations", ())
    ]
    control_fluctuations = tuple(
        scale * np.asarray(fluctuation_h, dtype=complex)
        for fluctuation_h in getattr(system, "control_fluctuations", ())
    )
    control_norms = [float(np.linalg.norm(fluctuation_h, ord=2)) for fluctuation_h in control_fluctuations]

    sigma_values = []
    static_variance = float(np.sum(np.asarray(static_norms, dtype=float) ** 2))
    for step_index in range(pulse.n_steps):
        controls = pulse.controls_at(step_index)
        control_variance = 0.0
        for amplitude, control_norm in zip(controls, control_norms):
            control_variance = control_variance + float((amplitude * control_norm) ** 2)
        sigma_values.append(np.sqrt(static_variance + control_variance))

    sigma_values = np.asarray(sigma_values, dtype=float)
    sigma_rms = float(np.sqrt(np.mean(sigma_values**2)))
    total_time = float(pulse.n_steps * pulse.dt)
    return {
        "sigma_rms": sigma_rms,
        "sigma_mean": float(np.mean(sigma_values)),
        "sigma_max": float(np.max(sigma_values)),
        "total_time": total_time,
        "sigmaT_squared": float((sigma_rms * total_time) ** 2),
    }


def _sample_coordinates(shape, count, seed):
    n_steps, n_controls = shape
    all_coordinates = [(step, control) for step in range(n_steps) for control in range(n_controls)]
    if count <= 0 or count >= len(all_coordinates):
        return tuple(all_coordinates)
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(all_coordinates), size=count, replace=False)
    return tuple(all_coordinates[int(index)] for index in np.sort(indices))


def _has_enumerable_fluctuations(system):
    return hasattr(system, "static_fluctuations") and hasattr(system, "control_fluctuations")


def _relative_matrix_error(approx, exact, ord):
    numerator = np.linalg.norm(approx - exact, ord=ord)
    denominator = np.linalg.norm(exact, ord=ord)
    if denominator == 0.0:
        return 0.0 if numerator == 0.0 else float("inf")
    return float(numerator / denominator)


def _relative_norm(approx, exact):
    denominator = max(1.0, _norm(exact))
    return _norm(approx - exact) / denominator


def _norm(values):
    return float(np.linalg.norm(np.asarray(values, dtype=float).reshape(-1)))


def _row(category, metric, scale, value, available, notes=""):
    return {
        "category": category,
        "metric": metric,
        "scale": scale,
        "value": value,
        "available": bool(available),
        "notes": notes,
    }


def _format_value(value):
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.12g}"
    return str(value)


def _summary_rows(rows):
    rows_by_key = {
        (row.get("category"), row.get("metric"), row.get("scale")): row
        for row in rows
        if row.get("available")
    }
    truncation_scales = sorted(
        {
            row.get("scale")
            for row in rows
            if row.get("available")
            and row.get("category") == "truncation"
            and row.get("metric") == "sigmaT_squared_estimate"
            and isinstance(row.get("scale"), (float, int, np.floating))
        }
    )
    final_scale = 1.0 if 1.0 in truncation_scales else (truncation_scales[-1] if truncation_scales else None)

    def pick(category, metric, scale=""):
        row = rows_by_key.get((category, metric, scale))
        if row is None:
            return None
        return row.get("value")

    summary_specs = [
        ("W error", "W", "unitarity_fro_max", ""),
        ("dW error", "dW", "relative_first_minus_fd", ""),
        ("V fidelity error", "V", "fidelity_leading_minus_frechet", ""),
        ("truncation fidelity error", "truncation", "sigmaT_squared_estimate", final_scale),
        ("sigmaT squared estimate", "truncation", "sigmaT_squared_estimate", final_scale),
        ("optimization perturbative fidelity", "truncation", "perturbative_fidelity", final_scale),
    ]
    summary = []
    for item, category, metric, scale in summary_specs:
        value = pick(category, metric, scale)
        if value is not None:
            scale_suffix = f"[scale={scale}]" if scale != "" else ""
            summary.append((item, value, f"{category}/{metric}{scale_suffix}"))
    return summary
