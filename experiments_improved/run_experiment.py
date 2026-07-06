from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field, fields, replace
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT_DIR / ".matplotlib"))
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from experiments.reporting import (
    FidelityTermsLog,
    StepLog,
    export_pulse_controls,
    timestamped_experiment_dir,
)
from quantum_control import (
    CombinedStateAverageProblem,
    ExpansionFidelity,
    ExpansionStateAverageFidelity,
    EvolutionContext,
    LindbladCorrectedStateFidelity,
    LindbladExpansionDifferentiator,
    LindbladExpansionEvolution,
    NominalUnitaryEvolution,
    ParameterSmoothPenalty,
    ParameterizedControlProblem,
    PenalizedParameterizedProblem,
    PerturbativeExpansionDifferentiator,
    PerturbativeExpansionEvolution,
    PerturbativeStepBuilder,
    StateTransferFidelity,
    UnitaryStepBuilder,
    closed_gate_fidelity,
    open_gate_fidelity,
)
from quantum_control.optimizers import ScipyOptimizer
from quantum_control.pulses.pulse import PiecewiseConstantPulse

from experiments_improved.config_io import load_experiment_config, write_config_snapshot
from experiments_improved.systems import get_system

N_STEPS = 200
MAXITER = 40
DEFAULT_L1_SMOOTH_WEIGHT = 0.0005
DEFAULT_L2_SMOOTH_WEIGHT = 0.0001


def _default_system_params():
    return get_system("spin_boson").default_params()


def _default_system_noise():
    return get_system("spin_boson").default_noise()


@dataclass(frozen=True)
class SystemSelection:
    """Pluggable physical system: registry key plus its params/noise configs.

    ``type`` selects the system definition (see ``systems/``); ``params`` and
    ``noise`` are that system's frozen dataclasses, so their YAML schema is
    owned by the system module rather than this driver.
    """

    type: str = "spin_boson"
    params: object = field(default_factory=_default_system_params)
    noise: object = field(default_factory=_default_system_noise)


@dataclass(frozen=True)
class PulseConfig:
    n_steps: int = N_STEPS
    total_time_us: float = 225.8
    random_seed: int | None = None


@dataclass(frozen=True)
class ObjectiveConfig:
    max_order: int = 2
    drop_odd_average: bool = True
    normalize_weights: bool = False


@dataclass(frozen=True)
class OptimizerConfig:
    method: str = "L-BFGS-B"
    maxiter: int = MAXITER
    gtol: float = 1e-12
    ftol: float = 1e-15
    maximize: bool = True

    @property
    def options(self):
        return {"maxiter": self.maxiter, "gtol": self.gtol, "ftol": self.ftol}


@dataclass(frozen=True)
class PenaltyConfig:
    l1_smooth_weight: float = DEFAULT_L1_SMOOTH_WEIGHT
    l2_smooth_weight: float = DEFAULT_L2_SMOOTH_WEIGHT


@dataclass(frozen=True)
class RuntimeConfig:
    workers: int = 1
    print_step: bool = False
    print_fidelity_terms: bool = False
    save_fidelity_terms: bool | None = None
    initial_pulse_npz: Path | None = None
    no_progress: bool = False

    @property
    def should_save_fidelity_terms(self):
        if self.save_fidelity_terms is None:
            return self.print_fidelity_terms
        return self.save_fidelity_terms


@dataclass(frozen=True)
class OutputConfig:
    output_root: Path = field(default_factory=lambda: OUTPUT_DIR)


@dataclass(frozen=True)
class ExperimentConfig:
    system: SystemSelection = field(default_factory=SystemSelection)
    pulse: PulseConfig = field(default_factory=PulseConfig)
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    penalty: PenaltyConfig = field(default_factory=PenaltyConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @property
    def n_steps(self):
        return self.pulse.n_steps

    @property
    def maxiter(self):
        return self.optimizer.maxiter

    @property
    def l1_smooth_weight(self):
        return self.penalty.l1_smooth_weight

    @property
    def l2_smooth_weight(self):
        return self.penalty.l2_smooth_weight

    @property
    def workers(self):
        return self.runtime.workers

    @property
    def print_step(self):
        return self.runtime.print_step

    @property
    def print_fidelity_terms(self):
        return self.runtime.print_fidelity_terms

    @property
    def save_fidelity_terms(self):
        return self.runtime.save_fidelity_terms

    @property
    def initial_pulse_npz(self):
        return self.runtime.initial_pulse_npz

    @property
    def no_progress(self):
        return self.runtime.no_progress

    @property
    def output_root(self):
        return self.output.output_root


def default_experiment_config():
    return ExperimentConfig()


def _with_fluctuations_enabled(system_selection, enabled):
    noise = system_selection.noise
    return replace(
        system_selection,
        noise=replace(
            noise,
            fluctuations=replace(noise.fluctuations, enabled=bool(enabled)),
        ),
    )


def no_fluctuation_experiment_config():
    config = default_experiment_config()
    return replace(
        config,
        system=_with_fluctuations_enabled(config.system, False),
    )


def _coerce_experiment_config(config):
    if config is None:
        return default_experiment_config()
    if isinstance(config, ExperimentConfig):
        return config

    defaults = default_experiment_config()
    system = defaults.system
    if hasattr(config, "include_fluctuations"):
        system = _with_fluctuations_enabled(system, config.include_fluctuations)
    rate_names = [
        field_info.name
        for field_info in fields(system.noise.decoherence)
        if field_info.name != "enabled"
    ]
    gamma_updates = {
        name: float(getattr(config, name))
        for name in rate_names
        if getattr(config, name, 0.0)
    }
    if gamma_updates:
        system = replace(
            system,
            noise=replace(
                system.noise,
                decoherence=replace(
                    system.noise.decoherence, enabled=True, **gamma_updates
                ),
            ),
        )
    return ExperimentConfig(
        system=system,
        pulse=replace(
            defaults.pulse,
            n_steps=int(getattr(config, "n_steps", defaults.pulse.n_steps)),
        ),
        objective=defaults.objective,
        optimizer=replace(
            defaults.optimizer,
            maxiter=int(getattr(config, "maxiter", defaults.optimizer.maxiter)),
        ),
        penalty=replace(
            defaults.penalty,
            l1_smooth_weight=float(
                getattr(config, "l1_smooth_weight", defaults.penalty.l1_smooth_weight)
            ),
            l2_smooth_weight=float(
                getattr(config, "l2_smooth_weight", defaults.penalty.l2_smooth_weight)
            ),
        ),
        runtime=replace(
            defaults.runtime,
            workers=int(getattr(config, "workers", defaults.runtime.workers)),
            print_step=bool(getattr(config, "print_step", defaults.runtime.print_step)),
            print_fidelity_terms=bool(
                getattr(config, "print_fidelity_terms", defaults.runtime.print_fidelity_terms)
            ),
            save_fidelity_terms=getattr(
                config,
                "save_fidelity_terms",
                defaults.runtime.save_fidelity_terms,
            ),
            initial_pulse_npz=getattr(
                config,
                "initial_pulse_npz",
                defaults.runtime.initial_pulse_npz,
            ),
            no_progress=bool(getattr(config, "no_progress", defaults.runtime.no_progress)),
        ),
        output=replace(
            defaults.output,
            output_root=Path(getattr(config, "output_root", defaults.output.output_root)),
        ),
    )


class OptimizationProgressBar:
    def __init__(self, maxiter, width=32):
        self.maxiter = max(1, int(maxiter))
        self.width = width
        self.iteration = 0

    def start(self):
        self._write()

    def __call__(self, _xk):
        self.iteration = min(self.iteration + 1, self.maxiter)
        self._write()

    def finish(self, result):
        final_iteration = min(int(getattr(result, "nit", self.iteration)), self.maxiter)
        if final_iteration != self.iteration:
            self.iteration = final_iteration
            self._write()
        sys.stderr.write("\n")
        sys.stderr.flush()

    def _write(self):
        filled = int(round(self.width * self.iteration / self.maxiter))
        bar = "#" * filled + "-" * (self.width - filled)
        sys.stderr.write(f"\roptimizing [{bar}] {self.iteration}/{self.maxiter}")
        sys.stderr.flush()


def _load_base_config(config_path):
    if config_path is None:
        return default_experiment_config()
    return load_experiment_config(config_path, default_experiment_config(), get_system)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Run a config-driven perturbative open-gate optimization; the "
            "physical system is selected by system.type (see experiments_improved/systems)."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML experiment configuration file; explicit flags override it.",
    )
    suppress = argparse.SUPPRESS
    parser.add_argument("--maxiter", type=int, default=suppress)
    parser.add_argument("--n-steps", type=int, default=suppress)
    parser.add_argument("--l1-smooth-weight", type=float, default=suppress)
    parser.add_argument("--l2-smooth-weight", type=float, default=suppress)
    parser.add_argument(
        "--workers",
        type=int,
        default=suppress,
        help="Number of worker processes for perturbative state-pair averaging.",
    )
    parser.add_argument(
        "--print-step",
        action="store_true",
        default=suppress,
        help="Print per-step close fidelity, open fidelity, and cost function.",
    )
    parser.add_argument(
        "--print-fidelity-terms",
        action="store_true",
        default=suppress,
        help="Print and save per-step perturbative fidelity term diagnostics.",
    )
    parser.add_argument(
        "--initial-pulse-npz",
        type=Path,
        default=suppress,
        help="Load a custom initial pulse .npz with an amplitudes array.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        default=suppress,
        help="Disable the optimization progress bar.",
    )
    parser.add_argument(
        "--close-grape",
        action="store_true",
        default=suppress,
        help="Run optimization without fluctuation terms.",
    )
    parser.add_argument(
        "--gamma-heating",
        type=float,
        default=suppress,
        help="Motional heating rate (rad/s) for the Lindblad channel I_spin ⊗ a†; implies decoherence enabled.",
    )
    parser.add_argument(
        "--gamma-motional-dephasing",
        type=float,
        default=suppress,
        help="Motional dephasing rate (rad/s) for the Lindblad channel I_spin ⊗ a†a; implies decoherence enabled.",
    )
    parser.add_argument(
        "--gamma-spin-dephasing",
        type=float,
        default=suppress,
        help="Collective spin dephasing rate (rad/s) for the Lindblad channel (sz⊗I + I⊗sz)/2; implies decoherence enabled.",
    )
    args = parser.parse_args(argv)
    return _apply_cli_overrides(_load_base_config(args.config), args)


def _apply_cli_overrides(base, args):
    def arg(name, fallback):
        return getattr(args, name, fallback)

    system = base.system
    gamma_names = ("gamma_heating", "gamma_motional_dephasing", "gamma_spin_dephasing")
    gamma_updates = {
        name: float(getattr(args, name)) for name in gamma_names if hasattr(args, name)
    }
    if gamma_updates:
        if system.type != "spin_boson":
            raise ValueError(
                "--gamma-* flags apply to the spin_boson system only; "
                f"config selects system type {system.type!r}."
            )
        system = replace(
            system,
            noise=replace(
                system.noise,
                decoherence=replace(
                    system.noise.decoherence, enabled=True, **gamma_updates
                ),
            ),
        )
    if getattr(args, "close_grape", False):
        system = _with_fluctuations_enabled(system, False)
    return replace(
        base,
        system=system,
        pulse=replace(
            base.pulse,
            n_steps=arg("n_steps", base.pulse.n_steps),
        ),
        optimizer=replace(
            base.optimizer,
            maxiter=arg("maxiter", base.optimizer.maxiter),
        ),
        penalty=replace(
            base.penalty,
            l1_smooth_weight=arg("l1_smooth_weight", base.penalty.l1_smooth_weight),
            l2_smooth_weight=arg("l2_smooth_weight", base.penalty.l2_smooth_weight),
        ),
        runtime=replace(
            base.runtime,
            workers=arg("workers", base.runtime.workers),
            print_step=arg("print_step", base.runtime.print_step),
            print_fidelity_terms=arg(
                "print_fidelity_terms", base.runtime.print_fidelity_terms
            ),
            initial_pulse_npz=arg("initial_pulse_npz", base.runtime.initial_pulse_npz),
            no_progress=arg("no_progress", base.runtime.no_progress),
        ),
    )


def parse_evaluate_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Evaluate a pulse against the configured system without running optimization."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML experiment configuration file; explicit flags override it.",
    )
    parser.add_argument(
        "--pulse-npz",
        type=Path,
        default=argparse.SUPPRESS,
        help="Pulse .npz to evaluate. If omitted, evaluate the configured initial pulse.",
    )
    parser.add_argument("--n-steps", type=int, default=argparse.SUPPRESS)
    parser.add_argument(
        "--workers",
        type=int,
        default=argparse.SUPPRESS,
        help="Number of worker processes for open-gate fidelity.",
    )
    args = parser.parse_args(argv)
    base = _load_base_config(args.config)
    return replace(
        base,
        pulse=replace(
            base.pulse,
            n_steps=getattr(args, "n_steps", base.pulse.n_steps),
        ),
        runtime=replace(
            base.runtime,
            workers=getattr(args, "workers", base.runtime.workers),
            initial_pulse_npz=getattr(args, "pulse_npz", base.runtime.initial_pulse_npz),
            no_progress=True,
        ),
    )


class CombinedCallback:
    def __init__(self, *callbacks):
        self.callbacks = tuple(callback for callback in callbacks if callback is not None)

    def __call__(self, parameters):
        for callback in self.callbacks:
            callback(parameters)


def propagate_states(evolution, system, pulse, context):
    result = evolution.evolve(system, pulse, context)
    states = [context.initial_state]
    state = context.initial_state
    for step in result.W_steps:
        state = step @ state
        states.append(state)
    return np.asarray(states)


def add_experiment_note(fig, note):
    if not note:
        return
    fig.text(
        0.685,
        0.785,
        note,
        ha="right",
        va="bottom",
        fontsize=8,
        family="monospace",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.7},
    )


def plot_pulses(time_us, initial_pulse, final_pulse, channels, output_path, note=None):
    n_channels = len(channels)
    fig, axes = plt.subplots(
        n_channels, 1, figsize=(9, 3 * n_channels), sharex=True, squeeze=False
    )
    for index, channel in enumerate(channels):
        axis = axes[index, 0]
        axis.plot(
            time_us,
            initial_pulse.amplitudes[:, index] * channel.display_scale,
            label=f"initial {channel.label}",
            linewidth=2,
        )
        axis.plot(
            time_us,
            final_pulse.amplitudes[:, index] * channel.display_scale,
            label=f"final {channel.label}",
            linewidth=2,
        )
        axis.set_ylabel(f"{channel.label} ({channel.display_unit})")
        axis.legend(loc="best")
        axis.grid(True, alpha=0.3)
    axes[-1, 0].set_xlabel("time (us)")

    fig.suptitle("Perturbative open-gate optimization: pulse parameters")
    add_experiment_note(fig, note)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_population_marginals(
    time_edges_us,
    initial_states,
    final_states,
    structure,
    output_path,
    note=None,
):
    initial_joint = (np.abs(initial_states) ** 2).reshape((-1, *structure.dims))
    final_joint = (np.abs(final_states) ** 2).reshape((-1, *structure.dims))
    # Marginal over the *other* subsystem: axis 2 traces out the second
    # subsystem (keeping the first) and vice versa.
    marginals = [
        (initial_joint.sum(axis=2), final_joint.sum(axis=2)),
        (initial_joint.sum(axis=1), final_joint.sum(axis=1)),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey="row")
    for row, ((initial_marginal, final_marginal), name, labels) in enumerate(
        zip(marginals, structure.names, structure.labels)
    ):
        for level, label in enumerate(labels):
            axes[row, 0].plot(time_edges_us, initial_marginal[:, level], label=label)
            axes[row, 1].plot(time_edges_us, final_marginal[:, level], label=label)
        axes[row, 0].set_title(f"{name.capitalize()} population, initial pulse")
        axes[row, 1].set_title(f"{name.capitalize()} population, optimized pulse")
        axes[row, 0].set_ylabel(f"{name} population")

    axes[1, 0].set_xlabel("time (us)")
    axes[1, 1].set_xlabel("time (us)")
    for axis in axes:
        for item in axis:
            item.grid(True, alpha=0.3)
            item.set_ylim(-0.02, 1.02)

    first_handles, first_labels = axes[0, 0].get_legend_handles_labels()
    second_handles, second_labels = axes[1, 0].get_legend_handles_labels()
    axes[0, 1].legend(first_handles, first_labels, loc="best")
    axes[1, 1].legend(second_handles, second_labels, loc="best", ncol=2)
    fig.suptitle("Nominal state propagation after perturbative optimization")
    add_experiment_note(fig, note)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def format_experiment_note(config, result, metrics):
    return "\n".join(
        [
            f"objective=open_gate_fidelity_expansion, system={config.system.type}",
            (
                f"n_steps={config.pulse.n_steps}, maxiter={config.optimizer.maxiter}, "
                f"workers={config.runtime.workers}"
            ),
            (
                f"smooth: l1={config.penalty.l1_smooth_weight:.6g}, "
                f"l2={config.penalty.l2_smooth_weight:.6g}"
            ),
            (
                f"open gate: {metrics['initial_open_gate_fidelity']:.6g} -> "
                f"{metrics['final_open_gate_fidelity']:.6g}; "
                f"close gate: {metrics['initial_close_gate_fidelity']:.6g} -> "
                f"{metrics['final_close_gate_fidelity']:.6g}"
            ),
            (
                f"objective: {metrics['initial_penalized_objective']:.6g} -> "
                f"{metrics['final_penalized_objective']:.6g}; "
                f"nit={getattr(result, 'nit', 'NA')}, nfev={getattr(result, 'nfev', 'NA')}"
            ),
        ]
    )


def print_section(title, rows):
    print(f"\n[{title}]")
    width = max(len(name) for name, _ in rows)
    for name, value in rows:
        print(f"{name:<{width}} : {value}")


def format_value(value):
    if isinstance(value, float):
        return f"{value:.12g}"
    return str(value)


def system_params_rows(params):
    """One ``(name, value)`` row per field of the system's params dataclass."""
    return [
        (field_info.name, format_value(getattr(params, field_info.name)))
        for field_info in fields(params)
    ]


def print_experiment_report(config, result, metrics, outputs):
    print("\n=== Perturbative Open-Gate Optimization ===")
    print_section(
        "Configuration",
        [
            ("objective", "open_gate_fidelity_expansion"),
            ("system_type", config.system.type),
            *system_params_rows(config.system.params),
            ("include_fluctuations", config.system.noise.fluctuations.enabled),
            ("n_steps", config.pulse.n_steps),
            ("maxiter", config.optimizer.maxiter),
            ("workers", config.runtime.workers),
            ("l1_smooth_weight", format_value(config.penalty.l1_smooth_weight)),
            ("l2_smooth_weight", format_value(config.penalty.l2_smooth_weight)),
        ],
    )
    fidelity_rows = []
    if "initial_fidelity" in metrics:
        fidelity_rows.append(
            ("single_state", _transition(metrics["initial_fidelity"], metrics["final_fidelity"]))
        )
    fidelity_rows.extend(
        [
            (
                "close_gate",
                _transition(
                    metrics["initial_close_gate_fidelity"],
                    metrics["final_close_gate_fidelity"],
                ),
            ),
            (
                "open_gate",
                _transition(
                    metrics["initial_open_gate_fidelity"],
                    metrics["final_open_gate_fidelity"],
                ),
            ),
        ]
    )
    print_section("Fidelity", fidelity_rows)
    print_section(
        "Objective And Penalty",
        [
            (
                "penalized_objective",
                _transition(
                    metrics["initial_penalized_objective"],
                    metrics["final_penalized_objective"],
                ),
            ),
            ("l1_penalty", _transition(metrics["initial_l1_penalty"], metrics["final_l1_penalty"])),
            ("l2_penalty", _transition(metrics["initial_l2_penalty"], metrics["final_l2_penalty"])),
        ],
    )
    print_section(
        "Optimizer",
        [
            ("success", result.success),
            ("message", result.message),
            ("nit", getattr(result, "nit", "NA")),
            ("nfev", getattr(result, "nfev", "NA")),
        ],
    )
    print_section(
        "Outputs",
        [
            ("pulse_plot", outputs["pulse_plot"]),
            *(
                [("propagation_plot", outputs["propagation_plot"])]
                if outputs.get("propagation_plot") is not None
                else []
            ),
            ("step_log", outputs["step_log"]),
            *(
                [
                    ("fidelity_terms", outputs["fidelity_terms"]),
                    ("fidelity_terms_by_pair", outputs["fidelity_terms_by_pair"]),
                ]
                if outputs.get("fidelity_terms") is not None
                else []
            ),
            ("initial_pulse_npz", outputs["initial_pulse_npz"]),
            ("initial_pulse_csv", outputs["initial_pulse_csv"]),
            ("final_pulse_npz", outputs["final_pulse_npz"]),
            ("final_pulse_csv", outputs["final_pulse_csv"]),
        ],
    )


def _transition(initial, final):
    delta = final - initial
    return f"{initial:.12g} -> {final:.12g} (delta {delta:+.3g})"


def control_bounds_rows(parameterization, pulse, channels):
    """One bounds row per control channel, in the channel's display unit."""
    lower, upper = parameterization.bounds_for(pulse.amplitudes.shape)
    return [
        (
            f"{channel.label}_bounds_{channel.display_unit}",
            f"{lower[0, index] * channel.display_scale:.12g} to "
            f"{upper[0, index] * channel.display_scale:.12g}",
        )
        for index, channel in enumerate(channels)
    ]


def write_optimization_preview_report(
    report_path,
    *,
    generated_at,
    experiment_dir,
    config,
    initial_pulse,
    parameterization,
    channels,
    noisy_system,
    noise_specs,
    state_pairs,
    kappa_metrics,
    custom_initial_metadata,
    extra_configuration,
):
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    if custom_initial_metadata is None:
        initial_pulse_source = "generated (built-in)"
    else:
        initial_pulse_source = Path(custom_initial_metadata["source_npz"]).name
    lines = [
        f"# Perturbative Open-Gate Optimization ({config.system.type})",
        "",
        f"Generated at: {generated_at.isoformat(timespec='seconds')}",
        "",
        "## Run Summary",
        "",
        _markdown_table(
            ("Parameter", "Value"),
            [
                ("experiment_dir", experiment_dir),
                ("objective", "open_gate_fidelity_expansion"),
                ("system_type", config.system.type),
                *system_params_rows(config.system.params),
                ("n_steps", config.pulse.n_steps),
                ("total_time_us", initial_pulse.n_steps * initial_pulse.dt * 1e6),
                ("include_fluctuations", config.system.noise.fluctuations.enabled),
                *control_bounds_rows(parameterization, initial_pulse, channels),
                ("max_order", config.objective.max_order),
                ("state_pair_count", len(state_pairs)),
                ("l1_smooth_weight", config.penalty.l1_smooth_weight),
                ("l2_smooth_weight", config.penalty.l2_smooth_weight),
                ("initial_pulse", initial_pulse_source),
                *extra_configuration,
            ],
        ),
        "",
        "## Validity (Perturbative Expansion)",
        "",
        _markdown_table(("Metric", "Value", "Definition"), validity_rows(kappa_metrics)),
        "",
        "## Noise Terms",
        "",
        _markdown_table(
            ("Term", "Coefficient", "Definition", "Usage", "Spectral Norm"),
            noise_term_rows(noisy_system, noise_specs),
        ),
        "",
        "## Results",
        "",
        "_Optimization has not completed yet. Final results will be appended here._",
        "",
    ]
    report_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return report_path


def append_optimization_results_report(
    report_path,
    *,
    metrics,
    result,
    figures,
    outputs,
    interrupted,
    optimizer_method,
    optimizer_options,
):
    report_path = Path(report_path)
    existing = report_path.read_text(encoding="utf-8") if report_path.exists() else ""
    placeholder = "## Results\n\n_Optimization has not completed yet. Final results will be appended here._\n"
    result_lines = [
        "## Results",
        "",
        _markdown_table(
            ("Metric", "Initial", "Final", "Delta"),
            [
                *(
                    [
                        _result_row(
                            "single_state_fidelity",
                            metrics["initial_fidelity"],
                            metrics["final_fidelity"],
                        )
                    ]
                    if "initial_fidelity" in metrics
                    else []
                ),
                _result_row(
                    "close_gate_fidelity",
                    metrics["initial_close_gate_fidelity"],
                    metrics["final_close_gate_fidelity"],
                ),
                _result_row(
                    "open_gate_fidelity",
                    metrics["initial_open_gate_fidelity"],
                    metrics["final_open_gate_fidelity"],
                ),
                _result_row(
                    "decoherence_correction",
                    metrics.get("initial_decoherence_correction", 0.0),
                    metrics.get("final_decoherence_correction", 0.0),
                ),
                _result_row("l1_penalty", metrics["initial_l1_penalty"], metrics["final_l1_penalty"]),
                _result_row("l2_penalty", metrics["initial_l2_penalty"], metrics["final_l2_penalty"]),
                _result_row(
                    "penalized_objective",
                    metrics["initial_penalized_objective"],
                    metrics["final_penalized_objective"],
                ),
            ],
        ),
        "",
        "## Optimizer",
        "",
        _markdown_table(
            ("Parameter", "Value"),
            [
                ("method", optimizer_method),
                ("options", optimizer_options),
                ("success", result.success),
                ("message", result.message),
                ("nit", getattr(result, "nit", "NA")),
                ("nfev", getattr(result, "nfev", "NA")),
                ("interrupted", interrupted),
            ],
        ),
        "",
        "## Key Files",
        "",
        _markdown_table(
            ("Output", "Path"),
            [
                ("final_pulse_npz", outputs["final_pulse_npz"]),
                ("final_pulse_csv", outputs["final_pulse_csv"]),
                ("initial_pulse_npz", outputs["initial_pulse_npz"]),
                ("step_log", outputs["step_log"]),
            ],
        ),
        "",
        "## Figures",
        "",
    ]
    for label, figure_path in figures:
        relative_path = Path(figure_path).relative_to(report_path.parent)
        result_lines.extend([f"### {_format_markdown_cell(label)}", "", f"![{label}]({relative_path.as_posix()})", ""])
    result_markdown = "\n".join(result_lines).rstrip() + "\n"
    if placeholder in existing:
        report_path.write_text(existing.replace(placeholder, result_markdown), encoding="utf-8")
    else:
        report_path.write_text(existing.rstrip() + "\n\n" + result_markdown, encoding="utf-8")
    return report_path


def noise_term_rows(system, specs):
    matrices = tuple(system.static_fluctuations) + tuple(system.control_fluctuations)
    rows = []
    for spec, matrix in zip(specs, matrices, strict=True):
        matrix = np.asarray(matrix)
        rows.append(
            (
                spec["name"],
                spec["coefficient"],
                spec["definition"],
                spec["usage"],
                float(np.linalg.norm(matrix, ord=2)),
            )
        )
    return rows


def calculate_kappa_metrics(system, noisy_system, pulse, parameterization, collapse_operators=()):
    lower, upper = parameterization.bounds_for(pulse.amplitudes.shape)
    channel_bounds = tuple(zip(lower[0], upper[0], strict=True))
    boundary_controls = np.asarray(
        np.meshgrid(*(bound for bound in channel_bounds), indexing="ij"),
        dtype=float,
    ).reshape(len(channel_bounds), -1).T
    nominal_norms = []
    fluctuation_norms = []
    has_fluctuations = bool(noisy_system.static_fluctuations or noisy_system.control_fluctuations)
    for controls in boundary_controls:
        nominal_norms.append(float(np.linalg.norm(system.nominal_hamiltonian(controls), ord=2)))
        fluctuation_norms.append(
            float(np.linalg.norm(noisy_system.fluctuation_hamiltonian(controls), ord=2))
            if has_fluctuations
            else 0.0
        )
    nominal_norms = np.asarray(nominal_norms, dtype=float)
    fluctuation_norms = np.asarray(fluctuation_norms, dtype=float)
    kappa_1_corner = int(np.argmax(nominal_norms)) if nominal_norms.size else 0
    kappa_2_corner = int(np.argmax(fluctuation_norms)) if fluctuation_norms.size else 0
    total_time = float(pulse.n_steps * pulse.dt)
    lindblad_norm = float(
        sum(
            np.linalg.norm(np.asarray(operator).conj().T @ np.asarray(operator), ord=2)
            for operator in collapse_operators
        )
    )
    return {
        "kappa_1": float(pulse.dt * nominal_norms[kappa_1_corner]),
        "kappa_2": float(total_time * fluctuation_norms[kappa_2_corner]),
        "kappa_3": float(total_time * lindblad_norm),
        "kappa_3_lindblad_norm": lindblad_norm,
        "kappa_1_corner": kappa_1_corner,
        "kappa_2_corner": kappa_2_corner,
        "kappa_1_alpha": tuple(float(value) for value in boundary_controls[kappa_1_corner]),
        "kappa_2_alpha": tuple(float(value) for value in boundary_controls[kappa_2_corner]),
        "kappa_1_h_norm": float(nominal_norms[kappa_1_corner]),
        "kappa_2_h_fluc_norm": float(fluctuation_norms[kappa_2_corner]),
        "dt_s": float(pulse.dt),
        "total_time_s": total_time,
        "boundary_corner_count": int(len(boundary_controls)),
    }


def validity_rows(metrics):
    rows = [
        (
            "kappa_1",
            metrics["kappa_1"],
            "dt * max_alpha ||H_nominal||_2 over bounds (nominal step size)",
        ),
        (
            "kappa_2",
            metrics["kappa_2"],
            "T * max_alpha ||H_fluctuation||_2 over bounds (expansion small parameter)",
        ),
    ]
    if metrics.get("kappa_3_lindblad_norm", 0.0) > 0.0:
        rows.append(
            (
                "kappa_3",
                metrics["kappa_3"],
                "T * sum_mu ||L_mu^dag L_mu||_2; first-order Lindblad correction requires kappa_3 < 1",
            )
        )
    return rows


def print_optimization_preview(report_path, config, initial_metrics, kappa_metrics):
    print("\n=== Optimization Preview ===")
    print_section(
        "Configuration",
        [
            ("system_type", config.system.type),
            ("include_fluctuations", config.system.noise.fluctuations.enabled),
            ("n_steps", config.pulse.n_steps),
            ("maxiter", config.optimizer.maxiter),
            ("workers", config.runtime.workers),
            ("l1_smooth_weight", format_value(config.penalty.l1_smooth_weight)),
            ("l2_smooth_weight", format_value(config.penalty.l2_smooth_weight)),
        ],
    )
    print_section("Initial Metrics", [(name, format_value(value)) for name, value in initial_metrics])
    print_section(
        "Kappa Diagnostics",
        [
            ("kappa_1", format_value(kappa_metrics["kappa_1"])),
            ("kappa_2", format_value(kappa_metrics["kappa_2"])),
            ("kappa_1_corner", kappa_metrics["kappa_1_corner"]),
            ("kappa_2_corner", kappa_metrics["kappa_2_corner"]),
        ],
    )
    print(f"preview_report={report_path}")


def _result_row(name, initial, final):
    return (name, _format_markdown_value(initial), _format_markdown_value(final), _format_markdown_value(final - initial))


def _markdown_table(headers, rows):
    table = [
        "| " + " | ".join(_format_markdown_cell(header) for header in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        table.append("| " + " | ".join(_format_markdown_cell(value) for value in row) + " |")
    return "\n".join(table)


def _format_markdown_cell(value):
    return str(_format_markdown_value(value)).replace("|", "\\|").replace("\n", "<br>")


def _format_markdown_value(value):
    if isinstance(value, Path):
        return value.name
    if isinstance(value, float):
        return f"{value:.12g}"
    if value is None:
        return "disabled"
    return value


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

    parameters = parameterization.to_parameters(amplitudes)
    if np.any(parameters < -1.0 - atol) or np.any(parameters > 1.0 + atol):
        raise ValueError(f"{npz_path} amplitudes exceed the configured parameter bounds.")
    parameters = np.clip(parameters, -1.0, 1.0)
    # Round-tripping through the parameterization applies its structural
    # constraints (e.g. frozen endpoints); a mismatch means the loaded pulse
    # violates them.
    projected = parameterization.to_physical(np.array(parameters, copy=True))
    if not np.allclose(projected, amplitudes, atol=1e-6):
        raise ValueError(
            f"{npz_path} amplitudes violate the parameterization constraints "
            "(e.g. fixed endpoint values)."
        )

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


def perturbative_fidelity_terms(system, pulse, state_pairs, max_order=2, drop_odd_average=True):
    step_builder = PerturbativeStepBuilder()
    objective = ExpansionFidelity(max_order=max_order, drop_odd_average=drop_odd_average)
    evolution = PerturbativeExpansionEvolution(step_builder, max_order=max_order)
    pair_rows = []
    for pair_index, pair in enumerate(state_pairs):
        context = EvolutionContext(
            initial_state=pair.initial_state,
            target_state=pair.target_state,
        )
        amplitudes = objective.amplitudes(evolution.evolve(system, pulse, context))
        a0 = amplitudes.get(0, 0.0 + 0.0j)
        a1 = amplitudes.get(1, 0.0 + 0.0j)
        a2 = amplitudes.get(2, 0.0 + 0.0j)
        weight = float(pair.weight)
        closed_term = weight * float(np.abs(a0) ** 2)
        first_order_sq = weight * float(np.abs(a1) ** 2)
        second_order_cross = weight * float(2.0 * np.real(np.conj(a0) * a2))
        dropped_order1_cross = weight * float(2.0 * np.real(np.conj(a0) * a1))
        pair_rows.append(
            {
                "pair_index": pair_index,
                "weight": weight,
                "a0_real": float(np.real(a0)),
                "a0_imag": float(np.imag(a0)),
                "a1_real": float(np.real(a1)),
                "a1_imag": float(np.imag(a1)),
                "a2_real": float(np.real(a2)),
                "a2_imag": float(np.imag(a2)),
                "closed_term": closed_term,
                "first_order_sq": first_order_sq,
                "second_order_cross": second_order_cross,
                "perturbative_open": closed_term + first_order_sq + second_order_cross,
                "dropped_order1_cross": dropped_order1_cross,
            }
        )

    closed_term = float(sum(row["closed_term"] for row in pair_rows))
    first_order_sq = float(sum(row["first_order_sq"] for row in pair_rows))
    second_order_cross = float(sum(row["second_order_cross"] for row in pair_rows))
    perturbative_open = closed_term + first_order_sq + second_order_cross
    pair_open_values = [row["perturbative_open"] for row in pair_rows]
    summary = {
        "closed_term": closed_term,
        "first_order_sq": first_order_sq,
        "second_order_cross": second_order_cross,
        "perturbative_open": perturbative_open,
        "correction": first_order_sq + second_order_cross,
        "excess_over_1": perturbative_open - 1.0,
        "max_pair_open": float(max(pair_open_values)),
        "min_pair_open": float(min(pair_open_values)),
    }
    return summary, pair_rows


def system_definition(config):
    return get_system(config.system.type)


def build_systems(config):
    return system_definition(config).build_systems(
        config.system.params, config.system.noise
    )


def build_initial_pulse(config):
    return system_definition(config).build_initial_pulse(
        config.system.params, config.pulse
    )


def build_parameterization(config, pulse):
    return system_definition(config).build_parameterization(
        config.system.params, pulse
    )


def build_collapse_operators(config):
    return system_definition(config).build_collapse_operators(
        config.system.params, config.system.noise
    )


def build_target_gate(config):
    return system_definition(config).target_gate(config.system.params)


def build_state_pairs(config):
    return system_definition(config).state_pairs(config.system.params)


def build_decoherence_correction_problem(
    config,
    noisy_system,
    pulse,
    state_pairs,
    collapse_operators,
    include_closed=False,
    n_workers=None,
):
    step_builder = UnitaryStepBuilder()
    return ExpansionStateAverageFidelity(
        system=noisy_system,
        pulse=pulse,
        evolution=LindbladExpansionEvolution(
            step_builder,
            collapse_operators=collapse_operators,
        ),
        objective=LindbladCorrectedStateFidelity(include_closed=include_closed),
        differentiator=LindbladExpansionDifferentiator(
            step_builder,
            include_closed=include_closed,
        ),
        state_pairs=state_pairs,
        normalize_weights=config.objective.normalize_weights,
        n_workers=config.runtime.workers if n_workers is None else n_workers,
    )


def build_objective_problem(config, noisy_system, initial_pulse, state_pairs):
    step_builder = PerturbativeStepBuilder()
    expansion_objective = ExpansionFidelity(
        max_order=config.objective.max_order,
        drop_odd_average=config.objective.drop_odd_average,
    )
    expansion_problem = ExpansionStateAverageFidelity(
        system=noisy_system,
        pulse=initial_pulse,
        evolution=PerturbativeExpansionEvolution(
            step_builder,
            max_order=config.objective.max_order,
        ),
        objective=expansion_objective,
        differentiator=PerturbativeExpansionDifferentiator(
            step_builder,
            expansion_objective,
        ),
        state_pairs=state_pairs,
        normalize_weights=config.objective.normalize_weights,
        n_workers=config.runtime.workers,
    )
    collapse_operators = build_collapse_operators(config)
    if not collapse_operators:
        return expansion_problem
    decoherence_problem = build_decoherence_correction_problem(
        config,
        noisy_system,
        initial_pulse,
        state_pairs,
        collapse_operators,
        include_closed=False,
    )
    return CombinedStateAverageProblem(expansion_problem, decoherence_problem)


def build_optimizer(config):
    return ScipyOptimizer(
        method=config.optimizer.method,
        maximize=config.optimizer.maximize,
        options=config.optimizer.options,
    )


def run_perturbative_experiment(
    config=None,
    initial_parameters=None,
    run_label=None,
    output_root=None,
    experiment_dir=None,
    generated_at=None,
    extra_configuration=(),
    print_report=True,
):
    config = _coerce_experiment_config(config)
    definition = system_definition(config)
    params = config.system.params
    channels = definition.control_channels(params)
    channel_names = tuple(channel.label for channel in channels)
    # export_pulse_controls divides amplitudes by this to get display units;
    # assumes a common display scale across channels.
    export_unit_divisor = 1.0 / channels[0].display_scale
    system, noisy_system, noise_specs = build_systems(config)
    initial_pulse = build_initial_pulse(config)
    parameterization = build_parameterization(config, initial_pulse)
    custom_initial_metadata = None
    if initial_parameters is None and config.runtime.initial_pulse_npz is not None:
        initial_parameters, custom_initial_metadata = load_custom_initial_parameters(
            config.runtime.initial_pulse_npz,
            initial_pulse,
            parameterization,
        )
        for warning in custom_initial_metadata["warnings"]:
            print(warning, file=sys.stderr, flush=True)
    state_pairs = build_state_pairs(config)
    collapse_operators = build_collapse_operators(config)
    optimization_problem = build_objective_problem(config, noisy_system, initial_pulse, state_pairs)
    parameterized_problem = ParameterizedControlProblem(
        optimization_problem,
        parameterization,
    )
    penalty = ParameterSmoothPenalty(
        l1_weight=config.penalty.l1_smooth_weight,
        l2_weight=config.penalty.l2_smooth_weight,
    )
    penalized_problem = PenalizedParameterizedProblem(parameterized_problem, penalty)
    initial_parameters = (
        penalized_problem.initial_parameters()
        if initial_parameters is None
        else np.asarray(initial_parameters, dtype=float).reshape(
            penalized_problem.parameter_shape
        )
    )
    initial_objective = penalized_problem.value(initial_parameters)
    initial_l1_penalty = penalty.l1_value(
        initial_parameters,
        penalized_problem.parameter_shape,
    )
    initial_l2_penalty = penalty.l2_value(
        initial_parameters,
        penalized_problem.parameter_shape,
    )
    masked_initial_pulse = penalized_problem.pulse_from_parameters(initial_parameters)
    generated_at = generated_at or datetime.now()
    output_root = Path(output_root) if output_root is not None else config.output.output_root
    experiment_dir = Path(experiment_dir) if experiment_dir is not None else timestamped_experiment_dir(
        output_root,
        run_label or f"{config.system.type}_perturbative",
        generated_at,
    )
    experiment_dir.mkdir(parents=True, exist_ok=True)
    config_snapshot_path = write_config_snapshot(config, experiment_dir / "config.yaml")

    optimizer_options = config.optimizer.options
    optimizer = build_optimizer(config)
    progress = (
        None
        if config.runtime.no_progress or config.runtime.print_step
        else OptimizationProgressBar(config.optimizer.maxiter)
    )
    step_log_path = experiment_dir / "step_log.csv"
    fidelity_terms_path = experiment_dir / "fidelity_terms.csv"
    fidelity_pair_terms_path = experiment_dir / "fidelity_terms_by_pair.csv"
    latest_pulse_stem = experiment_dir / "latest_pulse"
    latest_parameters_path = experiment_dir / "latest_parameters.npz"
    pulse_path = experiment_dir / f"{config.system.type}_perturbative_pulses.png"
    propagation_path = experiment_dir / f"{config.system.type}_perturbative_state_propagation.png"
    initial_pulse_stem = experiment_dir / "initial_pulse"
    final_pulse_stem = experiment_dir / "final_pulse"
    report_path = experiment_dir / "report.md"
    step_log = StepLog(step_log_path, print_steps=config.runtime.print_step)
    save_fidelity_terms = config.runtime.should_save_fidelity_terms
    fidelity_terms_log = (
        FidelityTermsLog(
            fidelity_terms_path,
            fidelity_pair_terms_path,
            print_steps=config.runtime.print_fidelity_terms,
        )
        if save_fidelity_terms
        else None
    )
    step_counter = {"value": 0}
    latest_state = {
        "step": 0,
        "parameters": np.asarray(initial_parameters, dtype=float),
        "pulse": masked_initial_pulse,
    }
    initial_preview_metrics = [
        ("initial_penalized_objective", initial_objective),
        ("initial_raw_fidelity", penalized_problem.raw_value(initial_parameters)),
        (
            "initial_close_gate_fidelity",
            closed_gate_fidelity(system, masked_initial_pulse, state_pairs),
        ),
        (
            "initial_open_gate_fidelity",
            open_gate_fidelity(
                noisy_system,
                masked_initial_pulse,
                state_pairs,
                n_workers=config.runtime.workers,
            ),
        ),
        ("initial_l1_penalty", initial_l1_penalty),
        ("initial_l2_penalty", initial_l2_penalty),
    ]

    def decoherence_correction(pulse):
        if not collapse_operators:
            return 0.0
        correction_problem = build_decoherence_correction_problem(
            config,
            noisy_system,
            pulse,
            state_pairs,
            collapse_operators,
            include_closed=False,
            n_workers=1,
        )
        try:
            return correction_problem.value(pulse)
        finally:
            correction_problem.shutdown()

    if collapse_operators:
        initial_preview_metrics.append(
            ("initial_decoherence_correction", decoherence_correction(masked_initial_pulse))
        )
    kappa_metrics = calculate_kappa_metrics(
        system,
        noisy_system,
        masked_initial_pulse,
        parameterization,
        collapse_operators=collapse_operators,
    )
    write_optimization_preview_report(
        report_path,
        generated_at=generated_at,
        experiment_dir=experiment_dir,
        config=config,
        initial_pulse=initial_pulse,
        parameterization=parameterization,
        channels=channels,
        noisy_system=noisy_system,
        noise_specs=noise_specs,
        state_pairs=state_pairs,
        kappa_metrics=kappa_metrics,
        custom_initial_metadata=custom_initial_metadata,
        extra_configuration=extra_configuration,
    )
    if print_report or config.runtime.print_step or config.runtime.print_fidelity_terms:
        print_optimization_preview(report_path, config, initial_preview_metrics, kappa_metrics)

    def record_step(step, parameters):
        pulse = penalized_problem.pulse_from_parameters(parameters)
        l1_penalty = penalty.l1_value(parameters, penalized_problem.parameter_shape)
        l2_penalty = penalty.l2_value(parameters, penalized_problem.parameter_shape)
        step_log.append(
            step=step,
            close_fidelity=closed_gate_fidelity(system, pulse, state_pairs),
            open_fidelity=open_gate_fidelity(
                noisy_system,
                pulse,
                state_pairs,
                n_workers=config.runtime.workers,
            ),
            cost_function=penalized_problem.value(parameters),
            raw_fidelity=penalized_problem.raw_value(parameters),
            l1_penalty=l1_penalty,
            l2_penalty=l2_penalty,
            gradient_norm=np.linalg.norm(penalized_problem.gradient(parameters)),
        )
        if fidelity_terms_log is not None:
            fidelity_summary, fidelity_pair_rows = perturbative_fidelity_terms(
                noisy_system,
                pulse,
                state_pairs,
                max_order=config.objective.max_order,
                drop_odd_average=config.objective.drop_odd_average,
            )
            fidelity_summary["step"] = int(step)
            for row in fidelity_pair_rows:
                row["step"] = int(step)
            fidelity_terms_log.append(fidelity_summary, fidelity_pair_rows)
        latest_pulse_npz_path, latest_pulse_csv_path = export_pulse_controls(
            pulse,
            latest_pulse_stem,
            export_unit_divisor,
            channel_names=channel_names,
        )
        np.savez(
            latest_parameters_path,
            parameters=np.asarray(parameters, dtype=float),
            step=int(step),
            pulse_npz=str(latest_pulse_npz_path),
            pulse_csv=str(latest_pulse_csv_path),
        )
        latest_state["step"] = int(step)
        latest_state["parameters"] = np.asarray(parameters, dtype=float).reshape(
            penalized_problem.parameter_shape
        )
        latest_state["pulse"] = pulse

    def step_callback(parameters):
        step_counter["value"] += 1
        record_step(step_counter["value"], parameters)

    callback = CombinedCallback(progress, step_callback)
    record_step(0, initial_parameters)
    if progress is not None:
        progress.start()
    interrupted = False
    try:
        try:
            result = optimizer.optimize_parameters(
                penalized_problem,
                initial_parameters=initial_parameters,
                callback=callback,
            )
            final_pulse = result.optimized_pulse
            final_parameters = result.x.reshape(penalized_problem.parameter_shape)
        except KeyboardInterrupt:
            interrupted = True
            final_parameters = np.asarray(latest_state["parameters"], dtype=float).reshape(
                penalized_problem.parameter_shape
            )
            final_pulse = latest_state["pulse"]
            result = SimpleNamespace(
                success=False,
                message=(
                    "INTERRUPTED: optimization stopped by user; "
                    "using latest accepted pulse for report."
                ),
                nit=int(latest_state["step"]),
                nfev="NA",
                x=final_parameters.reshape(-1),
                optimized_pulse=final_pulse,
            )
            print(
                "\ninterrupted=True; using latest accepted pulse for final report.",
                file=sys.stderr,
                flush=True,
            )
        final_objective = penalized_problem.value(final_parameters)
    finally:
        optimization_problem.shutdown()
    if progress is not None:
        progress.finish(result)
    final_l1_penalty = penalty.l1_value(
        final_parameters,
        penalized_problem.parameter_shape,
    )
    final_l2_penalty = penalty.l2_value(
        final_parameters,
        penalized_problem.parameter_shape,
    )

    for pulse_label, checked_pulse in (
        ("Initial", masked_initial_pulse),
        ("Optimized", final_pulse),
    ):
        # Round-tripping applies the parameterization's structural
        # constraints (e.g. frozen endpoints); a mismatch means the pulse
        # escaped them during optimization.
        projected = parameterization.to_physical(
            parameterization.to_parameters(checked_pulse.amplitudes)
        )
        if not np.allclose(projected, checked_pulse.amplitudes, atol=1e-6):
            raise RuntimeError(
                f"{pulse_label} pulse violates the parameterization constraints."
            )

    initial_close_gate_fidelity = closed_gate_fidelity(
        system,
        masked_initial_pulse,
        state_pairs,
    )
    final_close_gate_fidelity = closed_gate_fidelity(
        system,
        final_pulse,
        state_pairs,
    )
    initial_open_gate_fidelity = open_gate_fidelity(
        noisy_system,
        masked_initial_pulse,
        state_pairs,
        n_workers=config.runtime.workers,
    )
    final_open_gate_fidelity = open_gate_fidelity(
        noisy_system,
        final_pulse,
        state_pairs,
        n_workers=config.runtime.workers,
    )

    lower, upper = parameterization.bounds_for(final_pulse.amplitudes.shape)
    bound_tolerance = 1e-8
    if np.any(final_pulse.amplitudes < lower - bound_tolerance) or np.any(
        final_pulse.amplitudes > upper + bound_tolerance
    ):
        raise RuntimeError("Optimized pulse violates amplitude bounds.")

    probe = definition.probe_state_pair(params)
    structure = definition.population_structure(params)
    probe_metrics = {}
    initial_states = final_states = None
    if probe is not None:
        context = EvolutionContext(
            initial_state=probe.initial_state,
            target_state=probe.target_state,
        )
        nominal_evolution = NominalUnitaryEvolution(UnitaryStepBuilder())
        probe_metrics["initial_fidelity"] = StateTransferFidelity(context.target_state).evaluate(
            nominal_evolution.evolve(system, masked_initial_pulse, context)
        )
        probe_metrics["final_fidelity"] = StateTransferFidelity(context.target_state).evaluate(
            nominal_evolution.evolve(system, final_pulse, context)
        )
        initial_states = propagate_states(nominal_evolution, system, masked_initial_pulse, context)
        final_states = propagate_states(nominal_evolution, system, final_pulse, context)
        if not np.allclose(np.sum(np.abs(initial_states) ** 2, axis=1), 1.0, atol=1e-8):
            raise RuntimeError("Initial propagation populations are not normalized.")
        if not np.allclose(np.sum(np.abs(final_states) ** 2, axis=1), 1.0, atol=1e-8):
            raise RuntimeError("Final propagation populations are not normalized.")

    time_us = (np.arange(initial_pulse.n_steps) + 0.5) * initial_pulse.dt * 1e6
    time_edges_us = np.arange(initial_pulse.n_steps + 1) * initial_pulse.dt * 1e6
    initial_pulse_npz_path, initial_pulse_csv_path = export_pulse_controls(
        masked_initial_pulse,
        initial_pulse_stem,
        export_unit_divisor,
        channel_names=channel_names,
    )
    final_pulse_npz_path, final_pulse_csv_path = export_pulse_controls(
        final_pulse,
        final_pulse_stem,
        export_unit_divisor,
        channel_names=channel_names,
    )

    initial_decoherence_correction = decoherence_correction(masked_initial_pulse)
    final_decoherence_correction = decoherence_correction(final_pulse)

    metrics = {
        **probe_metrics,
        "initial_close_gate_fidelity": initial_close_gate_fidelity,
        "final_close_gate_fidelity": final_close_gate_fidelity,
        "initial_open_gate_fidelity": initial_open_gate_fidelity,
        "final_open_gate_fidelity": final_open_gate_fidelity,
        "initial_decoherence_correction": initial_decoherence_correction,
        "final_decoherence_correction": final_decoherence_correction,
        "initial_l1_penalty": initial_l1_penalty,
        "final_l1_penalty": final_l1_penalty,
        "initial_l2_penalty": initial_l2_penalty,
        "final_l2_penalty": final_l2_penalty,
        "initial_penalized_objective": initial_objective,
        "final_penalized_objective": final_objective,
    }
    experiment_note = format_experiment_note(config, result, metrics)
    plot_pulses(time_us, masked_initial_pulse, final_pulse, channels, pulse_path, note=experiment_note)
    has_propagation_plot = structure is not None and initial_states is not None
    if has_propagation_plot:
        plot_population_marginals(
            time_edges_us,
            initial_states,
            final_states,
            structure,
            propagation_path,
            note=experiment_note,
        )

    for path in (
        pulse_path,
        *((propagation_path,) if has_propagation_plot else ()),
        step_log_path,
        latest_pulse_stem.with_suffix(".npz"),
        latest_pulse_stem.with_suffix(".csv"),
        latest_parameters_path,
        initial_pulse_npz_path,
        initial_pulse_csv_path,
        final_pulse_npz_path,
        final_pulse_csv_path,
    ):
        if not path.exists() or path.stat().st_size == 0:
            raise RuntimeError(f"Expected non-empty output at {path}.")
    if fidelity_terms_log is not None:
        for path in (fidelity_terms_path, fidelity_pair_terms_path):
            if not path.exists() or path.stat().st_size == 0:
                raise RuntimeError(f"Expected non-empty fidelity diagnostics at {path}.")

    outputs = {
        "config_snapshot": config_snapshot_path,
        "pulse_plot": pulse_path,
        "propagation_plot": propagation_path if has_propagation_plot else None,
        "step_log": step_log_path,
        "fidelity_terms": fidelity_terms_path if save_fidelity_terms else None,
        "fidelity_terms_by_pair": fidelity_pair_terms_path if save_fidelity_terms else None,
        "latest_pulse_npz": latest_pulse_stem.with_suffix(".npz"),
        "latest_pulse_csv": latest_pulse_stem.with_suffix(".csv"),
        "latest_parameters": latest_parameters_path,
        "initial_pulse_npz": initial_pulse_npz_path,
        "initial_pulse_csv": initial_pulse_csv_path,
        "final_pulse_npz": final_pulse_npz_path,
        "final_pulse_csv": final_pulse_csv_path,
    }
    append_optimization_results_report(
        report_path,
        metrics=metrics,
        result=result,
        figures=[
            ("Pulse parameters", pulse_path),
            *(
                [("State propagation", propagation_path)]
                if has_propagation_plot
                else []
            ),
        ],
        outputs=outputs,
        interrupted=interrupted,
        optimizer_method=config.optimizer.method,
        optimizer_options=optimizer_options,
    )
    if print_report:
        print_experiment_report(config, result, metrics, outputs)
        print(f"experiment_dir={experiment_dir}")
        print(f"step_log={step_log_path}")
        if save_fidelity_terms:
            print(f"fidelity_terms={fidelity_terms_path}")
            print(f"fidelity_terms_by_pair={fidelity_pair_terms_path}")
        print(f"latest_pulse_npz={latest_pulse_stem.with_suffix('.npz')}")
        print(f"latest_pulse_csv={latest_pulse_stem.with_suffix('.csv')}")
        print(f"latest_parameters={latest_parameters_path}")
        print(f"initial_pulse_npz={initial_pulse_npz_path}")
        print(f"initial_pulse_csv={initial_pulse_csv_path}")
        print(f"final_pulse_npz={final_pulse_npz_path}")
        print(f"final_pulse_csv={final_pulse_csv_path}")
        print(f"markdown_report={report_path}")

    return {
        "args": config,
        "config": config,
        "result": result,
        "metrics": metrics,
        "outputs": outputs,
        "experiment_dir": experiment_dir,
        "report_path": report_path,
        "initial_parameters": initial_parameters,
        "final_parameters": final_parameters,
        "custom_initial_metadata": custom_initial_metadata,
        "interrupted": interrupted,
    }


def custom_initial_configuration(metadata):
    if metadata is None:
        return ()
    return (
        ("initial_pulse_source", "custom_npz"),
        ("source_npz", metadata["source_npz"]),
        ("source_dt", metadata["source_dt"]),
        ("experiment_dt", metadata["experiment_dt"]),
        ("dt_missing", metadata["dt_missing"]),
        ("dt_mismatch", metadata["dt_mismatch"]),
    )


def write_evaluation_report(
    report_path,
    *,
    generated_at,
    experiment_dir,
    config,
    pulse_source,
    metrics,
    outputs,
    custom_initial_metadata,
):
    lines = [
        f"# Pulse Evaluation ({config.system.type})",
        "",
        f"Generated at: {generated_at.isoformat(timespec='seconds')}",
        "",
        "## Configuration",
        "",
        _markdown_table(
            ("Parameter", "Value"),
            [
                ("experiment_dir", experiment_dir),
                ("pulse_source", pulse_source),
                ("system_type", config.system.type),
                *system_params_rows(config.system.params),
                ("n_steps", config.pulse.n_steps),
                ("include_fluctuations", config.system.noise.fluctuations.enabled),
                ("workers", config.runtime.workers),
                *custom_initial_configuration(custom_initial_metadata),
            ],
        ),
        "",
        "## Metrics",
        "",
        _markdown_table(("Metric", "Value"), metrics.items()),
        "",
        "## Outputs",
        "",
        _markdown_table(
            ("Output", "Path"),
            [(name, path) for name, path in outputs.items() if path is not None],
        ),
        "",
        *(
            (
                "## Figures",
                "",
                f"![State propagation]({Path(outputs['propagation_plot']).name})",
                "",
            )
            if outputs.get("propagation_plot") is not None
            else ()
        ),
    ]
    report_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return report_path


def evaluate_pulse(config=None, *, experiment_dir=None, generated_at=None, print_report=True):
    config = _coerce_experiment_config(config)
    definition = system_definition(config)
    params = config.system.params
    channels = definition.control_channels(params)
    channel_names = tuple(channel.label for channel in channels)
    export_unit_divisor = 1.0 / channels[0].display_scale
    system, noisy_system, _noise_specs = build_systems(config)
    reference_pulse = build_initial_pulse(config)
    parameterization = build_parameterization(config, reference_pulse)
    custom_initial_metadata = None

    if config.runtime.initial_pulse_npz is None:
        pulse_source = "initial_pulse"
        parameters = parameterization.to_parameters(reference_pulse.amplitudes)
    else:
        pulse_source = str(config.runtime.initial_pulse_npz)
        parameters, custom_initial_metadata = load_custom_initial_parameters(
            config.runtime.initial_pulse_npz,
            reference_pulse,
            parameterization,
        )
        for warning in custom_initial_metadata["warnings"]:
            print(warning, file=sys.stderr, flush=True)

    evaluated_pulse = PiecewiseConstantPulse(
        amplitudes=parameterization.to_physical(parameters),
        dt=reference_pulse.dt,
    )
    reference_parameters = parameterization.to_parameters(reference_pulse.amplitudes)
    masked_reference_pulse = PiecewiseConstantPulse(
        amplitudes=parameterization.to_physical(reference_parameters),
        dt=reference_pulse.dt,
    )

    state_pairs = build_state_pairs(config)
    close_fidelity = closed_gate_fidelity(system, evaluated_pulse, state_pairs)
    open_fidelity = open_gate_fidelity(
        noisy_system,
        evaluated_pulse,
        state_pairs,
        n_workers=config.runtime.workers,
    )
    probe = definition.probe_state_pair(params)
    structure = definition.population_structure(params)
    state_fidelity = None
    reference_states = evaluated_states = None
    if probe is not None:
        context = EvolutionContext(
            initial_state=probe.initial_state,
            target_state=probe.target_state,
        )
        nominal_evolution = NominalUnitaryEvolution(UnitaryStepBuilder())
        state_fidelity = StateTransferFidelity(context.target_state).evaluate(
            nominal_evolution.evolve(system, evaluated_pulse, context)
        )
        reference_states = propagate_states(nominal_evolution, system, masked_reference_pulse, context)
        evaluated_states = propagate_states(nominal_evolution, system, evaluated_pulse, context)
        if not np.allclose(np.sum(np.abs(evaluated_states) ** 2, axis=1), 1.0, atol=1e-8):
            raise RuntimeError("Evaluation propagation populations are not normalized.")

    generated_at = generated_at or datetime.now()
    experiment_dir = Path(experiment_dir) if experiment_dir is not None else timestamped_experiment_dir(
        config.output.output_root,
        f"{config.system.type}_evaluation",
        generated_at,
    )
    experiment_dir.mkdir(parents=True, exist_ok=True)
    config_snapshot_path = write_config_snapshot(config, experiment_dir / "config.yaml")
    pulse_stem = experiment_dir / "evaluated_pulse"
    pulse_npz_path, pulse_csv_path = export_pulse_controls(
        evaluated_pulse,
        pulse_stem,
        export_unit_divisor,
        channel_names=channel_names,
    )
    propagation_path = experiment_dir / "eva_state_propagation.png"
    report_path = experiment_dir / "eva_report.md"
    time_edges_us = np.arange(evaluated_pulse.n_steps + 1) * evaluated_pulse.dt * 1e6
    has_propagation_plot = structure is not None and evaluated_states is not None
    if has_propagation_plot:
        plot_population_marginals(
            time_edges_us,
            reference_states,
            evaluated_states,
            structure,
            propagation_path,
            note=f"pulse_source={Path(pulse_source).name if pulse_source != 'initial_pulse' else pulse_source}",
        )

    metrics = {
        **({"state_fidelity": state_fidelity} if state_fidelity is not None else {}),
        "close_gate_fidelity": close_fidelity,
        "open_gate_fidelity": open_fidelity,
    }
    collapse_operators = build_collapse_operators(config)
    if collapse_operators:
        correction_problem = build_decoherence_correction_problem(
            config,
            noisy_system,
            evaluated_pulse,
            state_pairs,
            collapse_operators,
            include_closed=False,
            n_workers=1,
        )
        try:
            metrics["decoherence_correction"] = correction_problem.value(evaluated_pulse)
        finally:
            correction_problem.shutdown()
    outputs = {
        "config_snapshot": config_snapshot_path,
        "evaluated_pulse_npz": pulse_npz_path,
        "evaluated_pulse_csv": pulse_csv_path,
        "propagation_plot": propagation_path if has_propagation_plot else None,
        "eva_report": report_path,
    }
    write_evaluation_report(
        report_path,
        generated_at=generated_at,
        experiment_dir=experiment_dir,
        config=config,
        pulse_source=pulse_source,
        metrics=metrics,
        outputs=outputs,
        custom_initial_metadata=custom_initial_metadata,
    )
    for path in outputs.values():
        if path is not None and (not path.exists() or path.stat().st_size == 0):
            raise RuntimeError(f"Expected non-empty evaluation output at {path}.")
    if print_report:
        print("\n=== Pulse Evaluation ===")
        print_section("Metrics", [(name, format_value(value)) for name, value in metrics.items()])
        print(f"experiment_dir={experiment_dir}")
        print(f"evaluated_pulse_npz={pulse_npz_path}")
        print(f"evaluated_pulse_csv={pulse_csv_path}")
        if has_propagation_plot:
            print(f"propagation_plot={propagation_path}")
        print(f"eva_report={report_path}")

    return {
        "config": config,
        "metrics": metrics,
        "outputs": outputs,
        "experiment_dir": experiment_dir,
        "report_path": report_path,
        "pulse": evaluated_pulse,
        "custom_initial_metadata": custom_initial_metadata,
    }


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "evaluate":
        evaluate_pulse(parse_evaluate_args(sys.argv[2:]))
    else:
        run_perturbative_experiment(parse_args())


if __name__ == "__main__":
    main()
