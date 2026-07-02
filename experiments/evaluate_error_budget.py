from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quantum_control import (  # noqa: E402
    ErrorBudgetConfig,
    evaluate_error_budget,
    load_pulse_npz,
    write_error_budget_report,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate perturbative fluctuation error budget for a pulse and system."
    )
    parser.add_argument("--pulse-npz", type=Path, required=True)
    parser.add_argument(
        "--system-factory",
        required=True,
        help="Python callable as module:function. It must return (system, state_pairs) or "
        "(system, state_pairs, metadata).",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fallback-dt", type=float, default=None)
    parser.add_argument("--gradient-samples", type=int, default=16)
    parser.add_argument("--finite-difference-epsilon", type=float, default=1e-6)
    parser.add_argument("--random-seed", type=int, default=12345)
    parser.add_argument(
        "--scales",
        type=float,
        nargs="+",
        default=[0.25, 0.5, 1.0],
        help="Fluctuation scales used for sigmaT truncation diagnostics.",
    )
    parser.add_argument(
        "--normalize-weights",
        action="store_true",
        help="Normalize state-pair weights before evaluating averaged fidelities.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    factory = _load_factory(args.system_factory)
    factory_result = factory()
    if len(factory_result) == 2:
        system, state_pairs = factory_result
        metadata = {}
    elif len(factory_result) == 3:
        system, state_pairs, metadata = factory_result
    else:
        raise ValueError("system factory must return (system, state_pairs) or (system, state_pairs, metadata).")

    pulse = load_pulse_npz(args.pulse_npz, fallback_dt=args.fallback_dt)
    config = ErrorBudgetConfig(
        finite_difference_epsilon=args.finite_difference_epsilon,
        gradient_samples=args.gradient_samples,
        fluctuation_scales=tuple(args.scales),
        random_seed=args.random_seed,
        normalize_weights=args.normalize_weights,
    )
    report = evaluate_error_budget(system, pulse, state_pairs, config)
    merged_metadata = {
        **report.metadata,
        "pulse_npz": str(args.pulse_npz),
        "system_factory": args.system_factory,
        **dict(metadata),
    }
    report = type(report)(rows=report.rows, metadata=merged_metadata)
    outputs = write_error_budget_report(report, args.output_dir)
    print(f"error_budget_md={outputs['markdown']}")
    print(f"error_budget_csv={outputs['csv']}")


def _load_factory(spec):
    if ":" not in spec:
        raise ValueError("--system-factory must have form module:function.")
    module_name, function_name = spec.split(":", 1)
    if str(Path.cwd()) not in sys.path:
        sys.path.insert(0, str(Path.cwd()))
    module = importlib.import_module(module_name)
    factory = getattr(module, function_name)
    if not callable(factory):
        raise TypeError(f"{spec} is not callable.")
    return factory


if __name__ == "__main__":
    os.environ.setdefault("PYTHONHASHSEED", "0")
    main()
