"""Evaluation metrics: reporting-side fidelities, decoupled from optimization.

``gate_metrics`` holds the fast perturbative metrics with fixed definitions
(comparable across runs regardless of objective settings);
``density_matrix`` holds the faithful evaluator (exact Lindblad propagation +
Gauss-Hermite fluctuation averaging) used to validate them.
"""

from quantum_control.evaluation.density_matrix import faithful_gate_fidelity
from quantum_control.evaluation.gate_metrics import (
    closed_gate_fidelity,
    noisy_gate_fidelity,
)

__all__ = [
    "closed_gate_fidelity",
    "faithful_gate_fidelity",
    "noisy_gate_fidelity",
]
