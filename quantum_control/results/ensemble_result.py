from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class EnsembleResult:
    scenario_results: list
    weights: list[float]
    metadata: dict = field(default_factory=dict)
