from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class ExpansionState:
    components: dict[int, np.ndarray]

    def component(self, order):
        return self.components[order]


@dataclass(frozen=True)
class ExpansionResult:
    steps: list
    forward: list[ExpansionState]
    backward: list[ExpansionState] | None
    max_order: int
    metadata: dict = field(default_factory=dict)
