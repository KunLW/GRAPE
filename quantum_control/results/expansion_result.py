from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class ExpansionState:
    """Order zero is (d,); independent multi-noise orders 1/2 are (n_noise, d).

    A zero/one-source expansion and the Lindblad path keep vectors at all
    orders. Multi-source order two stores same-channel insertions only.
    """
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
