from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EvolutionContext:
    initial_state: np.ndarray
    target_state: np.ndarray | None = None
    compute_backward: bool = True
