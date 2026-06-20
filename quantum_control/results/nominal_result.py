from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class NominalUnitaryResult:
    W_steps: list[np.ndarray]
    U_total: np.ndarray
    metadata: dict = field(default_factory=dict)
