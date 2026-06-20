from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from quantum_control.systems.base import ControlSystem


@dataclass(frozen=True)
class ClosedSystem(ControlSystem):
    drift: np.ndarray
    controls: Sequence[np.ndarray]

    def nominal_hamiltonian(self, controls, t=None):
        hamiltonian = np.array(self.drift, dtype=complex, copy=True)
        for amplitude, control_h in zip(controls, self.controls, strict=True):
            hamiltonian = hamiltonian + amplitude * control_h
        return hamiltonian

    def control_hamiltonian(self, control_index, controls=None, t=None):
        return self.controls[control_index]
