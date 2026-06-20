from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from quantum_control.systems.closed_system import ClosedSystem


@dataclass(frozen=True)
class IonTrapRFSystem(ClosedSystem):
    static_fluctuations: Sequence[np.ndarray] = ()
    control_fluctuations: Sequence[np.ndarray] = ()

    def fluctuation_hamiltonian(self, controls, t=None):
        hamiltonian = np.zeros_like(self.drift, dtype=complex)
        for fluctuation_h in self.static_fluctuations:
            hamiltonian = hamiltonian + fluctuation_h
        for amplitude, fluctuation_h in zip(controls, self.control_fluctuations, strict=True):
            hamiltonian = hamiltonian + amplitude * fluctuation_h
        return hamiltonian

    def fluctuation_control_derivative(self, control_index, controls=None, t=None):
        if control_index >= len(self.control_fluctuations):
            return np.zeros_like(self.drift, dtype=complex)
        return self.control_fluctuations[control_index]
