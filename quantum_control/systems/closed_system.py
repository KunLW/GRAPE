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


@dataclass(frozen=True)
class FluctuatingClosedSystem(ClosedSystem):
    """Closed control system with long-correlation Hamiltonian fluctuations."""

    static_fluctuations: Sequence[np.ndarray] = ()
    control_fluctuations: Sequence[np.ndarray] = ()

    def fluctuation_hamiltonian(self, controls, t=None):
        if len(self.control_fluctuations) > len(controls):
            raise ValueError("control_fluctuations cannot exceed the number of controls.")
        hamiltonian = np.zeros_like(self.drift, dtype=complex)
        for fluctuation_h in self.static_fluctuations:
            hamiltonian = hamiltonian + fluctuation_h
        for amplitude, fluctuation_h in zip(controls, self.control_fluctuations):
            hamiltonian = hamiltonian + amplitude * fluctuation_h
        return hamiltonian

    def fluctuation_control_derivative(self, control_index, controls=None, t=None):
        if control_index >= len(self.control_fluctuations):
            return np.zeros_like(self.drift, dtype=complex)
        return self.control_fluctuations[control_index]
