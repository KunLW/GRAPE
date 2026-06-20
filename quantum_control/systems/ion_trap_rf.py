from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from quantum_control.systems.closed_system import ClosedSystem


@dataclass(frozen=True)
class IonTrapRFSystem(ClosedSystem):
    """Closed ion-trap control system with long-correlation fluctuations.

    ``static_fluctuations`` stores the already-scaled terms ``sigma_xi H_xi``.
    ``control_fluctuations`` stores ``sigma_chi_i H_chi_i`` for each control
    channel, so the full fluctuation Hamiltonian is
    ``sum static + sum control_i * control_fluctuation_i``.
    """

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
