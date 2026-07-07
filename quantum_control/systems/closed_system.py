from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class ClosedSystem:
    """Nominal + control Hamiltonians only — the root of the system hierarchy.

    ``H(t) = drift + sum_i controls_amplitude_i(t) * controls[i]``. The
    zero-valued noise defaults below let a purely closed system run through
    every evolution path unchanged; systems with noise are represented by
    ``OpenSystem``, which combines a closed system with ``NoiseTerm``s.
    """

    drift: np.ndarray
    controls: Sequence[np.ndarray]

    def nominal_hamiltonian(self, controls, t=None):
        hamiltonian = np.array(self.drift, dtype=complex, copy=True)
        for amplitude, control_h in zip(controls, self.controls, strict=True):
            hamiltonian = hamiltonian + amplitude * control_h
        return hamiltonian

    def control_hamiltonian(self, control_index, controls=None, t=None):
        return self.controls[control_index]

    def fluctuation_hamiltonian(self, controls, t=None):
        return 0.0 * self.nominal_hamiltonian(controls, t=t)

    def fluctuation_control_derivative(self, control_index, controls=None, t=None):
        return 0.0 * self.control_hamiltonian(control_index, controls=controls, t=t)
