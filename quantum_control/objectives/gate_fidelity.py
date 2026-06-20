from __future__ import annotations

import numpy as np

from quantum_control.objectives.base import Objective


class GateFidelity(Objective):
    def __init__(self, target_unitary):
        self.target_unitary = np.asarray(target_unitary, dtype=complex)

    def evaluate(self, result):
        dimension = self.target_unitary.shape[0]
        overlap = np.trace(self.target_unitary.conj().T @ result.U_total)
        return float(np.abs(overlap / dimension) ** 2)
