from __future__ import annotations

import numpy as np

from quantum_control.objectives.base import Objective


class StateTransferFidelity(Objective):
    def __init__(self, target_state):
        self.target_state = np.asarray(target_state, dtype=complex)

    def evaluate(self, result):
        final_state = result.U_total @ result.metadata["initial_state"]
        amplitude = np.vdot(self.target_state, final_state)
        return float(np.abs(amplitude) ** 2)
