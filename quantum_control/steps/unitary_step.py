from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm

from quantum_control.steps.base import StepBuilder


@dataclass(frozen=True)
class UnitaryStep:
    W: np.ndarray


class UnitaryStepBuilder(StepBuilder):
    def build_step(self, system, controls, dt, t=None):
        hamiltonian = system.nominal_hamiltonian(controls, t=t)
        return UnitaryStep(W=expm(-1j * dt * hamiltonian))

    def derivative_step(self, system, controls, dt, control_index, step, t=None):
        control_h = system.control_hamiltonian(control_index, controls=controls, t=t)
        return UnitaryStep(W=-1j * dt * control_h @ step.W)
