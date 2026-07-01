from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm, expm_frechet

from quantum_control.steps.base import StepBuilder


@dataclass(frozen=True)
class UnitaryStep:
    W: np.ndarray


class UnitaryStepBuilder(StepBuilder):
    def __init__(self, dW_method="first_order"):
        self.dW_method = dW_method

    def build_step(self, system, controls, dt, t=None):
        hamiltonian = system.nominal_hamiltonian(controls, t=t)
        return UnitaryStep(W=expm(-1j * dt * hamiltonian))

    def derivative_step(self, system, controls, dt, control_index, step, t=None):
        control_h = system.control_hamiltonian(control_index, controls=controls, t=t)
        if self.dW_method == "first_order":
            dW = -1j * dt * control_h @ step.W
        elif self.dW_method == "frechet":
            hamiltonian = system.nominal_hamiltonian(controls, t=t)
            dW = expm_frechet(
                -1j * dt * hamiltonian,
                -1j * dt * control_h,
                compute_expm=False,
            )
        else:
            raise ValueError("dW_method must be 'first_order' or 'frechet'.")
        return UnitaryStep(W=dW)
