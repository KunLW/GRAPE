from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quantum_control.steps.unitary_step import UnitaryStepBuilder


@dataclass(frozen=True)
class PerturbativeStep:
    W: np.ndarray
    V: np.ndarray


class PerturbativeStepBuilder(UnitaryStepBuilder):
    """Build the long-correlation fluctuation expansion steps from the notes.

    The nominal propagator is ``W = exp(-i dt H_nominal)`` and the leading
    fluctuation insertion is ``V = -i dt H_fluctuation W``. By default the
    control derivative of ``V`` includes both the explicit fluctuation
    derivative and the derivative of the nominal propagator:
    ``dV = -i dt (dH_fluctuation W + H_fluctuation dW)``.
    """

    def __init__(self, dW_method="first_order", dV_method="include_dW"):
        self.dW_method = dW_method
        self.dV_method = dV_method

    def build_step(self, system, controls, dt, t=None):
        unitary_step = super().build_step(system, controls, dt, t=t)
        fluctuation_h = system.fluctuation_hamiltonian(controls, t=t)
        return PerturbativeStep(
            W=unitary_step.W,
            V=-1j * dt * fluctuation_h @ unitary_step.W,
        )

    def derivative_step(self, system, controls, dt, control_index, step, t=None):
        dW = super().derivative_step(system, controls, dt, control_index, step, t=t).W
        dfluc_h = system.fluctuation_control_derivative(
            control_index,
            controls=controls,
            t=t,
        )
        dV = -1j * dt * dfluc_h @ step.W
        if self.dV_method == "include_dW":
            fluctuation_h = system.fluctuation_hamiltonian(controls, t=t)
            dV = dV + -1j * dt * fluctuation_h @ dW
        elif self.dV_method != "leading":
            raise ValueError("dV_method must be 'include_dW' or 'leading'.")
        return PerturbativeStep(W=dW, V=dV)
