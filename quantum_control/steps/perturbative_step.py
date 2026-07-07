from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm_frechet

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

    def __init__(
        self,
        dW_method="first_order",
        dV_method="include_dW",
        V_method="leading",
        v_derivative_epsilon=1e-7,
    ):
        self.dW_method = dW_method
        self.dV_method = dV_method
        self.V_method = V_method
        self.v_derivative_epsilon = float(v_derivative_epsilon)

    def build_step(self, system, controls, dt, t=None):
        unitary_step = super().build_step(system, controls, dt, t=t)
        fluctuation_h = _fluctuation_hamiltonian(system, controls, unitary_step.W, t=t)
        if self.V_method == "leading":
            V = -1j * dt * fluctuation_h @ unitary_step.W
        elif self.V_method == "frechet":
            hamiltonian = system.nominal_hamiltonian(controls, t=t)
            V = expm_frechet(
                -1j * dt * hamiltonian,
                -1j * dt * fluctuation_h,
                compute_expm=False,
            )
        else:
            raise ValueError("V_method must be 'leading' or 'frechet'.")
        return PerturbativeStep(
            W=unitary_step.W,
            V=V,
        )

    def derivative_step(self, system, controls, dt, control_index, step, t=None):
        dW = super().derivative_step(system, controls, dt, control_index, step, t=t).W
        if self.V_method == "frechet":
            dV = self._frechet_V_finite_difference(
                system,
                controls,
                dt,
                control_index,
                t=t,
            )
            return PerturbativeStep(W=dW, V=dV)

        dfluc_h = _fluctuation_control_derivative(
            system,
            control_index,
            step.W,
            controls=controls,
            t=t,
        )
        dV = -1j * dt * dfluc_h @ step.W
        if self.dV_method == "include_dW":
            fluctuation_h = _fluctuation_hamiltonian(system, controls, step.W, t=t)
            dV = dV + -1j * dt * fluctuation_h @ dW
        elif self.dV_method != "leading":
            raise ValueError("dV_method must be 'include_dW' or 'leading'.")
        return PerturbativeStep(W=dW, V=dV)

    def _frechet_V_finite_difference(self, system, controls, dt, control_index, t=None):
        epsilon = self.v_derivative_epsilon
        plus = np.asarray(controls, dtype=float).copy()
        minus = np.asarray(controls, dtype=float).copy()
        plus[control_index] += epsilon
        minus[control_index] -= epsilon
        plus_step = self.build_step(system, plus, dt, t=t)
        minus_step = self.build_step(system, minus, dt, t=t)
        return (plus_step.V - minus_step.V) / (2.0 * epsilon)


def _fluctuation_hamiltonian(system, controls, like, t=None):
    """H_fluctuation, or zero for systems without a noise model (ClosedSystem)."""
    method = getattr(system, "fluctuation_hamiltonian", None)
    if method is None:
        return np.zeros_like(like)
    return method(controls, t=t)


def _fluctuation_control_derivative(system, control_index, like, controls=None, t=None):
    """dH_fluctuation/d(control), or zero for systems without a noise model."""
    method = getattr(system, "fluctuation_control_derivative", None)
    if method is None:
        return np.zeros_like(like)
    return method(control_index, controls=controls, t=t)
