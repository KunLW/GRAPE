from __future__ import annotations

from abc import ABC, abstractmethod


class ControlSystem(ABC):
    @abstractmethod
    def nominal_hamiltonian(self, controls, t=None):
        raise NotImplementedError

    @abstractmethod
    def control_hamiltonian(self, control_index, controls=None, t=None):
        raise NotImplementedError

    def fluctuation_hamiltonian(self, controls, t=None):
        return 0.0 * self.nominal_hamiltonian(controls, t=t)

    def fluctuation_control_derivative(self, control_index, controls=None, t=None):
        return 0.0 * self.control_hamiltonian(control_index, controls=controls, t=t)
