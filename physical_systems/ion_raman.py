"""Single driven qubit: the minimal sample system definition.

This module is the smallest complete example of the two-file recipe from
``physical_systems/README.md``: one module here plus one YAML config
(``experiments/ion_raman/configs/example.yaml``). Drop a module like this
one into ``physical_systems/`` and the registry in
``physical_systems/__init__.py`` discovers it automatically — no driver or
engine edits, no registration call. Use it as the starting template for a new
system; ``spin_boson.py`` and ``two_ion_raman.py`` show the fully featured
versions (noise terms, custom pulses, presentation hooks).

Physical model
--------------


Units follow the repo convention: the YAML config speaks kHz (and µs), all
internal quantities are angular frequency in rad/s, converted through
``quantum_control.units`` — never a hand-rolled ``2 * pi * 1000``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import override

import numpy as np

from physical_systems.common import ControlChannel, SystemDefinitionBase
from quantum_control import RAD_S_PER_KHZ, khz_bounds_to_rad_s
from quantum_control.problems.state_average import StatePair
from quantum_control.systems import ClosedSystem

PAULI_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
PAULI_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)


def qubit_test_states():
    """The four tomography input states ``|0>``, ``|1>``, ``|+>``, ``|+i>``.

    Averaging a state-transfer fidelity over these four inputs measures the
    fidelity of the whole gate, not just of one transition.
    """
    zero = np.array([1.0, 0.0], dtype=complex)
    one = np.array([0.0, 1.0], dtype=complex)
    plus = (zero + one) / np.sqrt(2.0)
    plus_i = (zero + 1j * one) / np.sqrt(2.0)
    return (zero, one, plus, plus_i)


@dataclass(frozen=True)
class SingleQubitParams:
    """Physical parameters; fields define the ``system.params`` YAML schema.

    The config loader validates ``system.params`` keys against these fields
    generically: unknown keys raise, absent keys keep the defaults below.

    Attributes:
        detuning_khz: Constant drive detuning ``delta`` in kHz; enters the
            drift as ``0.5 * delta * sigma_z``.
        rabi_khz_bounds: ``(lower, upper)`` amplitude bounds of the Rabi
            control ``omega(t)`` in kHz (converted to rad/s internally).
    """

    detuning_khz: float = 1.0
    rabi_khz_bounds: tuple[float, float] = (0.0, 50.0)


class SingleQubitDefinition(SystemDefinitionBase):
    """System definition registered under ``system.type: single_qubit``.

    The registry instantiates this class with no arguments and the driver
    reaches every system-specific fact through its hooks; the generic
    assembly (``build_systems``, initial pulse, parameterization) comes from
    ``SystemDefinitionBase``.
    """

    name = "single_qubit"

    @override
    def default_params(self):
        """The params dataclass whose fields are the YAML schema."""
        return SingleQubitParams()

    @override
    def build_closed_system(self, params):
        """The noiseless system: drift plus one control Hamiltonian.

        ``ClosedSystem`` applies the controls linearly, so the nominal
        Hamiltonian is ``drift + omega(t) * controls[0]`` per time step.
        """
        drift = 0.5 * RAD_S_PER_KHZ * params.detuning_khz * PAULI_Z
        return ClosedSystem(drift=drift, controls=[0.5 * PAULI_X])

    @override
    def control_bounds(self, params):
        """Per-channel ``(lower, upper)`` amplitude bounds in rad/s."""
        lower, upper = khz_bounds_to_rad_s(params.rabi_khz_bounds)
        return np.array([lower]), np.array([upper])

    @override
    def target_gate(self, params):
        """The X gate (informational; printed in the run report)."""
        return PAULI_X

    @override
    def state_pairs(self, params):
        """The fidelity definition: X-gate fidelity averaged over four inputs.

        Each pair contributes ``weight * |<target|U|initial>|^2``; with weight
        ``1/4`` the weighted sum is the average gate fidelity in ``[0, 1]``.
        """
        return tuple(
            StatePair(state, PAULI_X @ state, 0.25) for state in qubit_test_states()
        )

    @override
    def control_channels(self, params):
        """Label the control channel in kHz for plots and CSV export."""
        return [
            ControlChannel(label="omega", display_scale=1.0 / RAD_S_PER_KHZ, display_unit="kHz")
        ]
