from __future__ import annotations

from dataclasses import dataclass

from quantum_control.systems.closed_system import FluctuatingClosedSystem


@dataclass(frozen=True)
class IonTrapRFSystem(FluctuatingClosedSystem):
    """Closed ion-trap control system with long-correlation fluctuations.

    ``static_fluctuations`` stores the already-scaled terms ``sigma_xi H_xi``.
    ``control_fluctuations`` stores ``sigma_chi_i H_chi_i`` for each control
    channel, so the full fluctuation Hamiltonian is
    ``sum static + sum control_i * control_fluctuation_i``.
    """
