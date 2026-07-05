from __future__ import annotations

import numpy as np

from quantum_control.objectives.base import Objective


class LindbladCorrectedStateFidelity(Objective):
    r"""State-transfer fidelity with the first-order decoherence correction.

    $$
    \mathcal{F} = |\langle\psi_T|U|\psi_0\rangle|^2
                + \tau\,\mathrm{Re}\,\langle\psi_T|F_N^{(1)}\rangle
    $$
    where the order-1 forward component $F_N^{(1)}$ carries one insertion of
    the frozen-coefficient operator $x_k$ per slice (doc/lindblad_note.md) and
    $\tau$ is the time step ``dt``. With ``include_closed=False`` only the
    correction $\tau\,\mathrm{Re}\,\langle\psi_T|F_N^{(1)}\rangle$ is
    returned, so the term can be combined with another objective that already
    contains the closed fidelity.
    """

    def __init__(self, include_closed=True):
        self.include_closed = include_closed

    def evaluate(self, result):
        target_state = result.backward[-1].components[0] if result.backward else None
        if target_state is None:
            target_state = result.metadata.get("target_state")
        if target_state is None:
            raise ValueError(
                "Lindblad corrected fidelity requires a target state or backward states."
            )
        final_components = result.forward[-1].components
        correction_amplitude = np.vdot(target_state, final_components[1])
        value = result.metadata["dt"] * float(np.real(correction_amplitude))
        if self.include_closed:
            closed_amplitude = np.vdot(target_state, final_components[0])
            value = value + float(np.abs(closed_amplitude) ** 2)
        return float(value)
