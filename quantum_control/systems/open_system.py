from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from quantum_control.systems.closed_system import FluctuatingClosedSystem


@dataclass(frozen=True)
class LindbladOpenSystem(FluctuatingClosedSystem):
    r"""Closed control system with Markovian decoherence channels.

    ``collapse_operators`` are already-scaled jump operators
    $L_\mu = \sqrt{\gamma_\mu}\, A_\mu$ entering the Lindblad dissipator
    $$
    \mathcal{L}[\rho] = \sum_\mu \Big( L_\mu \rho L_\mu^\dagger
        - \tfrac{1}{2}\{L_\mu^\dagger L_\mu,\, \rho\} \Big).
    $$
    """

    collapse_operators: Sequence[np.ndarray] = ()
