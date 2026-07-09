"""Declarative noise vocabulary shared by every physical system.

``NoiseTerm`` is the umbrella base for *all* noise; the quasi-static coherent
kind is a ``FluctuationTerm`` and the Markovian kind a ``DecoherenceChannel``
(a future non-Markovian model would be another subtype). An ``OpenSystem`` is
built by combining a ``ClosedSystem`` with a chosen list of these terms, so
selecting what noise acts on a system is selecting the term list.

Each term is declared *unscaled* — a physical operator plus its strength —
and exposes the scaled ``matrix`` the propagators consume. The scaling
conventions differ by type: a fluctuation strength multiplies linearly
(``sigma * A``) while a decoherence rate enters through the jump operator
(``L = sqrt(gamma) * A``).
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True)
class NoiseTerm:
    """Base of all declarative noise terms.

    ``name`` labels the term in reports and logs; ``definition`` is the
    human-readable operator description carried into the optimization report.
    """

    name: str
    operator: np.ndarray
    definition: str

    def __post_init__(self):
        object.__setattr__(self, "operator", np.asarray(self.operator, dtype=complex))

    @property
    def matrix(self):
        """The scaled matrix handed to the propagators (subtype-specific)."""
        raise NotImplementedError


@dataclass(frozen=True)
class FluctuationTerm(NoiseTerm):
    """One quasi-static coherent fluctuation term of the noisy Hamiltonian.

    ``kind`` is ``"static"`` (added to H as-is) or ``"control"`` (scaled by
    the instantaneous control amplitude at propagation time, which makes the
    coefficient a *relative* error; control terms align with control
    channels positionally). ``coefficient`` is the standard deviation sigma;
    ``usage`` is a human-readable string carried into the report.
    """

    coefficient: float
    kind: str = "static"
    usage: str = ""

    def __post_init__(self):
        super().__post_init__()
        if self.kind not in ("static", "control"):
            raise ValueError(f"fluctuation kind must be 'static' or 'control', got {self.kind!r}.")
        object.__setattr__(self, "coefficient", float(self.coefficient))

    @property
    def matrix(self):
        """The already-scaled ``sigma * operator`` entering H_fluctuation."""
        return self.coefficient * self.operator


@dataclass(frozen=True)
class DecoherenceChannel(NoiseTerm):
    """One Lindblad channel; ``rate`` is gamma in 1/s (must be non-negative)."""

    rate: float

    def __post_init__(self):
        super().__post_init__()
        rate = float(self.rate)
        if rate < 0.0:
            raise ValueError("decoherence rates must be non-negative.")
        object.__setattr__(self, "rate", rate)

    @property
    def matrix(self):
        """The scaled jump operator ``L = sqrt(rate) * operator``."""
        return np.sqrt(self.rate) * self.operator
