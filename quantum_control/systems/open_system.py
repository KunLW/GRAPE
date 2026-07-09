from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from quantum_control.systems.closed_system import ClosedSystem
from quantum_control.systems.noise import DecoherenceChannel, FluctuationTerm, NoiseTerm


@dataclass(frozen=True)
class OpenSystem(ClosedSystem):
    """A closed system combined with any number of declarative ``NoiseTerm``s.

    ``noise_terms`` may mix every noise type — ``FluctuationTerm``s feed the
    quasi-static fluctuation Hamiltonian, ``DecoherenceChannel``s feed the
    Lindblad correction, and future non-Markovian subtypes would slot in the
    same way. Choosing what noise acts on the system is choosing this list;
    an empty list is exactly the closed system.
    """

    noise_terms: Sequence[NoiseTerm] = ()

    def __post_init__(self):
        object.__setattr__(self, "noise_terms", tuple(self.noise_terms))
        for term in self.noise_terms:
            if not isinstance(term, NoiseTerm):
                raise TypeError(
                    f"noise_terms must contain NoiseTerm instances, got {type(term).__name__}."
                )
        if len(self.control_fluctuations) > len(self.controls):
            raise ValueError("control fluctuation terms cannot exceed the number of controls.")

    # ---- derived views consumed by the evolutions ---------------------------

    @property
    def fluctuation_terms(self):
        return tuple(term for term in self.noise_terms if isinstance(term, FluctuationTerm))

    @property
    def decoherence_channels(self):
        return tuple(term for term in self.noise_terms if isinstance(term, DecoherenceChannel))

    @property
    def static_fluctuations(self):
        return tuple(term.matrix for term in self.fluctuation_terms if term.kind == "static")

    @property
    def control_fluctuations(self):
        """Scaled control-kind matrices, aligned with control channels positionally."""
        return tuple(term.matrix for term in self.fluctuation_terms if term.kind == "control")

    @property
    def collapse_operators(self):
        """Scaled jump operators ``L = sqrt(gamma) * A`` of the active channels."""
        return tuple(channel.matrix for channel in self.decoherence_channels)

    def fluctuation_hamiltonian(self, controls, t=None):
        hamiltonian = np.zeros_like(self.drift, dtype=complex)
        for fluctuation_h in self.static_fluctuations:
            hamiltonian = hamiltonian + fluctuation_h
        for amplitude, fluctuation_h in zip(controls, self.control_fluctuations):
            hamiltonian = hamiltonian + amplitude * fluctuation_h
        return hamiltonian

    def fluctuation_control_derivative(self, control_index, controls=None, t=None):
        control_fluctuations = self.control_fluctuations
        if control_index >= len(control_fluctuations):
            return np.zeros_like(self.drift, dtype=complex)
        return control_fluctuations[control_index]
