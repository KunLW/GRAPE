"""Problem assembly: system + pulse + evolution + objective -> value/gradient.

The layering mirrors how the driver builds a run: a state-average problem
(``ExpansionStateAverageFidelity``, optionally summed with the Lindblad
correction via ``CombinedStateAverageProblem``) is wrapped by a pulse
parameterization (``ParameterizedControlProblem``) and smoothness penalties
(``PenalizedParameterizedProblem``) before being handed to an optimizer.
"""

from quantum_control.problems.context import EvolutionContext
from quantum_control.problems.parameterized_problem import ParameterizedControlProblem
from quantum_control.problems.penalties import (
    ParameterSmoothPenalty,
    PenalizedParameterizedProblem,
)
from quantum_control.problems.problem import ControlProblem
from quantum_control.problems.state_average import (
    CombinedStateAverageProblem,
    ExpansionStateAverageFidelity,
    StatePair,
)

__all__ = [
    "CombinedStateAverageProblem",
    "ControlProblem",
    "EvolutionContext",
    "ExpansionStateAverageFidelity",
    "ParameterSmoothPenalty",
    "ParameterizedControlProblem",
    "PenalizedParameterizedProblem",
    "StatePair",
]
