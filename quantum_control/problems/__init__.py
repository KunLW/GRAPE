"""Problem assembly: system + pulse + evolution + objective -> value/gradient.

Two coordinate systems, documented as protocols in ``problems/base.py``:
pulse-space problems (``ControlProblem``, ``StateAverageProblem``,
``SumProblem``) and parameter-space problems
(``ParameterizedControlProblem``, ``PenalizedParameterizedProblem``), with
the parameterization wrapper as the boundary between them. The driver stacks
them as: ``StateAverageProblem`` (+ ``SumProblem`` for the Lindblad
correction) -> ``ParameterizedControlProblem`` ->
``PenalizedParameterizedProblem`` -> optimizer.

Conventions:

- Problems are fidelities to be **maximized**; penalties (physical-space
  ``constraints`` on the parameterized problem, parameter-space smoothness on
  the penalized problem) are always **subtracted**. The optimizer's
  ``maximize`` flag only flips the sign handed to scipy.
- Problems providing ``value_and_gradient`` are evaluated with a single
  propagation per optimizer iteration.
"""

from quantum_control.problems.base import ParameterProblem, PulseProblem
from quantum_control.problems.context import EvolutionContext
from quantum_control.problems.parameterized_problem import ParameterizedControlProblem
from quantum_control.problems.penalties import (
    ParameterSmoothPenalty,
    PenalizedParameterizedProblem,
)
from quantum_control.problems.problem import ControlProblem
from quantum_control.problems.state_average import (
    StateAverageProblem,
    StatePair,
    SumProblem,
)

__all__ = [
    "ControlProblem",
    "EvolutionContext",
    "ParameterProblem",
    "ParameterSmoothPenalty",
    "ParameterizedControlProblem",
    "PenalizedParameterizedProblem",
    "PulseProblem",
    "StateAverageProblem",
    "StatePair",
    "SumProblem",
]
