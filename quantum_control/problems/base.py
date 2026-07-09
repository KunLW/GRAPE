"""The two problem protocols of this package.

The package is split by coordinate system:

- **pulse space** (physics side): ``value(pulse)`` / ``gradient(pulse)`` on a
  physical ``PiecewiseConstantPulse`` — ``ControlProblem``,
  ``StateAverageProblem``, ``SumProblem``.
- **parameter space** (optimizer side): the same methods on the
  parameterization's normalized parameters, plus ``initial_parameters`` /
  ``parameter_bounds`` / ``pulse_from_parameters`` —
  ``ParameterizedControlProblem``, ``PenalizedParameterizedProblem``.

``ParameterizedControlProblem`` is the boundary that changes coordinates.
Optimizers consume ``ParameterProblem`` via ``optimize_parameters`` (or a
``PulseProblem`` directly via ``optimize``); when a problem provides
``value_and_gradient``, they fuse the two evaluations into one propagation.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class PulseProblem(Protocol):
    """Pulse-space problem: evaluated on a physical pulse."""

    pulse: object

    def value(self, pulse=None) -> float: ...

    def gradient(self, pulse=None) -> np.ndarray: ...

    def value_and_gradient(self, pulse=None) -> tuple[float, np.ndarray]: ...


@runtime_checkable
class ParameterProblem(Protocol):
    """Parameter-space problem: evaluated on optimizer parameters."""

    def initial_parameters(self, pulse=None) -> np.ndarray: ...

    def pulse_from_parameters(self, parameters) -> object: ...

    def parameter_bounds(self): ...

    def value(self, parameters) -> float: ...

    def gradient(self, parameters) -> np.ndarray: ...

    def value_and_gradient(self, parameters) -> tuple[float, np.ndarray]: ...
