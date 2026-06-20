from __future__ import annotations

from abc import ABC, abstractmethod


class StepBuilder(ABC):
    @abstractmethod
    def build_step(self, system, controls, dt, t=None):
        raise NotImplementedError

    @abstractmethod
    def derivative_step(self, system, controls, dt, control_index, step, t=None):
        raise NotImplementedError
