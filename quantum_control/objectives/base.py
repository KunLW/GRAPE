from __future__ import annotations

from abc import ABC, abstractmethod


class Objective(ABC):
    @abstractmethod
    def evaluate(self, result):
        raise NotImplementedError
