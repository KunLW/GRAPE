from __future__ import annotations

from abc import ABC, abstractmethod


class Evolution(ABC):
    @abstractmethod
    def evolve(self, system, pulse, context):
        raise NotImplementedError
