from __future__ import annotations

from abc import ABC, abstractmethod


class Differentiator(ABC):
    @abstractmethod
    def gradient(self, system, pulse, context, result):
        raise NotImplementedError
