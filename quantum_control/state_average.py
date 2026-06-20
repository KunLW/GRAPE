from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from quantum_control.context import EvolutionContext


@dataclass(frozen=True)
class StatePair:
    initial_state: np.ndarray
    target_state: np.ndarray
    weight: float = 1.0


class ExpansionStateAverageFidelity:
    """Average a perturbative fidelity objective over multiple state pairs."""

    def __init__(
        self,
        system,
        pulse,
        evolution,
        objective,
        differentiator,
        state_pairs: Sequence[StatePair | tuple],
        compute_backward=True,
        normalize_weights=True,
    ):
        if not state_pairs:
            raise ValueError("state_pairs must contain at least one pair.")
        self.system = system
        self.pulse = pulse
        self.evolution = evolution
        self.objective = objective
        self.differentiator = differentiator
        self.state_pairs = tuple(self._coerce_pair(pair) for pair in state_pairs)
        self.compute_backward = compute_backward
        self.normalize_weights = normalize_weights
        self._weights = self._normalized_weights()

    def value(self, pulse=None):
        pulse = pulse or self.pulse
        value = 0.0
        for weight, pair in zip(self._weights, self.state_pairs):
            result = self.evolution.evolve(self.system, pulse, self._context(pair))
            value = value + weight * self.objective.evaluate(result)
        return float(value)

    def gradient(self, pulse=None):
        if self.differentiator is None:
            raise ValueError("A differentiator is required to compute gradients.")
        pulse = pulse or self.pulse
        gradient = np.zeros_like(pulse.amplitudes)
        for weight, pair in zip(self._weights, self.state_pairs):
            context = self._context(pair)
            result = self.evolution.evolve(self.system, pulse, context)
            gradient = gradient + weight * self.differentiator.gradient(
                self.system,
                pulse,
                context,
                result,
            )
        return gradient

    @staticmethod
    def _coerce_pair(pair):
        if isinstance(pair, StatePair):
            return pair
        if len(pair) == 2:
            initial_state, target_state = pair
            return StatePair(initial_state, target_state)
        if len(pair) == 3:
            initial_state, target_state, weight = pair
            return StatePair(initial_state, target_state, weight)
        raise ValueError("state pairs must be StatePair, (initial, target), or (initial, target, weight).")

    def _normalized_weights(self):
        weights = np.asarray([pair.weight for pair in self.state_pairs], dtype=float)
        if np.any(weights < 0.0):
            raise ValueError("state pair weights must be non-negative.")
        total = float(np.sum(weights))
        if total <= 0.0:
            raise ValueError("state pair weights must have positive total weight.")
        if self.normalize_weights:
            weights = weights / total
        return tuple(weights)

    def _context(self, pair):
        return EvolutionContext(
            initial_state=np.asarray(pair.initial_state, dtype=complex),
            target_state=np.asarray(pair.target_state, dtype=complex),
            compute_backward=self.compute_backward,
        )
