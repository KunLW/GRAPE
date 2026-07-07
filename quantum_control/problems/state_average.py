from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from quantum_control.problems.context import EvolutionContext


@dataclass(frozen=True)
class StatePair:
    initial_state: np.ndarray
    target_state: np.ndarray
    weight: float = 1.0


def _context_for_pair(pair, compute_backward):
    return EvolutionContext(
        initial_state=np.asarray(pair.initial_state, dtype=complex),
        target_state=np.asarray(pair.target_state, dtype=complex),
        compute_backward=compute_backward,
    )


def _weighted_value_chunk(args):
    system, pulse, evolution, objective, compute_backward, weighted_pairs = args
    value = 0.0
    for weight, pair in weighted_pairs:
        result = evolution.evolve(system, pulse, _context_for_pair(pair, compute_backward))
        value = value + weight * objective.evaluate(result)
    return value


def _weighted_gradient_chunk(args):
    system, pulse, evolution, differentiator, compute_backward, weighted_pairs = args
    gradient = np.zeros_like(pulse.amplitudes)
    for weight, pair in weighted_pairs:
        context = _context_for_pair(pair, compute_backward)
        result = evolution.evolve(system, pulse, context)
        gradient = gradient + weight * differentiator.gradient(system, pulse, context, result)
    return gradient


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
        n_workers=1,
    ):
        if not state_pairs:
            raise ValueError("state_pairs must contain at least one pair.")
        if n_workers < 1:
            raise ValueError("n_workers must be at least 1.")
        self.system = system
        self.pulse = pulse
        self.evolution = evolution
        self.objective = objective
        self.differentiator = differentiator
        self.state_pairs = tuple(self._coerce_pair(pair) for pair in state_pairs)
        self.compute_backward = compute_backward
        self.normalize_weights = normalize_weights
        self.n_workers = int(n_workers)
        self._executor = None
        self._weights = self._normalized_weights()

    def value(self, pulse=None):
        pulse = pulse or self.pulse
        if self.n_workers > 1:
            args = [
                (
                    self.system,
                    pulse,
                    self.evolution,
                    self.objective,
                    self.compute_backward,
                    chunk,
                )
                for chunk in self._weighted_chunks()
            ]
            return float(np.sum(list(self._pool().map(_weighted_value_chunk, args))))

        value = 0.0
        for weight, pair in zip(self._weights, self.state_pairs):
            result = self.evolution.evolve(self.system, pulse, self._context(pair))
            value = value + weight * self.objective.evaluate(result)
        return float(value)

    def gradient(self, pulse=None):
        if self.differentiator is None:
            raise ValueError("A differentiator is required to compute gradients.")
        pulse = pulse or self.pulse
        if self.n_workers > 1:
            args = [
                (
                    self.system,
                    pulse,
                    self.evolution,
                    self.differentiator,
                    self.compute_backward,
                    chunk,
                )
                for chunk in self._weighted_chunks()
            ]
            return np.sum(list(self._pool().map(_weighted_gradient_chunk, args)), axis=0)

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

    def shutdown(self):
        if self._executor is not None:
            self._executor.shutdown()
            self._executor = None

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
        return _context_for_pair(pair, self.compute_backward)

    def _weighted_chunks(self):
        weighted_pairs = tuple(zip(self._weights, self.state_pairs))
        n_chunks = min(self.n_workers, len(weighted_pairs))
        chunks = np.array_split(np.asarray(weighted_pairs, dtype=object), n_chunks)
        return [tuple(chunk) for chunk in chunks if len(chunk)]

    def _pool(self):
        if self._executor is None:
            self._executor = ProcessPoolExecutor(max_workers=self.n_workers)
        return self._executor


class CombinedStateAverageProblem:
    """Sum of state-average problems evaluated on the same pulse.

    Used to add leading-order noise corrections from different channels, e.g.
    the fluctuation expansion fidelity plus the Lindblad decoherence
    correction (with ``include_closed=False`` so the closed term is counted
    once).
    """

    def __init__(self, *problems):
        if not problems:
            raise ValueError("CombinedStateAverageProblem requires at least one problem.")
        self.problems = tuple(problems)

    @property
    def pulse(self):
        return self.problems[0].pulse

    def value(self, pulse=None):
        return float(sum(problem.value(pulse) for problem in self.problems))

    def gradient(self, pulse=None):
        gradient = self.problems[0].gradient(pulse)
        for problem in self.problems[1:]:
            gradient = gradient + problem.gradient(pulse)
        return gradient

    def shutdown(self):
        for problem in self.problems:
            problem.shutdown()
