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


def _evaluate_chunk(args):
    """Weighted (value, gradient) over one chunk of state pairs.

    ``mode`` is ``"value"``, ``"gradient"``, or ``"both"``; the unrequested
    half is returned as ``None``. Each pair is evolved exactly once, so the
    ``"both"`` mode shares the evolution between the objective and the
    differentiator.
    """
    system, pulse, evolution, objective, differentiator, compute_backward, mode, weighted_pairs = args
    value = 0.0 if mode != "gradient" else None
    gradient = np.zeros_like(pulse.amplitudes) if mode != "value" else None
    for weight, pair in weighted_pairs:
        context = _context_for_pair(pair, compute_backward)
        result = evolution.evolve(system, pulse, context)
        if value is not None:
            value = value + weight * objective.evaluate(result)
        if gradient is not None:
            gradient = gradient + weight * differentiator.gradient(system, pulse, context, result)
    return value, gradient


class StateAverageProblem:
    """Weighted state-pair average of any evolution + objective.

    Generic over the evolution/objective pair: the fluctuation expansion, the
    Lindblad correction, and plain unitary propagation all run through it.
    Exposes the pulse-space problem interface (``value`` / ``gradient`` /
    ``value_and_gradient``); with ``n_workers > 1`` the pairs are averaged in
    a process pool, whose lifetime is tied to the ``with`` statement (or an
    explicit ``shutdown()``).
    """

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
        value, _ = self._evaluate(pulse, "value")
        return float(value)

    def gradient(self, pulse=None):
        _, gradient = self._evaluate(pulse, "gradient")
        return gradient

    def value_and_gradient(self, pulse=None):
        """Both at once with a single evolution per state pair."""
        value, gradient = self._evaluate(pulse, "both")
        return float(value), gradient

    def _evaluate(self, pulse, mode):
        if mode != "value" and self.differentiator is None:
            raise ValueError("A differentiator is required to compute gradients.")
        if pulse is None:
            pulse = self.pulse
        weighted_pairs = tuple(zip(self._weights, self.state_pairs))
        if self.n_workers > 1:
            args = [
                (
                    self.system,
                    pulse,
                    self.evolution,
                    self.objective,
                    self.differentiator,
                    self.compute_backward,
                    mode,
                    chunk,
                )
                for chunk in self._chunks(weighted_pairs)
            ]
            results = list(self._pool().map(_evaluate_chunk, args))
            value = float(np.sum([v for v, _ in results])) if mode != "gradient" else None
            gradient = np.sum([g for _, g in results], axis=0) if mode != "value" else None
            return value, gradient
        return _evaluate_chunk(
            (
                self.system,
                pulse,
                self.evolution,
                self.objective,
                self.differentiator,
                self.compute_backward,
                mode,
                weighted_pairs,
            )
        )

    def shutdown(self):
        if self._executor is not None:
            self._executor.shutdown()
            self._executor = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.shutdown()
        return False

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

    def _chunks(self, weighted_pairs):
        n_chunks = min(self.n_workers, len(weighted_pairs))
        chunks = np.array_split(np.asarray(weighted_pairs, dtype=object), n_chunks)
        return [tuple(chunk) for chunk in chunks if len(chunk)]

    def _pool(self):
        if self._executor is None:
            self._executor = ProcessPoolExecutor(max_workers=self.n_workers)
        return self._executor


class SumProblem:
    """Sum of pulse-space problems evaluated on the same pulse.

    Used to add leading-order noise corrections from different channels, e.g.
    the fluctuation expansion fidelity plus the Lindblad decoherence
    correction (with ``include_closed=False`` so the closed term is counted
    once). The children must be built on the same pulse grid; this is
    validated at construction.
    """

    def __init__(self, *problems):
        if not problems:
            raise ValueError("SumProblem requires at least one problem.")
        self.problems = tuple(problems)
        reference = self.problems[0].pulse
        for problem in self.problems[1:]:
            pulse = problem.pulse
            if pulse.amplitudes.shape != reference.amplitudes.shape or pulse.dt != reference.dt:
                raise ValueError("SumProblem children must share the same pulse grid.")

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

    def value_and_gradient(self, pulse=None):
        value, gradient = self.problems[0].value_and_gradient(pulse)
        for problem in self.problems[1:]:
            child_value, child_gradient = problem.value_and_gradient(pulse)
            value = value + child_value
            gradient = gradient + child_gradient
        return float(value), gradient

    def shutdown(self):
        for problem in self.problems:
            problem.shutdown()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.shutdown()
        return False
