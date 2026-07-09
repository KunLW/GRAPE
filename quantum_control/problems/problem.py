from __future__ import annotations


class ControlProblem:
    """Single-state-pair pulse-space problem (the minimal assembly).

    ``StateAverageProblem`` generalizes this to weighted state-pair
    averages; this class remains as the simplest entry point (see the
    README quick example).
    """

    def __init__(self, system, pulse, context, evolution, objective, differentiator=None):
        self.system = system
        self.pulse = pulse
        self.context = context
        self.evolution = evolution
        self.objective = objective
        self.differentiator = differentiator

    def value(self, pulse=None):
        if pulse is None:
            pulse = self.pulse
        result = self.evolution.evolve(self.system, pulse, self.context)
        return self.objective.evaluate(result)

    def gradient(self, pulse=None):
        if self.differentiator is None:
            raise ValueError("A differentiator is required to compute gradients.")
        if pulse is None:
            pulse = self.pulse
        result = self.evolution.evolve(self.system, pulse, self.context)
        return self.differentiator.gradient(self.system, pulse, self.context, result)

    def value_and_gradient(self, pulse=None):
        """Both at once with a single evolution."""
        if self.differentiator is None:
            raise ValueError("A differentiator is required to compute gradients.")
        if pulse is None:
            pulse = self.pulse
        result = self.evolution.evolve(self.system, pulse, self.context)
        return (
            self.objective.evaluate(result),
            self.differentiator.gradient(self.system, pulse, self.context, result),
        )
