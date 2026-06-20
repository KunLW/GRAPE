from __future__ import annotations


class ControlProblem:
    def __init__(self, system, pulse, context, evolution, objective, differentiator=None):
        self.system = system
        self.pulse = pulse
        self.context = context
        self.evolution = evolution
        self.objective = objective
        self.differentiator = differentiator

    def value(self, pulse=None):
        pulse = pulse or self.pulse
        result = self.evolution.evolve(self.system, pulse, self.context)
        return self.objective.evaluate(result)

    def gradient(self, pulse=None):
        if self.differentiator is None:
            raise ValueError("A differentiator is required to compute gradients.")
        pulse = pulse or self.pulse
        result = self.evolution.evolve(self.system, pulse, self.context)
        return self.differentiator.gradient(self.system, pulse, self.context, result)
