from __future__ import annotations


class CompositeObjective:
    def __init__(self, weighted_objectives):
        self.weighted_objectives = list(weighted_objectives)

    def evaluate(self, result):
        return sum(weight * objective.evaluate(result) for weight, objective in self.weighted_objectives)
