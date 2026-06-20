from __future__ import annotations


class EnsembleEvolution:
    def __init__(self, base_evolution, scenario_generator):
        self.base_evolution = base_evolution
        self.scenario_generator = scenario_generator

    def evolve(self, system, pulse, context):
        raise NotImplementedError("Exact ensemble evolution is reserved for a future module.")
