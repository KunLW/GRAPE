from __future__ import annotations

import numpy as np

from quantum_control.objectives.base import Objective


class ExpansionFidelity(Objective):
    def __init__(self, max_order=2, drop_odd_average=True):
        self.max_order = max_order
        self.drop_odd_average = drop_odd_average

    def evaluate(self, result):
        amplitudes = self.amplitudes(result)
        value = self.contract(amplitudes)
        return float(np.real_if_close(value).real)

    def amplitudes(self, result):
        target_state = result.backward[-1].components[0] if result.backward else None
        if target_state is None:
            target_state = result.metadata.get("target_state")
        if target_state is None:
            raise ValueError("Expansion fidelity requires a target state or backward states.")
        final_components = result.forward[-1].components
        return {
            order: np.vdot(target_state, final_components[order])
            for order in range(min(self.max_order, result.max_order) + 1)
        }

    def contract(self, amplitudes, derivative_amplitudes=None):
        value = 0.0 + 0.0j
        orders = range(self.max_order + 1)
        for left_order in orders:
            for right_order in orders:
                total_order = left_order + right_order
                if total_order > self.max_order:
                    continue
                if self.drop_odd_average and total_order % 2 == 1:
                    continue
                left = amplitudes.get(left_order, 0.0)
                right = amplitudes.get(right_order, 0.0)
                if derivative_amplitudes is None:
                    value = value + np.conj(left) * right
                else:
                    dleft = derivative_amplitudes.get(left_order, 0.0)
                    dright = derivative_amplitudes.get(right_order, 0.0)
                    value = value + np.conj(dleft) * right + np.conj(left) * dright
        return value


class SecondOrderFluctuationFidelity(ExpansionFidelity):
    def __init__(self, drop_odd_average=True):
        super().__init__(max_order=2, drop_odd_average=drop_odd_average)
