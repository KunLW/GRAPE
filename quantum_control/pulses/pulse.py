from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PiecewiseConstantPulse:
    amplitudes: np.ndarray
    dt: float

    def __post_init__(self):
        amplitudes = np.asarray(self.amplitudes, dtype=float)
        if amplitudes.ndim != 2:
            raise ValueError("amplitudes must have shape (n_steps, n_controls).")
        object.__setattr__(self, "amplitudes", amplitudes)

    @property
    def n_steps(self):
        return self.amplitudes.shape[0]

    @property
    def n_controls(self):
        return self.amplitudes.shape[1]

    def controls_at(self, step_index):
        return self.amplitudes[step_index]

    def with_amplitudes(self, amplitudes):
        return type(self)(amplitudes=np.asarray(amplitudes, dtype=float), dt=self.dt)
