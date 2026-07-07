from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BoundedAmplitudeParameterization:
    lower: np.ndarray | float
    upper: np.ndarray | float

    def to_physical(self, normalized):
        normalized = np.asarray(normalized, dtype=float)
        lower, upper = self.bounds_for(normalized.shape)
        return self.center(lower, upper) + self.scale(lower, upper) * normalized

    def to_parameters(self, amplitudes):
        amplitudes = np.asarray(amplitudes, dtype=float)
        lower, upper = self.bounds_for(amplitudes.shape)
        return (amplitudes - self.center(lower, upper)) / self.scale(lower, upper)

    def pullback_gradient(self, physical_gradient):
        physical_gradient = np.asarray(physical_gradient, dtype=float)
        lower, upper = self.bounds_for(physical_gradient.shape)
        return physical_gradient * self.scale(lower, upper)

    def parameter_bounds(self, shape):
        size = int(np.prod(shape))
        return [(-1.0, 1.0)] * size

    def bounds_for(self, shape):
        """Broadcast the physical ``(lower, upper)`` bounds to ``shape``."""
        lower = np.broadcast_to(np.asarray(self.lower, dtype=float), shape)
        upper = np.broadcast_to(np.asarray(self.upper, dtype=float), shape)
        if np.any(upper <= lower):
            raise ValueError("upper bounds must be greater than lower bounds.")
        return lower, upper

    # Deprecated alias: external wrappers and legacy experiments/ scripts
    # still call the pre-publication underscore name.
    _bounds_for = bounds_for

    @staticmethod
    def center(lower, upper):
        return 0.5 * (upper + lower)

    @staticmethod
    def scale(lower, upper):
        return 0.5 * (upper - lower)


@dataclass(frozen=True)
class MaskedPulseParameterization:
    base: BoundedAmplitudeParameterization
    free_mask: np.ndarray
    fixed_values: np.ndarray

    def __post_init__(self):
        free_mask = np.asarray(self.free_mask, dtype=bool)
        fixed_values = np.asarray(self.fixed_values, dtype=float)
        if free_mask.ndim != 2:
            raise ValueError("free_mask must have shape (n_steps, n_controls).")
        if fixed_values.shape != free_mask.shape:
            raise ValueError("fixed_values must have the same shape as free_mask.")
        object.__setattr__(self, "free_mask", free_mask)
        object.__setattr__(self, "fixed_values", fixed_values)

    @property
    def pulse_shape(self):
        return self.free_mask.shape

    @property
    def parameter_shape(self):
        return (int(np.count_nonzero(self.free_mask)),)

    def to_physical(self, parameters):
        parameters = np.asarray(parameters, dtype=float).reshape(self.parameter_shape)
        amplitudes = np.array(self.fixed_values, dtype=float, copy=True)
        lower, upper = self._free_bounds()
        amplitudes[self.free_mask] = self.base.center(lower, upper) + self.base.scale(
            lower,
            upper,
        ) * parameters
        return amplitudes

    def to_parameters(self, amplitudes):
        amplitudes = np.asarray(amplitudes, dtype=float)
        if amplitudes.shape != self.pulse_shape:
            raise ValueError(f"amplitudes must have shape {self.pulse_shape}.")
        lower, upper = self._free_bounds()
        return (
            (amplitudes[self.free_mask] - self.base.center(lower, upper))
            / self.base.scale(lower, upper)
        ).reshape(self.parameter_shape)

    def pullback_gradient(self, physical_gradient):
        physical_gradient = np.asarray(physical_gradient, dtype=float)
        if physical_gradient.shape != self.pulse_shape:
            raise ValueError(f"physical_gradient must have shape {self.pulse_shape}.")
        lower, upper = self._free_bounds()
        return (physical_gradient[self.free_mask] * self.base.scale(lower, upper)).reshape(
            self.parameter_shape
        )

    def parameter_bounds(self, shape=None):
        """Flat parameter bounds; ``shape`` is accepted for interface
        uniformity and ignored (the mask fixes the parameter shape)."""
        return self.base.parameter_bounds(self.parameter_shape)

    def _free_bounds(self):
        lower, upper = self.base.bounds_for(self.pulse_shape)
        return lower[self.free_mask], upper[self.free_mask]


def endpoint_masked_parameterization(
    n_steps,
    n_controls,
    lower,
    upper,
    initial_value=0.0,
    final_value=0.0,
):
    free_mask = np.ones((n_steps, n_controls), dtype=bool)
    fixed_values = np.zeros((n_steps, n_controls), dtype=float)
    free_mask[0, :] = False
    free_mask[-1, :] = False
    fixed_values[0, :] = np.broadcast_to(initial_value, (n_controls,))
    fixed_values[-1, :] = np.broadcast_to(final_value, (n_controls,))
    return MaskedPulseParameterization(
        base=BoundedAmplitudeParameterization(lower=lower, upper=upper),
        free_mask=free_mask,
        fixed_values=fixed_values,
    )
