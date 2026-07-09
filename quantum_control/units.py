"""Unit conversions shared across systems and experiment drivers."""

from __future__ import annotations

import numpy as np

RAD_S_PER_KHZ = 2.0 * np.pi * 1000.0


def khz_bounds_to_rad_s(bounds):
    """Convert a ``(lower, upper)`` kHz bound pair to rad/s, validating order."""
    lower, upper = np.asarray(bounds, dtype=float)
    if upper <= lower:
        raise ValueError("upper bounds must be greater than lower bounds.")
    return RAD_S_PER_KHZ * lower, RAD_S_PER_KHZ * upper
