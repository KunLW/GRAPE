from __future__ import annotations

import numpy as np


class ScipyOptimizer:
    def __init__(self, method="L-BFGS-B", maximize=True, options=None):
        self.method = method
        self.maximize = maximize
        self.options = options or {}

    def optimize(self, problem, initial_pulse=None, bounds=None, constraints=(), callback=None):
        """Optimize a pulse-space problem (``value(pulse)`` interface).

        When the problem provides ``value_and_gradient``, scipy is handed a
        single fused callable (``jac=True``) so each iterate propagates only
        once; otherwise value and gradient are evaluated separately.
        """
        try:
            from scipy.optimize import minimize
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "ScipyOptimizer requires scipy. Install the 'scipy' package or use AdamOptimizer."
            ) from exc

        if initial_pulse is None:
            initial_pulse = problem.pulse
        pulse = initial_pulse
        shape = pulse.amplitudes.shape
        sign = -1.0 if self.maximize else 1.0

        if hasattr(problem, "value_and_gradient"):

            def objective(flat_amplitudes):
                trial_pulse = pulse.with_amplitudes(flat_amplitudes.reshape(shape))
                value, gradient = problem.value_and_gradient(trial_pulse)
                return sign * value, sign * gradient.reshape(-1)

            jacobian = True
        else:

            def objective(flat_amplitudes):
                trial_pulse = pulse.with_amplitudes(flat_amplitudes.reshape(shape))
                return sign * problem.value(trial_pulse)

            def jacobian(flat_amplitudes):
                trial_pulse = pulse.with_amplitudes(flat_amplitudes.reshape(shape))
                return sign * problem.gradient(trial_pulse).reshape(-1)

        result = minimize(
            objective,
            pulse.amplitudes.reshape(-1),
            jac=jacobian,
            method=self.method,
            bounds=bounds,
            constraints=constraints,
            callback=callback,
            options=self.options,
        )
        result.optimized_pulse = pulse.with_amplitudes(np.reshape(result.x, shape))
        return result

    def optimize_parameters(
        self,
        problem,
        initial_parameters=None,
        bounds=None,
        constraints=(),
        callback=None,
    ):
        """Optimize a parameter-space problem (``value(parameters)`` interface).

        Uses the fused ``value_and_gradient`` (``jac=True``) when available,
        halving the propagation cost per iterate.
        """
        try:
            from scipy.optimize import minimize
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "ScipyOptimizer requires scipy. Install the 'scipy' package or use AdamOptimizer."
            ) from exc

        parameters = (
            problem.initial_parameters().reshape(-1)
            if initial_parameters is None
            else np.asarray(initial_parameters, dtype=float).reshape(-1)
        )
        sign = -1.0 if self.maximize else 1.0
        bounds = bounds if bounds is not None else problem.parameter_bounds()

        if hasattr(problem, "value_and_gradient"):

            def objective(flat_parameters):
                value, gradient = problem.value_and_gradient(flat_parameters)
                return sign * value, sign * gradient.reshape(-1)

            jacobian = True
        else:

            def objective(flat_parameters):
                return sign * problem.value(flat_parameters)

            def jacobian(flat_parameters):
                return sign * problem.gradient(flat_parameters).reshape(-1)

        result = minimize(
            objective,
            parameters,
            jac=jacobian,
            method=self.method,
            bounds=bounds,
            constraints=constraints,
            callback=callback,
            options=self.options,
            tol=1e-15
        )
        result.optimized_pulse = problem.pulse_from_parameters(result.x)
        return result
