from __future__ import annotations


class ParameterizedControlProblem:
    """Coordinate change: wraps a pulse-space problem as a parameter-space one.

    ``parameterization`` maps bounded physical amplitudes to the optimizer's
    normalized parameters (and pulls gradients back). The optional
    ``constraints`` + ``penalty_weight`` apply a penalty in *physical
    amplitude* space (e.g. a hardware slew-rate limit via
    ``PulseConstraints``) — distinct from the *parameter-space* smoothness
    penalties layered on top by ``PenalizedParameterizedProblem``.
    """

    def __init__(
        self,
        problem,
        parameterization,
        constraints=None,
        penalty_weight=0.0,
    ):
        self.problem = problem
        self.parameterization = parameterization
        self.constraints = constraints
        self.penalty_weight = penalty_weight

    def initial_parameters(self, pulse=None):
        if pulse is None:
            pulse = self.problem.pulse
        return self.parameterization.to_parameters(pulse.amplitudes)

    def pulse_from_parameters(self, parameters):
        amplitudes = self.parameterization.to_physical(parameters)
        return self.problem.pulse.with_amplitudes(amplitudes)

    def value(self, parameters):
        pulse = self.pulse_from_parameters(parameters)
        value = self.problem.value(pulse)
        if self.constraints is not None and self.penalty_weight:
            value = value - self.constraints.penalty(pulse.amplitudes, self.penalty_weight)
        return value

    def gradient(self, parameters):
        pulse = self.pulse_from_parameters(parameters)
        physical_gradient = self.problem.gradient(pulse)
        if self.constraints is not None and self.penalty_weight:
            physical_gradient = physical_gradient - self.constraints.penalty_gradient(
                pulse.amplitudes,
                self.penalty_weight,
            )
        return self.parameterization.pullback_gradient(physical_gradient)

    def value_and_gradient(self, parameters):
        """Both at once; the wrapped problem evolves only once."""
        pulse = self.pulse_from_parameters(parameters)
        value, physical_gradient = self.problem.value_and_gradient(pulse)
        if self.constraints is not None and self.penalty_weight:
            value = value - self.constraints.penalty(pulse.amplitudes, self.penalty_weight)
            physical_gradient = physical_gradient - self.constraints.penalty_gradient(
                pulse.amplitudes,
                self.penalty_weight,
            )
        return value, self.parameterization.pullback_gradient(physical_gradient)

    def parameter_bounds(self):
        return self.parameterization.parameter_bounds(self.initial_parameters().shape)
