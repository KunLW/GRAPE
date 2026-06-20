from __future__ import annotations


class ParameterizedControlProblem:
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
        pulse = pulse or self.problem.pulse
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

    def parameter_bounds(self):
        return self.parameterization.parameter_bounds()
