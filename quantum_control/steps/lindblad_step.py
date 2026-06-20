from __future__ import annotations


class LindbladStepBuilder:
    def build_step(self, system, controls, dt, t=None):
        raise NotImplementedError("Lindblad evolution is reserved for a future module.")

    def derivative_step(self, system, controls, dt, control_index, step, t=None):
        raise NotImplementedError("Lindblad gradients are reserved for a future module.")
