from __future__ import annotations

import numpy as np

from quantum_control.steps.unitary_step import UnitaryStep, UnitaryStepBuilder


class LindbladExpansionDifferentiator:
    r"""Analytic gradient of the Lindblad-corrected state-transfer fidelity.

    Implements the frozen-coefficient contraction (doc/lindblad_note.md):
    $$
    \frac{\partial \mathcal{F}}{\partial c_i(k)}
    = \underbrace{2\,\mathrm{Re}\big\{ s^*\,
        \langle B_k|\partial W_k|F_{k-1}\rangle \big\}}_{\text{closed}}
    + \underbrace{2\tau\,\mathrm{Re}\big\{
        \langle B_k|\partial W_k|F^{(1)}_{k-1}\rangle
        + \langle B_k|x_k\,\partial W_k|F_{k-1}\rangle
        + \langle B^{(1)}_k|\partial W_k|F_{k-1}\rangle
      \big\}}_{\text{correction}}
    $$
    with $\tau$ the time step ``pulse.dt``. The scalars inside $x_k$ are
    frozen; the $2\,\mathrm{Re}\{\cdots\}$ supplies the missing conjugate
    halves exactly, so no $\partial x/\partial c$ terms appear.
    """

    def __init__(self, step_builder=None, include_closed=True):
        self.step_builder = step_builder or UnitaryStepBuilder()
        self.include_closed = include_closed

    def gradient(self, system, pulse, context, result):
        r"""Contract $\partial\mathcal{F}/\partial c_i(k)$ for every slice and control.

        ``result`` must come from ``LindbladExpansionEvolution``: it carries the
        order-$\{0,1\}$ forward/backward chains and the frozen insertion
        operators $x_k$ in ``metadata``. Index convention (0-based
        ``step_index`` maps to math slice $k = \mathtt{step\_index} + 1$):

        - ``forward[step_index]`` $\to F_{k-1}$, components {0: plain, 1: single insertion}
        - ``backward[step_index + 1]`` $\to B_k$, components {0: plain, 1: single insertion}
        - ``insertion_operators[step_index + 1]`` $\to x_k$ (list is indexed by
          time boundary $0 \ldots N$; the boundary-0 entry only enters through
          the $F^{(1)}$ seed, not through this loop)

        Per slice, the three correction contractions pick up the insertion at
        $j < k$ (via $F^{(1)}$), $j = k$ (via $x_k\,\partial W_k$), and $j > k$
        (via $B^{(1)}$); the $2\,\mathrm{Re}\{\cdots\}$ completes the
        frozen-coefficient scalars exactly, so $x_k$ must not be differentiated
        here. $\partial W_k$ comes from ``step_builder.derivative_step``
        (first-order or Frechet, matching the builder used by the evolution).
        Returns an array shaped like ``pulse.amplitudes``.
        """
        if result.backward is None:
            raise ValueError(
                "LindbladExpansionDifferentiator requires backward states."
            )
        insertion_operators = result.metadata.get("insertion_operators")
        if insertion_operators is None:
            raise ValueError(
                "LindbladExpansionDifferentiator requires a LindbladExpansionEvolution result."
            )

        dt = result.metadata["dt"]
        target_state = result.backward[-1].components[0]
        closed_amplitude = np.vdot(target_state, result.forward[-1].components[0])
        gradient = np.zeros_like(pulse.amplitudes)

        for step_index, step in enumerate(result.steps):
            unitary_step = UnitaryStep(W=step.W)
            controls = pulse.controls_at(step_index)
            t = step_index * pulse.dt
            forward_plain = result.forward[step_index].components[0]
            forward_single = result.forward[step_index].components[1]
            backward_plain = result.backward[step_index + 1].components[0]
            backward_single = result.backward[step_index + 1].components[1]
            insertion = insertion_operators[step_index + 1]

            for control_index in range(pulse.n_controls):
                dW = self.step_builder.derivative_step(
                    system,
                    controls,
                    pulse.dt,
                    control_index,
                    unitary_step,
                    t=t,
                ).W
                dW_forward = dW @ forward_plain
                correction = 2.0 * dt * np.real(
                    np.vdot(backward_plain, dW @ forward_single)
                    + np.vdot(backward_plain, insertion @ dW_forward)
                    + np.vdot(backward_single, dW_forward)
                )
                value = correction
                if self.include_closed:
                    value = value + 2.0 * np.real(
                        np.conj(closed_amplitude) * np.vdot(backward_plain, dW_forward)
                    )
                gradient[step_index, control_index] = value

        return gradient
