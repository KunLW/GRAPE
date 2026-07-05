from __future__ import annotations

import numpy as np

from quantum_control.evolution.expansion_evolution import PerturbativeExpansionEvolution
from quantum_control.results.expansion_result import ExpansionResult
from quantum_control.steps.perturbative_step import PerturbativeStep
from quantum_control.steps.unitary_step import UnitaryStepBuilder


class LindbladExpansionEvolution(PerturbativeExpansionEvolution):
    r"""First-order Lindblad decoherence correction chains (doc/lindblad_note.md).

    Two-pass evolution: the plain forward/backward chains $F_k/B_k$ fix
    the per-slice scalars $a_{k\mu} = \langle B_k|L_\mu|F_k\rangle$,
    $b_{k\mu} = \langle B_k|L_\mu^\dagger L_\mu|F_k\rangle$,
    $s = \langle B_k|F_k\rangle$; those define the frozen-coefficient insertion
    $$
        x_k = \sum_\mu \Big[ a_{k\mu}^* L_\mu
                             - \frac{1}{2} s^* L_\mu^\dagger L_\mu
                             - \frac{1}{2} b_{k\mu}^* \mathbb{1} \Big]
    $$
    and the second pass runs the single-insertion lemma recursion with
    $V_k = x_k W_k$, seeded with $F_0^{(1)} = x_0 F_0^{(0)}$ so the slice sum
    covers $k = 0 \ldots N$. Backward states are always computed because $x_k$
    needs $B_k$ even for value-only evaluation.
    """

    def __init__(self, step_builder=None, collapse_operators=None):
        super().__init__(step_builder or UnitaryStepBuilder(), max_order=1)
        self.collapse_operators = collapse_operators

    def evolve(self, system, pulse, context):
        if context.target_state is None:
            raise ValueError(
                "LindbladExpansionEvolution requires context.target_state: "
                "the insertion operators x_k contract backward states."
            )
        initial_state = np.asarray(context.initial_state, dtype=complex)
        target_state = np.asarray(context.target_state, dtype=complex)

        unitary_steps = [
            self.step_builder.build_step(
                system,
                pulse.controls_at(step_index),
                pulse.dt,
                t=step_index * pulse.dt,
            )
            for step_index in range(pulse.n_steps)
        ]

        forward_plain = self._plain_forward(unitary_steps, initial_state)
        backward_plain = self._plain_backward(unitary_steps, target_state)
        overlap = np.vdot(target_state, forward_plain[-1])

        collapse_operators = self._collapse_operators(system)
        insertion_operators = self._insertion_operators(
            collapse_operators, forward_plain, backward_plain, overlap
        )

        steps = [
            PerturbativeStep(W=step.W, V=insertion_operators[step_index + 1] @ step.W)
            for step_index, step in enumerate(unitary_steps)
        ]
        forward = self._forward_states(
            steps,
            initial_state,
            seed_components={1: insertion_operators[0] @ initial_state},
        )
        backward = self._backward_states(steps, target_state)

        return ExpansionResult(
            steps=steps,
            forward=forward,
            backward=backward,
            max_order=1,
            metadata={
                "dt": pulse.dt,
                "initial_state": initial_state,
                "target_state": target_state,
                "insertion_operators": insertion_operators,
                "collapse_operators": collapse_operators,
                "overlap": overlap,
            },
        )

    def _collapse_operators(self, system):
        operators = self.collapse_operators
        if operators is None:
            operators = getattr(system, "collapse_operators", ())
        return [np.asarray(operator, dtype=complex) for operator in operators]

    @staticmethod
    def _insertion_operators(collapse_operators, forward_plain, backward_plain, overlap):
        dimension = forward_plain[0].shape[0]
        identity = np.eye(dimension, dtype=complex)
        products = [
            operator.conj().T @ operator for operator in collapse_operators
        ]
        insertions = []
        for forward_state, backward_state in zip(forward_plain, backward_plain):
            x = np.zeros((dimension, dimension), dtype=complex)
            for operator, product in zip(collapse_operators, products):
                a = np.vdot(backward_state, operator @ forward_state)
                b = np.vdot(backward_state, product @ forward_state)
                x = x + (
                    np.conj(a) * operator
                    - 0.5 * np.conj(overlap) * product
                    - 0.5 * np.conj(b) * identity
                )
            insertions.append(x)
        return insertions

    @staticmethod
    def _plain_forward(steps, initial_state):
        states = [initial_state]
        for step in steps:
            states.append(step.W @ states[-1])
        return states

    @staticmethod
    def _plain_backward(steps, target_state):
        states = [None] * (len(steps) + 1)
        states[-1] = target_state
        for step_index in reversed(range(len(steps))):
            states[step_index] = steps[step_index].W.conj().T @ states[step_index + 1]
        return states
