``experiemnts/evaluation-check`` folder check four ways calculating the fidelity in spin-boson physical system. 

1. use the evaluation given by grape (raw objective)
2. ``quantum_control/evaluation`` noisy_gate_fidelity
3. ``quantum_control/evaluation`` closed_gate_fidelity
4. ``quantum_control/evaluation`` faithful_gate_fidelity

The experiment will go over a optimization process using ``spin-boson-origin.yaml`` the evaluation will be done each 20 steps. The evaluation is stored in ``outputs`` subfolder. 

