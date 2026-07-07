from experiments.spin_boson_perturbative_lbfgsb import (
    N_LEVELS,
    spin_boson_noisy_control_system,
)
from physical_systems.spin_boson import (
    motion_resolved_gate_state_pairs,
)
from quantum_control import (
    ms_xx_pi_over_2_gate,
)

def build():
    system = spin_boson_noisy_control_system(N_LEVELS, phi_s=0.0)
    state_pairs = motion_resolved_gate_state_pairs(ms_xx_pi_over_2_gate(), N_LEVELS)
    return system, state_pairs