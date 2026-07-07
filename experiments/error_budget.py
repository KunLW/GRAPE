from pathlib import Path

from experiments.spin_boson_perturbative_lbfgsb import (
    N_LEVELS,
    spin_boson_noisy_control_system,
)
from physical_systems.spin_boson import (
    motion_resolved_gate_state_pairs,
)
from quantum_control import (
    ErrorBudgetConfig,
    evaluate_error_budget,
    load_pulse_npz,
    ms_xx_pi_over_2_gate,
    write_error_budget_report,
)

if __name__ == "__main__":
    run_dir = Path(
        "/Users/kun/Documents/GRAPE VERGE/experiments/outputs/"
        "spin_boson_perturbative_sweep_20260701_141220/noise_seed_12345"
    )

    pulse = load_pulse_npz(run_dir / "final_pulse.npz")
    system = spin_boson_noisy_control_system(N_LEVELS, phi_s=0.0)
    state_pairs = motion_resolved_gate_state_pairs(ms_xx_pi_over_2_gate(), N_LEVELS)

    config = ErrorBudgetConfig(
        gradient_samples=16,
        mc_samples=64,
        fluctuation_scales=(0.25, 0.5, 1.0),
        random_seed=12345,
    )

    report = evaluate_error_budget(system, pulse, state_pairs, config)
    write_error_budget_report(report, run_dir / "error_budget_final")