"""
Preliminary CZ fidelity and process-tomography plots for the SiC STIRAP model.

This script is self-contained: the SiC parameters, Hamiltonian, collapse
operators, and pulse construction are copied from the SiC mesolve script.

Outputs:
    1) |++> input round-trip fidelity against the ideal CZ output state.
    2) Re(chi_sim) quantum-process-tomography bar plot in the two-qubit Pauli
       basis {II, IX, ..., ZZ}.

The process reconstruction follows the note workflow:
    16 physical inputs -> mesolve -> trace phonon -> project computational
    subspace without renormalizing -> S = Y @ pinv(X) -> chi_sim.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, colors
from qutip import basis, destroy, ket2dm, mesolve, qeye, tensor


PI = np.pi
BITS = ("00", "01", "10", "11")
PAULI_LABELS = (
    "II",
    "IX",
    "IY",
    "IZ",
    "XI",
    "XX",
    "XY",
    "XZ",
    "YI",
    "YX",
    "YY",
    "YZ",
    "ZI",
    "ZX",
    "ZY",
    "ZZ",
)


@dataclass(frozen=True)
class Params:
    wm: float = 2 * PI * 5.1939e9   
    g: float = 2 * PI * 31.9043e6
    gamma_e: float = 2 * PI * 5 * 1e6 # 我这里还是用的19年关于SiC实验的那篇文献（cleland引用的那篇）
    dephasing_e: float = 2 * PI * 30 * 1e3
    dephasing_g: float = 2 * PI * 1e3
    kappa: float = 2 * PI * 2.4676 * 2 
    # kappa: float = 1e3
    # 之前用的是Q算出来的振幅衰减速率，实际Lindblad算子里面用的是能量衰减速率

    n_phonon: int = 3
    n_steps: int = 300
    ramp_time: float = 15.0e-6
    coupling_scale: float = 1.0
    omega1_max: float = 2 * PI * 800.0e6
    omega1_mixing: float = 1.0
    omega2_mixing: float = 1.0
    omega2_max: float = 2 * PI * 40.0e6
    phase_slowdown: float = 80.0
    t0_scale: float = 1.0
    t1_scale: float = 1.0
    phase_t0_scale: float = 1.0
    phase_t1_scale: float = 1.0
    time_reference: str = "gap-mid"
    ramp_shape: str = "sin2"
    system: str = "closed"
    solver_method: str = "dop853"


def smoothstep(x):
    y = np.clip(x, 0.0, 1.0)
    return y * y * (3.0 - 2.0 * y)


def smootherstep(x):
    y = np.clip(x, 0.0, 1.0)
    return y * y * y * (y * (y * 6.0 - 15.0) + 10.0)


def ramp_profile(x, params: Params):
    y = np.clip(x, 0.0, 1.0)
    if params.ramp_shape == "sin":
        return np.sin(0.5 * PI * y)
    if params.ramp_shape == "sin2":
        return np.sin(0.5 * PI * y) ** 2
    if params.ramp_shape == "smoothstep":
        return smoothstep(y)
    if params.ramp_shape == "smootherstep":
        return smootherstep(y)
    raise ValueError("ramp_shape must be 'sin', 'sin2', 'smoothstep', or 'smootherstep'.")


def omega_r_max(params: Params):
    eta = params.g / params.wm
    return params.coupling_scale * params.omega1_mixing * eta * params.omega1_max


def omega2_max_value(params: Params):
    if params.omega2_max > 0.0:
        return params.omega2_mixing * params.omega2_max
    return omega_r_max(params)


def geometric_phase_coeff(theta):
    cos2 = np.cos(theta) ** 2
    cos4 = cos2**2
    sin2 = np.sin(theta) ** 2
    single = cos2
    extra = 2.0 * cos4 * sin2 / (2.0 - cos4)
    return single, extra


def phase_increments(params: Params):
    omega_r_plateau = omega_r_max(params)
    omega2_plateau = omega2_max_value(params)
    theta0 = np.arctan2(omega_r_plateau, omega2_plateau)
    single0, extra0 = geometric_phase_coeff(theta0)
    if extra0 <= 0.0:
        raise ValueError("The T0 plateau must have nonzero Omega_R and Omega2.")

    dphi0 = params.phase_t0_scale * PI / (2.0 * extra0)
    winding = int(np.ceil(single0 * dphi0 / PI))
    dphi1_base = 2.0 * PI * winding - 2.0 * single0 * dphi0
    if dphi1_base <= 1e-12:
        winding += 1
        dphi1_base = 2.0 * PI * winding - 2.0 * single0 * dphi0
    dphi1 = params.phase_t1_scale * dphi1_base
    return theta0, single0, extra0, dphi0, dphi1, winding


def cz_times(params: Params):
    omega_c = omega_r_max(params)
    omega2_plateau = omega2_max_value(params)
    if params.time_reference == "omega-r":
        omega_clock = omega_c
    elif params.time_reference == "omega2-max":
        omega_clock = omega2_plateau
    elif params.time_reference == "omega2-mid":
        omega_clock = omega2_plateau / np.sqrt(2.0)
    elif params.time_reference == "gap-mid":
        omega_clock = np.sqrt(0.5 * (omega_c**2 + omega2_plateau**2))
    else:
        raise ValueError(
            "time_reference must be 'omega-r', 'omega2-max', 'omega2-mid', or 'gap-mid'."
        )
    _, _, _, dphi0, dphi1, _ = phase_increments(params)
    phase_rate = omega_clock / params.phase_slowdown
    t0 = params.t0_scale * dphi0 / phase_rate
    t1 = params.t1_scale * dphi1 / phase_rate
    total_time = 4.0 * params.ramp_time + 2.0 * t0 + t1
    phase_rate0 = dphi0 / t0
    phase_rate1 = dphi1 / t1
    return omega_c, omega_clock, t0, t1, total_time, phase_rate0, phase_rate1


def pulse_envelopes(t, params: Params):
    _, _, t0, t1, _, phase_rate0, phase_rate1 = cz_times(params)
    r = params.ramp_time
    bounds = np.cumsum([0.0, r, t0, r, t1, r, t0, r])
    _, _, _, dphi0, dphi1, _ = phase_increments(params)
    omega_r_plateau = omega_r_max(params)
    omega2_plateau = omega2_max_value(params)

    omega_r = np.empty_like(t, dtype=float)
    omega2_abs = np.empty_like(t, dtype=float)
    phi = np.empty_like(t, dtype=float)
    for idx, x in enumerate(t):
        if x < bounds[1]:
            s = ramp_profile(x / r, params)
            omega_r[idx] = omega_r_plateau
            omega2_abs[idx] = omega2_plateau * s
            phi[idx] = 0.0
        elif x < bounds[2]:
            omega_r[idx] = omega_r_plateau
            omega2_abs[idx] = omega2_plateau
            phi[idx] = phase_rate0 * (x - bounds[1])
        elif x < bounds[3]:
            s = ramp_profile((x - bounds[2]) / r, params)
            omega_r[idx] = omega_r_plateau * (1.0 - s)
            omega2_abs[idx] = omega2_plateau
            phi[idx] = dphi0
        elif x < bounds[4]:
            omega_r[idx] = 0.0
            omega2_abs[idx] = omega2_plateau
            phi[idx] = dphi0 + phase_rate1 * (x - bounds[3])
        elif x < bounds[5]:
            s = ramp_profile((x - bounds[4]) / r, params)
            omega_r[idx] = omega_r_plateau * s
            omega2_abs[idx] = omega2_plateau
            phi[idx] = dphi0 + dphi1
        elif x < bounds[6]:
            omega_r[idx] = omega_r_plateau
            omega2_abs[idx] = omega2_plateau
            phi[idx] = dphi0 + dphi1 + phase_rate0 * (x - bounds[5])
        else:
            s = ramp_profile((x - bounds[6]) / r, params)
            omega_r[idx] = omega_r_plateau
            omega2_abs[idx] = omega2_plateau * (1.0 - s)
            phi[idx] = 2.0 * dphi0 + dphi1
    theta = np.arctan2(np.abs(omega_r), np.abs(omega2_abs))
    return theta, phi, omega_r, omega2_abs


def pulse_values(t, params: Params):
    eta = params.g / params.wm
    theta, phi, omega_r, omega2_abs = pulse_envelopes(t, params)
    omega1 = omega_r / eta
    omega2 = omega2_abs * np.exp(1j * phi)
    return theta, phi, omega1, omega2


def op_spin(op, which, n_phonon):
    i_el = qeye(4)
    i_ph = qeye(n_phonon)
    return tensor(op, i_el, i_ph) if which == 0 else tensor(i_el, op, i_ph)


def ket_state(bits, n_phonon):
    if bits not in {"00", "01", "10", "11"}:
        raise ValueError("bits must be one of 00, 01, 10, 11.")
    g2 = basis(4, 1)
    g3 = basis(4, 2)
    states = {"0": g3, "1": g2}
    return tensor(states[bits[0]], states[bits[1]], basis(n_phonon, 0))


def build_model(params: Params):
    if params.n_phonon < 3:
        raise ValueError("n_phonon must be at least 3 for the |11> two-phonon path.")

    omega_c, omega_clock, t0, t1, total_time, _, _ = cz_times(params)
    tlist = np.linspace(0.0, total_time, params.n_steps)
    theta, phi, omega1, omega2 = pulse_values(tlist, params)
    omega_r = (params.g / params.wm) * omega1

    g1 = basis(4, 0)
    g2 = basis(4, 1)
    e = basis(4, 3)
    p_g1 = g1 * g1.dag()
    p_g2 = g2 * g2.dag()
    p_e = e * e.dag()
    sigma_e_g1 = e * g1.dag()
    sigma_e_g2 = e * g2.dag()

    i_el = qeye(4)
    b = destroy(params.n_phonon)
    b_full = tensor(i_el, i_el, b)

    hamiltonian = []
    for spin in (0, 1):
        red_base = (
            tensor(sigma_e_g1, i_el, b)
            if spin == 0
            else tensor(i_el, sigma_e_g1, b)
        )
        red = -0.5 * red_base
        carrier_up = 0.5 * op_spin(sigma_e_g2, spin, params.n_phonon)
        hamiltonian += [
            [(red + red.dag()).to("csr"), omega_r],
            [carrier_up.to("csr"), omega2],
            [carrier_up.dag().to("csr"), np.conjugate(omega2)],
        ]

    c_ops = []
    if params.system == "open":
        decay_rate = params.gamma_e
        for spin in (0, 1):
            c_ops += [
                (np.sqrt(decay_rate) * op_spin(g1 * e.dag(), spin, params.n_phonon)).to("csr"),
                (np.sqrt(decay_rate) * op_spin(g2 * e.dag(), spin, params.n_phonon)).to("csr"),
            ]
            # pure dephasing of excited state
            c_ops.append(
                (np.sqrt(2 * params.dephasing_e) * op_spin(e * e.dag(), spin, params.n_phonon)).to("csr")
            )
            c_ops.append(
                (np.sqrt(2 * params.dephasing_g) * op_spin(g2 * g2.dag(), spin, params.n_phonon)).to("csr")
            )
        c_ops.append((np.sqrt(params.kappa) * b_full).to("csr"))

    return {
        "t": tlist,
        "theta": theta,
        "phi": phi,
        "Omega1": omega1,
        "Omega2": omega2,
        "H": hamiltonian,
        "c_ops": c_ops,
        "T0": t0,
        "T1": t1,
        "total_time": total_time,
        "omega_c": omega_c,
        "omega_clock": omega_clock,
    }

# only to change Qobj into complex number
def qobj_scalar(value):
    if hasattr(value, "full"):
        return complex(value.full()[0, 0])
    return complex(value)


def solver_options(params: Params, store_states: bool):
    return {
        "nsteps": 100000,
        "atol": 1e-8,
        "rtol": 1e-7,
        "progress_bar": "",
        "method": params.solver_method,
        "store_states": store_states,
    }


def single_logical_ket(label: str):
    g2 = basis(4, 1)
    g3 = basis(4, 2)
    if label == "0":
        return g3
    if label == "1":
        return g2
    if label == "+":
        return (g3 + g2).unit()
    if label == "+i":
        return (g3 + 1j * g2).unit()
    raise ValueError(f"Unknown single-qubit logical state: {label}")


def spin_logical_ket(two_label):
    a, b = two_label
    return tensor(single_logical_ket(a), single_logical_ket(b))


def full_input_density(params: Params, two_label):
    spin_ket = spin_logical_ket(two_label)
    phonon0 = basis(params.n_phonon, 0)
    return tensor(ket2dm(spin_ket), ket2dm(phonon0)) # |ij><ij| * |0><0|


def computational_spin_kets():
    return [spin_logical_ket((bits[0], bits[1])) for bits in BITS] # |00> |01> |10> |11>


def ideal_cz_state_from_spin_ket(spin_ket):
    amps = np.array([qobj_scalar(ket.dag() * spin_ket) for ket in computational_spin_kets()])
    amps[3] *= -1.0
    out = sum(amp * ket for amp, ket in zip(amps, computational_spin_kets()))
    return out.unit() # normalization


def spin_density_from_full_state(state):
    rho = ket2dm(state) if state.isket else state
    return rho.ptrace([0, 1])


def projected_spin_matrix(rho_spin):
    kets = computational_spin_kets()
    mat = np.zeros((4, 4), dtype=complex)
    for row, ket_row in enumerate(kets):
        for col, ket_col in enumerate(kets):
            mat[row, col] = qobj_scalar(ket_row.dag() * rho_spin * ket_col)
    return mat


def vec(mat):
    return np.asarray(mat, dtype=complex).reshape(-1, order="F")


def run_plus_roundtrip(params: Params):
    model = build_model(params)
    rho0 = full_input_density(params, ("+", "+")) # ==> |++><++| * |0><0|
    target_spin = ideal_cz_state_from_spin_ket(spin_logical_ket(("+", "+"))) # ==> CZ|++>
    p_target = target_spin * target_spin.dag()

    result = mesolve(
        model["H"],
        rho0,
        model["t"],
        c_ops=model["c_ops"] if params.system == "open" else [],
        e_ops=[],
        options=solver_options(params, store_states=True),
    )

    fidelities = []
    survival = []
    for state in result.states:
        rho_spin = spin_density_from_full_state(state)
        rho_comp = projected_spin_matrix(rho_spin)
        fidelities.append(np.real(qobj_scalar(target_spin.dag() * rho_spin * target_spin)))
        survival.append(np.real(np.trace(rho_comp)))

    return {
        "t": model["t"],
        "fidelity": np.array(fidelities),
        "survival": np.array(survival),
        "target_spin": target_spin,
        "final_spin": spin_density_from_full_state(result.states[-1]),
        "T0": model["T0"],
        "T1": model["T1"],
        "total_time": model["total_time"],
    }


def tomography_input_labels():
    single = ("0", "1", "+", "+i")
    return [(a, b) for a in single for b in single]


def reconstruct_superoperator(params: Params):
    model = build_model(params)
    input_cols = []
    output_cols = []

    for label in tomography_input_labels(): # 16 input states ==> (i,j)
        rho0 = full_input_density(params, label) # ==> |ij><ij| * |0><0|
        rho_in_spin = rho0.ptrace([0, 1]) # trace phonon ==> |ij><ij|
        rho_in_comp = projected_spin_matrix(rho_in_spin)

        result = mesolve(
            model["H"],
            rho0,
            model["t"],
            c_ops=model["c_ops"] if params.system == "open" else [],
            e_ops=[],
            options=solver_options(params, store_states=True),
        )
        rho_out_spin = spin_density_from_full_state(result.states[-1])
        rho_out_comp = projected_spin_matrix(rho_out_spin)

        input_cols.append(vec(rho_in_comp))
        output_cols.append(vec(rho_out_comp))

    x_mat = np.column_stack(input_cols)
    y_mat = np.column_stack(output_cols)
    return y_mat @ np.linalg.pinv(x_mat) # Y = SX ==> S = YX^-1


def pauli_basis():
    ident = np.array([[1, 0], [0, 1]], dtype=complex)
    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
    single = (ident, sigma_x, sigma_y, sigma_z) # define Pauli matrix as 4 basis
    return [np.kron(a, b) for a in single for b in single]


def superoperator_to_chi(superop):
    paulis = pauli_basis()
    cols = []
    for em in paulis:
        for en in paulis:
            basis_superop = np.kron(en.conj(), em)
            cols.append(vec(basis_superop))
    transfer = np.column_stack(cols)
    coeffs = np.linalg.solve(transfer, vec(superop))
    return coeffs.reshape((16, 16), order="C")


def ideal_cz_chi():
    coeffs = np.zeros(16, dtype=complex)
    for label, value in {"II": 0.5, "IZ": 0.5, "ZI": 0.5, "ZZ": -0.5}.items():
        coeffs[PAULI_LABELS.index(label)] = value
    return np.outer(coeffs, coeffs.conj())


def chi_metrics(chi):
    chi_cz = ideal_cz_chi()
    process_fidelity = float(np.real(np.trace(chi_cz.conj().T @ chi)))
    average_fidelity = (4.0 * process_fidelity + 1.0) / 5.0
    return process_fidelity, average_fidelity


def plot_plus_roundtrip(roundtrip, params: Params, output_path: Path):
    t_us = roundtrip["t"] * 1e6
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.plot(t_us, roundtrip["fidelity"], lw=2.4, label="Fidelity to CZ|++>")
    # ax.plot(t_us, roundtrip["survival"], "--", lw=1.8, label="Computational survival")
    ax.set_xlabel("Interaction time (us)")
    ax.set_ylabel("Fidelity")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.25)
    # ax.legend(loc="best")
    ax.set_title(
        "|++> round-trip check, "
        f"F(T)={roundtrip['fidelity'][-1]:.4f}, "
        f"T={roundtrip['total_time'] * 1e6:.1f} us"
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.show()


def draw_bar_outline(ax, x, y, dx, dy, dz, *, color="k", linestyle="--", linewidth=1.0):
    z0 = 0.0
    z1 = float(dz)
    corners0 = [
        (x, y, z0),
        (x + dx, y, z0),
        (x + dx, y + dy, z0),
        (x, y + dy, z0),
    ]
    corners1 = [
        (x, y, z1),
        (x + dx, y, z1),
        (x + dx, y + dy, z1),
        (x, y + dy, z1),
    ]
    for corners in (corners0, corners1):
        for start, end in zip(corners, corners[1:] + corners[:1]):
            ax.plot(
                [start[0], end[0]],
                [start[1], end[1]],
                [start[2], end[2]],
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
            )
    for start, end in zip(corners0, corners1):
        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            [start[2], end[2]],
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
        )


def draw_z_grid_lines(ax, z_ticks, xlim, ylim):
    grid_color = (0.86, 0.86, 0.86, 1.0)
    for z in z_ticks:
        ax.plot(xlim, [ylim[1], ylim[1]], [z, z], color=grid_color, linewidth=1.4, zorder=0)
        ax.plot([xlim[0], xlim[0]], ylim, [z, z], color=grid_color, linewidth=1.4, zorder=0)


def plot_chi_real(chi, output_path: Path, title_suffix: str = ""):
    data = np.real(chi)
    fig = plt.figure(figsize=(9.8, 7.4))
    ax = fig.add_subplot(111, projection="3d")

    xpos, ypos = np.meshgrid(np.arange(16), np.arange(16), indexing="ij")
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = np.zeros_like(xpos, dtype=float)
    dz = data.ravel(order="C")

    dx = 0.72 * np.ones_like(zpos)
    dy = 0.72 * np.ones_like(zpos)
    norm = colors.TwoSlopeNorm(vmin=-0.3, vcenter=0.0, vmax=0.3)
    checker = (xpos + ypos) % 2
    floor_colors = np.where(checker[:, None] == 0, 0.82, 0.52)
    floor_colors = np.repeat(floor_colors, 4, axis=1)
    floor_colors[:, 3] = 1.0
    ax.bar3d(
        xpos,
        ypos,
        zpos,
        dx,
        dy,
        np.zeros_like(dz),
        shade=False,
        color=floor_colors,
        edgecolor="k",
        linewidth=0.22,
        zsort="min",
    )

    visible = np.abs(dz) > 1e-5
    bar_colors = cm.RdBu_r(norm(dz[visible]))
    ax.bar3d(
        xpos[visible],
        ypos[visible],
        zpos[visible],
        dx[visible],
        dy[visible],
        dz[visible],
        shade=True,
        color=bar_colors,
        edgecolor="k",
        linewidth=0.35,
    )

    ideal = np.real(ideal_cz_chi()).ravel(order="C")
    ideal_visible = np.abs(ideal) > 1e-12
    for x, y, height in zip(xpos[ideal_visible], ypos[ideal_visible], ideal[ideal_visible]):
        draw_bar_outline(
            ax,
            x - 0.04,
            y - 0.04,
            0.80,
            0.80,
            height,
            color="k",
            linestyle="--",
            linewidth=1.05,
        )

    ax.set_xticks(np.arange(16) + 0.36)
    ax.set_yticks(np.arange(16) + 0.36)
    ax.set_xticklabels(PAULI_LABELS, rotation=65, ha="right", fontsize=8)
    ax.set_yticklabels(PAULI_LABELS, rotation=-25, ha="left", fontsize=8)
    ax.set_xlim(-0.4, 16.2)
    ax.set_ylim(-0.4, 16.2)
    ax.set_zlim(-0.32, 0.32)
    z_ticks = np.arange(-0.3, 0.31, 0.1)
    ax.set_zticks(z_ticks)
    ax.set_zlabel(r"Re(\chi_sim)")
    ax.set_title("Quantum process tomography: " + title_suffix)
    ax.view_init(elev=23, azim=-58)
    ax.grid(False)
    draw_z_grid_lines(ax, z_ticks, (-0.4, 16.2), (-0.4, 16.2))

    mappable = cm.ScalarMappable(norm=norm, cmap=cm.RdBu_r)
    mappable.set_array([])
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.65, pad=0.08)
    cbar.set_label("chi_real")
    fig.subplots_adjust(left=0.02, right=0.86, bottom=0.08, top=0.92)
    fig.savefig(output_path, dpi=180)
    plt.show()


def make_params(args):
    return replace(
        Params(),
        system=args.system,
        n_steps=args.n_steps,
        n_phonon=args.n_phonon,
        ramp_time=args.ramp_time_us * 1e-6,
        phase_slowdown=args.phase_slowdown,
        solver_method=args.solver_method,
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", choices=["open", "closed"], default="open")
    parser.add_argument("--n-steps", type=int, default=300)
    parser.add_argument("--n-phonon", type=int, default=3)
    parser.add_argument("--ramp-time-us", type=float, default=Params().ramp_time * 1e6)
    parser.add_argument("--phase-slowdown", type=float, default=Params().phase_slowdown)
    parser.add_argument(
        "--solver-method",
        choices=["adams", "bdf", "lsoda", "dop853", "vern9"],
        default=Params().solver_method,
    )
    parser.add_argument("--output-prefix", default="sic_cz")
    parser.add_argument("--show", action="store_true", default='true')
    return parser.parse_args()


def main():
    args = parse_args()
    params = make_params(args)
    out_dir = Path(__file__).resolve().parent
    plus_path = out_dir / f"{args.output_prefix}_plus_roundtrip_fidelity.png"
    chi_path = out_dir / f"{args.output_prefix}_chi_real_qpt.png"

    print("SiC STIRAP CZ preliminary fidelity/QPT")
    print(f"system={params.system}, n_steps={params.n_steps}, n_phonon={params.n_phonon}")
    print(
        f"ramp={params.ramp_time * 1e6:.3f} us, "
        f"phase_slowdown={params.phase_slowdown:.3f}, "
        f"dephasing_e/2pi={params.dephasing_e / (2 * PI * 1e3):.3f} kHz, "
        f"dephasing_g/2pi={params.dephasing_g / (2 * PI * 1e3):.3f} kHz"
    )

    roundtrip = run_plus_roundtrip(params)
    plot_plus_roundtrip(roundtrip, params, plus_path)
    print(f"|++> final fidelity: {roundtrip['fidelity'][-1]:.9f}")
    print(f"|++> final computational survival: {roundtrip['survival'][-1]:.9f}")

    superop = reconstruct_superoperator(params)
    chi = superoperator_to_chi(superop)
    process_fidelity, average_fidelity = chi_metrics(chi) # calculate fidelity
    plot_chi_real(
        chi,
        chi_path,
        title_suffix=f" F_process={process_fidelity:.4f}, F_average={average_fidelity:.4f}",
    )
    print(f"process fidelity Tr(chi_CZ chi_sim): {process_fidelity:.9f}")
    print(f"average gate fidelity: {average_fidelity:.9f}")



if __name__ == "__main__":
    main()
