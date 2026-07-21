import numpy as np
from scipy.integrate import solve_ivp

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from py.modules.visualization.astro_graphs import (
    plot_L_points,
    plot_eigenvalues,
    plot_trajectory,
)

from py.modules.cislunar_astrodynamics.cr3bp_engine import (
    cr3bp,
    eigenvalues_on_L_points,
    find_L_points,
)
from py.modules.cislunar_astrodynamics.lyapunov_halo_engine import (
    differential_corrector_lyapunov,
)

MU = 1.215e-2


if __name__ == "__main__":

    # ==========================================================
    # Lagrange points and L2 linear dynamics
    # ==========================================================

    L1, L2, L3 = find_L_points(MU)

    _, l2_eigen_solution, _ = eigenvalues_on_L_points(MU)
    eigenvalues_l2, eigenvectors_l2 = l2_eigen_solution

    seed_eigenvalue = eigenvalues_l2[2]
    seed_eigenvector = eigenvectors_l2[:, 2]

    seed_direction = np.real(seed_eigenvector)
    seed_direction /= np.linalg.norm(seed_direction)

    period_guess = 2.0 * np.pi / abs(seed_eigenvalue.imag)
    perturbation_amplitude = 5e-3

    x0_seed = L2 + perturbation_amplitude * seed_direction[0]
    ydot0_seed = perturbation_amplitude * seed_direction[4]

    print(f"Initial seed      : x0 = {x0_seed:.8f}, ydot0 = {ydot0_seed:.8f}")
    print(f"Initial period    : {period_guess:.6f}")
    print(f"L2 displacement   : {x0_seed - L2:.4e}")

    # ==========================================================
    # Differential correction
    # ==========================================================

    x0_corrected, ydot0_corrected, half_period = differential_corrector_lyapunov(
        x0_seed,
        ydot0_seed,
        MU,
        period_guess,
    )

    if x0_corrected is None:
        raise RuntimeError("Lyapunov differential correction failed.")

    full_period = 2.0 * half_period

    print("\nCorrected initial conditions")
    print(f"  x0     = {x0_corrected:.12f}")
    print(f"  ydot0  = {ydot0_corrected:.12f}")
    print(f"  Period = {full_period:.8f}")

    initial_state = np.array(
        [
            x0_corrected,
            0.0,
            0.0,
            0.0,
            ydot0_corrected,
            0.0,
        ]
    )

    solution = solve_ivp(
        cr3bp,
        [0.0, 3.0 * full_period],
        initial_state,
        args=(MU,),
        rtol=1e-12,
        atol=1e-13,
        dense_output=True,
    )

    final_state = solution.y[:, -1]

    print("\nClosure error")
    print(f"  Δx     = {final_state[0] - x0_corrected:+.2e}")
    print(f"  Δydot  = {final_state[4] - ydot0_corrected:+.2e}")
    print(f"  xdot   = {final_state[3]:+.2e}")

    plot_trajectory(
        solution,
        MU,
        L_point=L2,
    )