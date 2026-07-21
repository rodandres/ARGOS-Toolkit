import numpy as np

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from py.modules.visualization.astro_graphs import (
    plot_family_summary,
    plot_lyapunov_family,
)

from py.modules.cislunar_astrodynamics.cr3bp_engine import (
    eigenvalues_on_L_points,
    find_L_points,
)
from py.modules.cislunar_astrodynamics.lyapunov_halo_engine import (
    compute_lyapunov_family,
    differential_corrector_lyapunov,
)

MU = 1.215e-2


if __name__ == "__main__":

    # ==========================================================
    # Initial Lyapunov orbit seed from the L2 center manifold
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

    print(f"L2 location      : {L2:.10f}")
    print(f"Initial seed     : x0 = {x0_seed:.8f}, ydot0 = {ydot0_seed:.8f}")
    print(f"Initial period   : {period_guess:.6f}\n")

    # ==========================================================
    # Differential correction of the initial Lyapunov orbit
    # ==========================================================

    print("=== Correcting initial Lyapunov orbit ===")

    x0_corrected, ydot0_corrected, half_period = differential_corrector_lyapunov(
        x0_seed,
        ydot0_seed,
        MU,
        period_guess,
    )

    if x0_corrected is None:
        raise RuntimeError("Initial Lyapunov orbit correction failed.")

    full_period = 2.0 * half_period

    print("\nCorrected initial orbit")
    print(f"  x0     = {x0_corrected:.12f}")
    print(f"  ydot0  = {ydot0_corrected:.12f}")
    print(f"  Period = {full_period:.8f}\n")

    # ==========================================================
    # Generate the Lyapunov family
    # ==========================================================

    print("=== Generating Lyapunov family ===")

    family, _ = compute_lyapunov_family(
        x0_start=x0_corrected,
        ydot0_start=ydot0_corrected,
        initial_period=full_period,
        mu=MU,
        number_of_orbits=80,
        x0_step=3e-3,
        stop_at_bifurcation=False,
        verbose=True,
    )

    print(f"\nGenerated {len(family)} Lyapunov orbits.")

    # ==========================================================
    # Visualization
    # ==========================================================

    print("\n=== Plotting results ===")

    plot_lyapunov_family(family, MU, L2)
    plot_family_summary(family, MU, L2)