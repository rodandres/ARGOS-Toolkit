import numpy as np

from pathlib import Path
import sys

from scipy.integrate import solve_ivp

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from py.modules.cislunar_astrodynamics.cr3bp_engine import (
    cr3bp,
    eigenvalues_on_L_points,
    find_L_points,
)
from py.modules.cislunar_astrodynamics.lyapunov_halo_engine import (
    analyze_monodromy,
    build_halo_seed,
    compute_lyapunov_family,
    differential_corrector_halo,
    differential_corrector_lyapunov,
)
from py.modules.visualization.astro_graphs import (
    plot_beta_evolution,
    plot_halo,
)

MU = 1.215e-2


if __name__ == "__main__":

    # ==========================================================
    # L2 equilibrium point
    # ==========================================================

    L1, L2, L3 = find_L_points(MU)

    print(f"L2 = {L2:.10f}")

    # ==========================================================
    # Initial Lyapunov seed from the linear center manifold
    # ==========================================================

    _, l2_eigen_solution, _ = eigenvalues_on_L_points(MU)
    eigenvalues_l2, eigenvectors_l2 = l2_eigen_solution

    seed_direction = np.real(eigenvectors_l2[:, 2])
    seed_direction /= np.linalg.norm(seed_direction)

    period_guess = 2.0 * np.pi / abs(eigenvalues_l2[2].imag)
    perturbation_amplitude = 5e-3

    x0_seed = L2 + perturbation_amplitude * seed_direction[0]
    ydot0_seed = perturbation_amplitude * seed_direction[4]

    print(f"\nInitial seed: x0 = {x0_seed:.8f}, ydot0 = {ydot0_seed:.8f}")

    x0_corrected, ydot0_corrected, half_period = differential_corrector_lyapunov(
        x0_seed,
        ydot0_seed,
        MU,
        period_guess,
        verbose=False,
    )

    if x0_corrected is None:
        raise RuntimeError("Lyapunov differential corrector failed for the initial seed.")

    print(
        f"First Lyapunov orbit: "
        f"x0 = {x0_corrected:.10f}  "
        f"ydot0 = {ydot0_corrected:.10f}  "
        f"T = {2.0 * half_period:.8f}"
    )

    # ==========================================================
    # Generate the Lyapunov family
    # ==========================================================

    print(f"\n{'═' * 60}")
    print("  GENERATING LYAPUNOV FAMILY")
    print(f"{'═' * 60}")

    family, bifurcation_orbit = compute_lyapunov_family(
        x0_start=x0_corrected,
        ydot0_start=ydot0_corrected,
        initial_period=2.0 * half_period,
        mu=MU,
        number_of_orbits=100,
        x0_step=5e-4,
        stop_at_bifurcation=True,
        verbose=False,
    )

    print(f"\nGenerated {len(family)} Lyapunov orbits.")

    if bifurcation_orbit is None:
        print("\nERROR: No bifurcation detected.")
        print("Suggestions:")
        print("  - Increase number_of_orbits.")
        print("  - Reduce x0_step to avoid skipping the bifurcation.")
        print("  - Verify that the family reaches y_amplitude ≈ 0.05–0.10 DU.")

        plot_beta_evolution(family)
        sys.exit()

    # ==========================================================
    # Bifurcation point
    # ==========================================================

    print(f"\n{'═' * 60}")
    print("  BIFURCATION POINT")
    print(f"{'═' * 60}")

    print(f"  Family index      : {bifurcation_orbit['index']}")
    print(f"  x0                : {bifurcation_orbit['x0']:.10f}")
    print(f"  ydot0             : {bifurcation_orbit['ydot0']:.10f}")
    print(f"  Period            : {bifurcation_orbit['full_period']:.8f}")
    print(
        f"  y amplitude       : "
        f"{bifurcation_orbit['y_amplitude']:.6f} DU "
        f"({bifurcation_orbit['y_amplitude'] * 384400:.0f} km)"
    )
    print(f"  βz                : {bifurcation_orbit['beta_z']:.4f}°")
    print(f"  Distance to λ = 1 : {bifurcation_orbit['distance_to_unity']:.6f}")

    print("\nMonodromy analysis:")
    analyze_monodromy(
        bifurcation_orbit["monodromy_matrix"],
        verbose=True,
    )

    plot_beta_evolution(family)

    # ==========================================================
    # Construct Halo seed
    # ==========================================================

    print(f"\n{'═' * 60}")
    print("  BUILDING HALO SEED")
    print(f"{'═' * 60}")

    bifurcation_amplitude = bifurcation_orbit["y_amplitude"]

    for scale_factor in [0.01, 0.05, 0.10, 0.20, 0.30]:

        vertical_perturbation = scale_factor * bifurcation_amplitude

        print(
            f"\nTesting vertical perturbation = {vertical_perturbation:.4f} DU "
            f"({scale_factor * 100:.0f}% of y amplitude)"
        )

        x0_seed, z0_seed, ydot0_seed = build_halo_seed(
            bifurcation_orbit,
            vertical_perturbation=vertical_perturbation,
            verbose=(scale_factor == 0.01),
        )

        x0_halo, z0_halo, ydot0_halo, half_period_halo = differential_corrector_halo(
            x0_seed,
            z0_seed,
            ydot0_seed,
            MU,
            period_guess=bifurcation_orbit["full_period"],
            tol=1e-12,
            max_iter=50,
            verbose=True,
        )

        if x0_halo is not None and abs(z0_halo) > 1e-6:
            print(f"Converged with z0 = {z0_halo:.6f}.")
            break

        if x0_halo is not None:
            print(
                f"Solution collapsed back to the planar family "
                f"(z0 = {z0_halo:.2e})."
            )

    if x0_halo is None:
        print("\nERROR: Halo differential corrector failed.")
        print("Reduce x0_step in the Lyapunov continuation to improve the")
        print("bifurcation point accuracy.")
        sys.exit()

    # ==========================================================
    # Propagate the corrected Halo orbit
    # ==========================================================

    full_period_halo = 2.0 * half_period_halo

    print(f"\n{'═' * 60}")
    print("  CORRECTED HALO ORBIT")
    print(f"{'═' * 60}")

    print(f"  x0     = {x0_halo:.12f}")
    print(f"  z0     = {z0_halo:.12f}")
    print(f"  ydot0  = {ydot0_halo:.12f}")
    print(
        f"  Period = {full_period_halo:.8f} TU "
        f"({full_period_halo * 375700 / 86400:.2f} days)"
    )

    halo_solution = solve_ivp(
        cr3bp,
        [0.0, full_period_halo],
        [x0_halo, 0.0, z0_halo, 0.0, ydot0_halo, 0.0],
        args=(MU,),
        rtol=1e-12,
        atol=1e-13,
        max_step=5.0 * full_period_halo / 1000.0,
    )

    final_state = halo_solution.y[:, -1]

    y_amplitude = np.max(np.abs(halo_solution.y[1]))
    z_amplitude = np.max(np.abs(halo_solution.y[2]))

    distance_unit_km = 384400.0

    print("\nClosure error:")
    print(f"  Δx     = {final_state[0] - x0_halo:+.3e}")
    print(f"  Δz     = {final_state[2] - z0_halo:+.3e}")
    print(f"  xdot   = {final_state[3]:+.3e}")
    print(f"  ydot   = {final_state[4]:+.3e}")
    print(f"  zdot   = {final_state[5]:+.3e}")

    print("\nOrbit amplitudes:")
    print(f"  y amplitude = {y_amplitude:.6f} DU ({y_amplitude * distance_unit_km:.0f} km)")
    print(f"  z amplitude = {z_amplitude:.6f} DU ({z_amplitude * distance_unit_km:.0f} km)")

    plot_halo(
        halo_solution,
        bifurcation_orbit,
        MU,
        L2,
    )