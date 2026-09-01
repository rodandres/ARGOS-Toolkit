import numpy as np
import matplotlib.pyplot as plt
from fractions import Fraction
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from py.modules.solvers.RK45 import rk45
from py.modules.cislunar_astrodynamics.cr3bp_engine import cr3bp, find_L_points, eigenvalues_on_L_points

from py.modules.cislunar_astrodynamics.lyapunov_halo_engine import *

orbits = []

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

# Correcting initial Lyapunov orbit
x0_corrected, ydot0_corrected, half_period, _ = differential_corrector(
    x0=x0_seed,
    ydot0=ydot0_seed,
    mu=MU,
    period_guess=period_guess / 2.0,
    verbose=False,
)

print(f"Initial Lyapunov orbit: x0 = {x0_corrected:.6f}, ydot0 = {ydot0_corrected:.6f}, half_period = {half_period:.6f}")

orbits.append(OrbitData(
    x0=x0_corrected,
    ydot0=ydot0_corrected,
    T_half=half_period,
    )
)

family, _ = compute_lyapunov_family(
        x0_start=x0_corrected,
        ydot0_start=ydot0_corrected,
        initial_period=half_period * 2.0,
        mu=MU,
        number_of_orbits=15,
        x0_step=3e-3,
        stop_at_bifurcation=True,
        verbose=False,
    )

orbits.extend(family)

birfurcation_orbit = orbits[-1]
birfurcation_orbit_index = len(orbits) -1 

x0, z0, ydot0 = build_halo_seed(birfurcation_orbit, vertical_perturbation=-5e-2)

initial_state = build_state_vector(
    x0,
    z0,
    ydot0,
    birfurcation_orbit.T_half
)

family = compute_halo_family(
    initial_state,
        MU,
        initial_step_size=5e-4,
        maximum_step_size=1.5e-2,
        minimum_step_size=1e-8,
        target_perilune_km=1800.0,
        max_steps=20000,
        orientation_hint=np.sign(initial_state[1]),
        verbose=False,
    )

orbits.extend(family)

def plots(orbits: list[OrbitData]):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    n = len(orbits)

    for i, orbit in enumerate(orbits):
        
        if not (i == 1 or i >= n - 2 or i % 5 == 0 or i == 2 or i == birfurcation_orbit_index):
            continue

        state_vector = np.array([
            orbit.x0, orbit.y0, orbit.z0,
            orbit.xdot0, orbit.ydot0, orbit.zdot0
        ])

        period = orbit.T_half*2

        sol = rk45(
            fun=cr3bp,
            t_span=(0, period),
            Y0=state_vector,
            args=(MU,),
            h0=5e-4,
            h_min=1e-8,
            h_max=5e-3,
            h_adaptative=True,
        )

        pos = sol.y[:3]

        # Color y etiqueta
        if i == 2:
            color = "green"
            alpha = 1.0
            label = "Initial Lyapunov"

        elif i == birfurcation_orbit_index:
            color = "blue"
            alpha = 1.0
            label = "Birfurcation (Lyapunov to Halo)"        

        elif i == n - 2:
            lunar_period = 27.321661
            ratio = orbit.T_half*2 * TU_DAYS / lunar_period
            resonance = Fraction(ratio).limit_denominator(20)
            color = "tab:red"
            alpha = 1.0
            label = f"NRHO - Resonance:  {resonance.denominator}:{resonance.numerator}"

            print(f"NRHO orbit: period = {orbit.T_half*2 * TU_DAYS:.3f} days, resonance = {resonance}")
            state_vector = np.array([
                orbit.x0, orbit.z0,
                orbit.ydot0, orbit.T_half*2/2.0
            ])

            perilune, apolune = compute_orbit_extrema(state_vector, MU)
            perilune_km = perilune 
            apolune_km = apolune
            print(f"NRHO perlune = {perilune_km:.1f} km, apolune = {apolune_km:.1f} km")

        else:
            color = "lightgray"   # o "0.8"
            alpha = 0.8
            label = None

        ax.plot(
            pos[0],
            pos[1],
            pos[2],
            color=color,
            alpha=alpha,
            linewidth=1.5,
            label=label,
        )

    # Luna (CR3BP normalizado)
    moon = (1 - MU, 0, 0)
    ax.scatter(*moon, color="gray", s=80)
    ax.text(*moon, "Moon")
    
    _,  L2_X, _  = find_L_points(MU)
    L2 = (L2_X, 0, 0)
    ax.scatter(*L2, color="black", marker="*", s=120)
    ax.text(*L2, "L2")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    ax.legend()
    plt.tight_layout()
    plt.show()

plots(orbits)