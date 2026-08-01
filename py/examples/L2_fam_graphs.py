import numpy as np
import matplotlib.pyplot as plt
from fractions import Fraction
from dataclasses import dataclass
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from py.modules.cislunar_astrodynamics.nrho_engine import build_state_vector, compute_orbit_extrema, continue_family
from py.modules.solvers.RK45 import rk45
from py.modules.cislunar_astrodynamics.cr3bp_engine import cr3bp

@dataclass
class OrbitData:
    x0: float
    y0: float
    z0: float
    xdot0: float
    ydot0: float
    zdot0: float

    period: float

MU = 1.215e-2

lunar_period = 27.321661
TU_DAYS = 27.321661 / (2.0 * np.pi)     # time unit unidad -> sideral lunar month (days)

def plots(orbits: list[OrbitData]):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    n = len(orbits)

    for i, orbit in enumerate(orbits):
        
        if not (i == 1 or i >= n - 2 or i % 5 == 0 or i == 2 or i == 9):
            continue

        state_vector = np.array([
            orbit.x0, orbit.y0, orbit.z0,
            orbit.xdot0, orbit.ydot0, orbit.zdot0
        ])

        period = orbit.period

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

        elif i == 9:
            color = "blue"
            alpha = 1.0
            label = "Birfurcation (Lyapunov to Halo)"        

        elif i == n - 2:
            ratio = orbit.period * TU_DAYS / lunar_period
            resonance = Fraction(ratio).limit_denominator(20)
            color = "tab:red"
            alpha = 1.0
            label = f"NRHO - Resonance:  {resonance.denominator}:{resonance.numerator}"

            print(f"NRHO orbit: period = {orbit.period * TU_DAYS:.3f} days, resonance = {resonance}")
            state_vector = np.array([
                orbit.x0, orbit.z0,
                orbit.ydot0, orbit.period/2.0
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




Orbits = []



from py.modules.cislunar_astrodynamics.cr3bp_engine import (
    eigenvalues_on_L_points,
    find_L_points,
)
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

from py.modules.cislunar_astrodynamics.lyapunov_halo_engine import compute_lyapunov_family, differential_corrector_lyapunov

# Correcting initial Lyapunov orbit
x0_corrected, ydot0_corrected, half_period = differential_corrector_lyapunov(
    x0_seed,
    ydot0_seed,
    MU,
    period_guess / 2.0,
)

initial_orbit = OrbitData(
    x0=x0_corrected,
    y0=0.0,
    z0=0.0,
    xdot0=0.0,
    ydot0=ydot0_corrected,
    zdot0=0.0,
    period=half_period * 2.0,
)
Orbits.append(initial_orbit)


# Compute lyapunov family
family, _ = compute_lyapunov_family(
        x0_start=x0_corrected,
        ydot0_start=ydot0_corrected,
        initial_period=half_period * 2.0,
        mu=MU,
        number_of_orbits=80,
        x0_step=3e-3,
        stop_at_bifurcation=True,
        verbose=False,
    )


for orbit in family:
    Orbits.append(
        OrbitData(
            x0=orbit["x0"],
            y0=0.0,
            z0=0.0,
            xdot0=0.0,
            ydot0=orbit["ydot0"],
            zdot0=0.0,
            period=orbit["full_period"]
        )
    )

# Last Lyapunov orbit
last_lyapunov_orbit = family[-1]

#initial_state = np.array([last_lyapunov_orbit["x0"], 0.0, 0.0, 0.0, last_lyapunov_orbit["ydot0"], 0.0])
initial_state = build_state_vector(
    1.18,
    -0.0037,
    -0.156,
    1.707,
)



# Compute halo family

family = continue_family(
    initial_state,
    MU,
    initial_step_size=5e-4,
    maximum_step_size=1.5e-2,
    minimum_step_size=1e-8,
    target_perilune_km=1800.0,
    max_steps=20000,
    orientation_hint=np.sign(initial_state[1]),
    verbose_every=50,
)

for orbit in family:
    Orbits.append(
        OrbitData(
            x0=orbit.x0,
            y0=0.0,
            z0=orbit.z0,
            xdot0=0.0,
            ydot0=orbit.ydot0,
            zdot0=0.0,
            period=orbit.T_half * 2.0
        )
    )

plots(Orbits)