# """
# Continuador pseudo-arclength para familias Halo / NRHO en el CR3BP (sistema Tierra-Luna).

# Referencias principales:
#     - Keller, H.B. (1977), "Numerical solution of bifurcation and nonlinear
#       eigenvalue problems", Applications of Bifurcation Theory.
#     - Howell, K.C. (1984), "Three-dimensional, periodic, 'halo' orbits",
#       Celestial Mechanics, 32(1).
#     - Doedel, E. et al., AUTO-07p manual (formulación estándar de continuación
#       pseudo-arclength con predictor de Euler/secante y bordering de Keller).
#     - Koon, W.S., Lo, M.W., Marsden, J.E., Ross, S.D., "Dynamical Systems, the
#       Three-Body Problem and Space Mission Design".
#     - Grebow, D. (2006), Zimovan-Spreen, E. (2017): familias Halo L2 y su
#       continuación hacia el régimen NRHO.

# Este archivo reemplaza y consolida los tres módulos originales (el script de
# continuación y los dos módulos de física duplicados). Ver el mensaje de
# análisis para el detalle de qué se corrigió y por qué.
# """

# import numpy as np

# from pathlib import Path
# import sys

# REPO_ROOT = Path(__file__).resolve().parents[3]
# if str(REPO_ROOT) not in sys.path:
#     sys.path.insert(0, str(REPO_ROOT))

# from py.modules.cislunar_astrodynamics.cr3bp_engine import find_L_points
# from py.modules.cislunar_astrodynamics.nrho_engine import build_state_vector, continue_family


# MU_EARTH_MOON = 1.215058560962404e-2




# # =============================================================================
# # 1. FÍSICA DEL CR3BP -- ÚNICA FUENTE DE VERDAD
# # =============================================================================
# #
# # En el código original, `propagate_half_period` integraba con
# # `cr3bp_with_stm` importado de `physics.astrodynamics`, mientras que
# # `compute_residual` evaluaba la dinámica con `cr3bp_equations` importado de
# # `orbit_determination.nrho` (un módulo distinto, con su propia definición
# # de la dinámica y su propio jacobiano). Aunque en este caso concreto ambas
# # implementaciones son algebraicamente equivalentes, depender de dos fuentes
# # distintas para la MISMA física es una fuente de errores silenciosos: basta
# # con que una de las dos difiera en un signo o en una convención para que la
# # STM integrada no sea la STM real de la dinámica usada en el residuo, y el
# # Newton dejaría de tener convergencia cuadrática sin previo aviso. Por eso
# # aquí queda una sola definición de la dinámica, su jacobiano y su STM.

# # =============================================================================
# # 2. VARIABLES LIBRES Y PROPAGACIÓN DE MEDIO PERIODO
# # =============================================================================
# #
# # u = [x0, z0, ydot0, T_half]. Se conserva esta parametrización: es la
# # estándar para halos simétricas respecto al plano xz (Howell 1984): el
# # cruce inicial en y=0 con velocidad perpendicular al plano (xdot0=zdot0=0)
# # y el cruce final vuelve a exigir y=0 con velocidad perpendicular.









# # =============================================================================
# # 9. EJECUCIÓN DE EJEMPLO
# # =============================================================================

# if __name__ == "__main__":
#     MU = MU_EARTH_MOON

#     L1, L2, L3 = find_L_points(MU)
#     print("L2 =", L2)

#     u0 = build_state_vector(1.18, -0.0037, -0.156, 1.707)

#     family = continue_family(
#         u0, MU,
#         initial_step_size=5e-4, maximum_step_size=1.5e-2, minimum_step_size=1e-8,
#         target_perilune_km=3400.0,
#         max_steps=20000,
#         orientation_hint=np.sign(u0[1]),
#         verbose_every=50,
#     )

#     last = family[-1]
#     print(f"\nFamilia generada: {len(family)} orbitas.")
#     print(
#         f"Ultima orbita: z0={last.z0:+.6f}  T={last.period_days:.3f} d  "
#         f"r_p={last.perilune_km:.1f} km  r_a={last.apolune_km:.1f} km  "
#         f"nu={np.round(last.stability_indices.real, 3)}"
#     )

"""
Pseudo-arclength continuation of Halo and NRHO families in the Earth-Moon CR3BP.

Primary references
------------------
- Keller, H.B. (1977), "Numerical Solution of Bifurcation and Nonlinear
  Eigenvalue Problems", Applications of Bifurcation Theory.
- Howell, K.C. (1984), "Three-Dimensional, Periodic, Halo Orbits",
  Celestial Mechanics, 32(1).
- Doedel, E. et al., AUTO-07p User Manual (standard pseudo-arclength
  continuation formulation using Euler prediction and Keller bordering).
- Koon, W.S., Lo, M.W., Marsden, J.E., Ross, S.D.,
  "Dynamical Systems, the Three-Body Problem and Space Mission Design".
- Grebow, D. (2006); Zimovan-Spreen, E. (2017), continuation of the
  L2 Halo family into the NRHO regime.

This script consolidates the continuation framework into a single,
self-consistent implementation.
"""

import numpy as np

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from py.modules.cislunar_astrodynamics.cr3bp_engine import find_L_points
from py.modules.cislunar_astrodynamics.nrho_engine import (
    build_state_vector,
    continue_family,
)

MU_EARTH_MOON = 1.215058560962404e-2


# =============================================================================
# Example
# =============================================================================

if __name__ == "__main__":

    MU = MU_EARTH_MOON

    L1, L2, L3 = find_L_points(MU)

    print(f"L2 = {L2}")

    initial_state = build_state_vector(
        1.18,
        -0.0037,
        -0.156,
        1.707,
    )

    family = continue_family(
        initial_state,
        MU,
        initial_step_size=5e-4,
        maximum_step_size=1.5e-2,
        minimum_step_size=1e-8,
        target_perilune_km=3400.0,
        max_steps=20000,
        orientation_hint=np.sign(initial_state[1]),
        verbose_every=50,
    )

    final_orbit = family[-1]

    print(f"\nGenerated family: {len(family)} orbits.")
    print(
        f"Final orbit: "
        f"z0 = {final_orbit.z0:+.6f}  "
        f"Period = {final_orbit.period_days:.3f} days  "
        f"Perilune = {final_orbit.perilune_km:.1f} km  "
        f"Apolune = {final_orbit.apolune_km:.1f} km  "
        f"Stability indices = {np.round(final_orbit.stability_indices.real, 3)}"
    )
    print(f"State vector: {final_orbit.u} normalized units")
    print(f"State vector: {final_orbit.u*384400.0} km")