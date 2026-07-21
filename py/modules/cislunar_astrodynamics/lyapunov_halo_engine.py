import numpy as np
from scipy.integrate import solve_ivp

from py.modules.cislunar_astrodynamics.cr3bp_engine import *


def differential_corrector_lyapunov(
    x0: float,
    ydot0: float,
    mu: float,
    period_guess: float,
    tol: float = 1e-12,
    max_iter: int = 50,
    verbose: bool = False,
):
    """
    Differential Corrector for planar Lyapunov periodic orbits in the CR3BP.

    Free Variables
    --------------
        x0
        ydot0

    Fixed Initial Conditions
    ------------------------
        y0     = 0
        z0     = 0
        xdot0  = 0
        zdot0  = 0

    Constraint
    ----------
        xdot(tf / 2) = 0

    Method
    ------
        Newton-Raphson differential correction using the State Transition
        Matrix (STM), accounting for the time shift of the symmetry-plane
        crossing.
    """


    half_period_guard = 0.05 * period_guess
    max_propagation_time = 3.0 * period_guess

    # ==========================================================
    # Symmetry-plane crossing events (y = 0)
    # ==========================================================

    def positive_y_crossing(t, state_stm, mu):
        return state_stm[1] if t > half_period_guard else 1.0

    def negative_y_crossing(t, state_stm, mu):
        return state_stm[1] if t > half_period_guard else -1.0

    positive_y_crossing.terminal = True
    positive_y_crossing.direction = +1

    negative_y_crossing.terminal = True
    negative_y_crossing.direction = -1

    # ==========================================================
    # Newton iterations
    # ==========================================================

    for iteration in range(max_iter):

        initial_state = np.zeros(42)
        initial_state[:6] = [x0, 0.0, 0.0, 0.0, ydot0, 0.0]
        initial_state[6:] = np.eye(6).flatten()

        sol = solve_ivp(
            cr3bp_with_stm,
            [0.0, max_propagation_time],
            initial_state,
            args=(mu,),
            events=[positive_y_crossing, negative_y_crossing],
            rtol=1e-12,
            atol=1e-13,
        )

        # ======================================================
        # Select first symmetry-plane crossing
        # ======================================================

        t_positive = (
            sol.t_events[0][0]
            if len(sol.t_events[0])
            else np.inf
        )

        t_negative = (
            sol.t_events[1][0]
            if len(sol.t_events[1])
            else np.inf
        )

        if np.isinf(t_positive) and np.isinf(t_negative):
            if verbose:
                print(f"Iteration {iteration}: no y = 0 crossing found.")

            return None, None, None

        if t_positive < t_negative:
            half_period = t_positive
            state_half = sol.y_events[0][0]
        else:
            half_period = t_negative
            state_half = sol.y_events[1][0]
        
        # ======================================================
        # State and STM at half period
        # ======================================================

        STM_half = state_half[6:].reshape(6, 6)

        xdot_half = state_half[3]
        ydot_half = state_half[4]

        state_derivative = cr3bp(
            half_period,
            state_half[:6],
            mu,
        )

        xddot_half = state_derivative[3]

        if verbose:
            print(
                f"Iteration {iteration:2d} | "
                f"t_half = {half_period:.8f} | "
                f"xdot_half = {xdot_half:.3e}"
            )

        # ======================================================
        # Convergence check
        # ======================================================

        if abs(xdot_half) < tol:

            if verbose:
                print(f"Converged in {iteration} iterations.")

            return x0, ydot0, half_period

        # ======================================================
        # Singularity check
        # ======================================================

        if abs(ydot_half) < 1e-14:

            if verbose:
                print("ydot ≈ 0 at symmetry-plane crossing.")

            return None, None, None

        # ======================================================
        # Analytical Jacobian using the STM
        # ======================================================

        dF_dx0 = (
            STM_half[3, 0]
            - xddot_half * STM_half[1, 0] / ydot_half
        )

        dF_dydot0 = (
            STM_half[3, 4]
            - xddot_half * STM_half[1, 4] / ydot_half
        )

        denominator = dF_dx0**2 + dF_dydot0**2

        if denominator < 1e-30:

            if verbose:
                print("Degenerate Jacobian.")

            return None, None, None

        # ======================================================
        # Minimum-norm Newton correction
        # ======================================================

        dx0 = -xdot_half * dF_dx0 / denominator
        dydot0 = -xdot_half * dF_dydot0 / denominator

        # ======================================================
        # Trust-region limiter
        # ======================================================

        step_norm = np.hypot(dx0, dydot0)

        if step_norm > 0.05:

            step_scale = 0.05 / step_norm

            dx0 *= step_scale
            dydot0 *= step_scale

        x0 += dx0
        ydot0 += dydot0

    if verbose:
        print("Differential corrector did not converge.")

    return None, None, None

# def differential_corrector_halo(x0: float, z0: float, ydot0: float, mu: float, T_guess: float,
#                    tol: float = 1e-12, max_iter: int = 50, verbose: bool = True):
#     """
#     Variables libres : x0, z0, ydot0   (y0=0, xdot0=0, zdot0=0)
#     Restricciones    : ẋ_f=0, ż_f=0   en cruce y=0
#     Sistema          : 2 ecuaciones, 3 incógnitas → mínima norma
#     """
#     T_min_guard = T_guess * 0.05
#     T_max       = 3.0 * T_guess

#     def y_up(t, s, mu): return s[1] if t > T_min_guard else  1.0
#     def y_dn(t, s, mu): return s[1] if t > T_min_guard else -1.0
#     y_up.terminal = True;  y_up.direction =  1
#     y_dn.terminal = True;  y_dn.direction = -1

#     if verbose:
#         print(f"\n  IC seed: x0={x0:.8f}  z0={z0:.8f}  ydot0={ydot0:.8f}")

#     for it in range(max_iter):
#         ic     = np.zeros(42)
#         ic[:6] = [x0, 0, z0, 0, ydot0, 0]
#         ic[6:] = np.eye(6).flatten()

#         sol = solve_ivp(cr3bp_with_stm, [0, T_max], ic, args=(mu,),
#                         events=[y_up, y_dn], rtol=1e-12, atol=1e-13)

#         t_up = sol.t_events[0][0] if len(sol.t_events[0]) else np.inf
#         t_dn = sol.t_events[1][0] if len(sol.t_events[1]) else np.inf

#         if np.isinf(t_up) and np.isinf(t_dn):
#             if verbose: print(f"  iter {it}: sin cruce y=0 en [0, {T_max:.3f}]")
#             return None, None, None, None

#         if t_up < t_dn:
#             t_half, sf = t_up, sol.y_events[0][0]
#         else:
#             t_half, sf = t_dn, sol.y_events[1][0]

#         STM_f   = sf[6:].reshape(6, 6)
#         xdot_f  = sf[3];  ydot_f = sf[4];  zdot_f = sf[5]
#         accel   = cr3bp(t_half, sf[:6], mu)
#         xddot_f = accel[3];  zddot_f = accel[5]

#         if verbose:
#             print(f"  iter {it:2d}: t½={t_half:.6f}  "
#                   f"ẋ_f={xdot_f:+.4e}  ż_f={zdot_f:+.4e}  "
#                   f"ẏ_f={ydot_f:+.4e}  "
#                   f"z_f={sf[2]:+.6f}")

#         if abs(xdot_f) < tol and abs(zdot_f) < tol:
#             if verbose: print(f"  ✓ Converged in {it} iterations")
#             return x0, z0, ydot0, t_half

#         if abs(ydot_f) < 1e-14:
#             if verbose: print(f"  iter {it}: ẏ_f≈0 — singularidad")
#             return None, None, None, None

#         # Jacobiano analítico 2×3
#         # Variables: u = [x0, z0, ydot0]  →  índices STM: [0, 2, 4]
#         cols = [0, 2, 4]
#         DF   = np.zeros((2, 3))
#         for j, c in enumerate(cols):
#             DF[0, j] = STM_f[3, c] - xddot_f * STM_f[1, c] / ydot_f
#             DF[1, j] = STM_f[5, c] - zddot_f * STM_f[1, c] / ydot_f

#         F      = np.array([xdot_f, zdot_f])
#         DFDFt  = DF @ DF.T

#         if verbose:
#             print(f"         DF = {DF}")
#             print(f"         det(DF·DFᵀ) = {np.linalg.det(DFDFt):.3e}")

#         if abs(np.linalg.det(DFDFt)) < 1e-30:
#             if verbose: print(f"  iter {it}: DF degenerado")
#             return None, None, None, None

#         delta_u = -DF.T @ np.linalg.solve(DFDFt, F)
#         norm    = np.linalg.norm(delta_u)

#         if verbose:
#             print(f"         δu = [{delta_u[0]:+.4e}, {delta_u[1]:+.4e}, "
#                   f"{delta_u[2]:+.4e}]  ‖δu‖={norm:.4e}")

#         if norm > 0.05:
#             delta_u *= 0.05 / norm
#             if verbose:
#                 print(f"         paso limitado a 0.05")

#         x0    += delta_u[0]
#         z0    += delta_u[1]
#         ydot0 += delta_u[2]

#     if verbose: print("  ✗ No converged")
#     return None, None, None, None

def differential_corrector_halo(
    x0: float,
    z0: float,
    ydot0: float,
    mu: float,
    period_guess: float,
    tol: float = 1e-12,
    max_iter: int = 50,
    verbose: bool = True,
):
    """
    Differential corrector for periodic Halo orbits in the CR3BP.

    Free variables
    --------------
        x0, z0, ydot0

    Fixed initial conditions
    ------------------------
        y0 = 0
        xdot0 = 0
        zdot0 = 0

    Constraints
    -----------
        xdot(tf/2) = 0
        zdot(tf/2) = 0

    Method
    ------
        Newton iteration with an analytical Jacobian obtained from the
        State Transition Matrix (STM), including the crossing-time correction.
        The underdetermined 2×3 system is solved using the minimum-norm solution.
    """

    half_period_guard = 0.05 * period_guess
    max_propagation_time = 3.0 * period_guess

    # ==========================================================
    # Symmetry-plane crossing events (y = 0)
    # ==========================================================

    def positive_y_crossing(t, state_stm, mu):
        return state_stm[1] if t > half_period_guard else 1.0

    def negative_y_crossing(t, state_stm, mu):
        return state_stm[1] if t > half_period_guard else -1.0

    positive_y_crossing.terminal = True
    positive_y_crossing.direction = +1

    negative_y_crossing.terminal = True
    negative_y_crossing.direction = -1

    if verbose:
        print(
            f"\nInitial guess: "
            f"x0 = {x0:.8f}, "
            f"z0 = {z0:.8f}, "
            f"ydot0 = {ydot0:.8f}"
        )

    # ==========================================================
    # Newton iterations
    # ==========================================================

    for iteration in range(max_iter):

        initial_state = np.zeros(42)
        initial_state[:6] = [x0, 0.0, z0, 0.0, ydot0, 0.0]
        initial_state[6:] = np.eye(6).flatten()

        solution = solve_ivp(
            cr3bp_with_stm,
            [0.0, max_propagation_time],
            initial_state,
            args=(mu,),
            events=[
                positive_y_crossing,
                negative_y_crossing,
            ],
            rtol=1e-12,
            atol=1e-13,
        )

        # ======================================================
        # Select the first symmetry-plane crossing
        # ======================================================

        positive_crossing_time = (
            solution.t_events[0][0]
            if len(solution.t_events[0])
            else np.inf
        )

        negative_crossing_time = (
            solution.t_events[1][0]
            if len(solution.t_events[1])
            else np.inf
        )

        if (
            np.isinf(positive_crossing_time)
            and np.isinf(negative_crossing_time)
        ):
            if verbose:
                print(
                    f"Iteration {iteration}: "
                    "No y = 0 crossing found."
                )

            return None, None, None, None

        if positive_crossing_time < negative_crossing_time:
            half_period = positive_crossing_time
            final_state = solution.y_events[0][0]
        else:
            half_period = negative_crossing_time
            final_state = solution.y_events[1][0]

        # ======================================================
        # State and STM at the crossing
        # ======================================================

        final_stm = final_state[6:].reshape(6, 6)

        xdot_final = final_state[3]
        ydot_final = final_state[4]
        zdot_final = final_state[5]

        state_derivative = cr3bp(
            half_period,
            final_state[:6],
            mu,
        )

        xddot_final = state_derivative[3]
        zddot_final = state_derivative[5]

        if verbose:
            print(
                f"Iteration {iteration:2d} | "
                f"t_half = {half_period:.8f} | "
                f"xdot_f = {xdot_final:+.3e} | "
                f"zdot_f = {zdot_final:+.3e}"
            )

        # ======================================================
        # Convergence
        # ======================================================

        if (
            abs(xdot_final) < tol
            and abs(zdot_final) < tol
        ):
            if verbose:
                print(f"Converged in {iteration} iterations.")

            return (
                x0,
                z0,
                ydot0,
                half_period,
            )

        # ======================================================
        # Avoid singularities
        # ======================================================

        if abs(ydot_final) < 1e-14:
            if verbose:
                print("ydot ≈ 0 at the crossing.")

            return None, None, None, None

        # ======================================================
        # Analytical Jacobian
        # ======================================================

        free_variable_indices = [0, 2, 4]

        jacobian = np.zeros((2, 3))

        for column, state_index in enumerate(free_variable_indices):

            jacobian[0, column] = (
                final_stm[3, state_index]
                - xddot_final
                * final_stm[1, state_index]
                / ydot_final
            )

            jacobian[1, column] = (
                final_stm[5, state_index]
                - zddot_final
                * final_stm[1, state_index]
                / ydot_final
            )

        residual = np.array(
            [
                xdot_final,
                zdot_final,
            ]
        )

        gram_matrix = jacobian @ jacobian.T

        if verbose:
            print(f"Jacobian:\n{jacobian}")
            print(
                f"det(JJᵀ) = "
                f"{np.linalg.det(gram_matrix):.3e}"
            )

        if abs(np.linalg.det(gram_matrix)) < 1e-30:
            if verbose:
                print("Degenerate Jacobian.")

            return None, None, None, None

        # ======================================================
        # Minimum-norm Newton step
        # ======================================================

        correction = (
            -jacobian.T
            @ np.linalg.solve(
                gram_matrix,
                residual,
            )
        )

        correction_norm = np.linalg.norm(correction)

        if verbose:
            print(
                f"Correction = {correction} | "
                f"||Δ|| = {correction_norm:.4e}"
            )

        # Simple trust-region

        if correction_norm > 0.05:

            correction *= 0.05 / correction_norm

            if verbose:
                print("Newton step limited to 0.05.")

        x0 += correction[0]
        z0 += correction[1]
        ydot0 += correction[2]

    if verbose:
        print("Failed to converge.")

    return None, None, None, None

def compute_monodromy(
    x0: float,
    ydot0: float,
    period: float,
    mu: float,
):
    """
    Propagate the State Transition Matrix (STM) over one full period
    to compute the monodromy matrix.

    Returns
    -------
        Monodromy matrix:
            M = Φ(T)
    """

    initial_state = np.zeros(42)
    initial_state[:6] = [x0, 0.0, 0.0, 0.0, ydot0, 0.0]
    initial_state[6:] = np.eye(6).flatten()

    solution = solve_ivp(
        cr3bp_with_stm,
        [0.0, period],
        initial_state,
        args=(mu,),
        rtol=1e-12,
        atol=1e-13,
    )

    monodromy_matrix = solution.y[6:, -1].reshape(6, 6)

    return monodromy_matrix

def analyze_monodromy(
    monodromy_matrix: np.ndarray,
    verbose: bool = False,
):
    """
    Analyze the eigenstructure of the monodromy matrix.

    The eigenvalues are classified into three groups:

        - Real pair:
            (λ, 1/λ), associated with planar instability.

        - Marginal pair:
            (e^{±iα}), associated with neutral planar motion.

        - Vertical pair:
            (e^{±iβ}), associated with the out-of-plane mode.

    Returns
    -------
        Dictionary containing:

            eigenvalues
            eigenvectors
            lambda_z
                Eigenvalue associated with the vertical mode.

            beta_z
                Vertical-mode angle β.
                β = 0 corresponds to the bifurcation point.

            distance_to_unity
                |λ_z - 1|.
                Zero indicates an exact bifurcation.

            vertical_eigenvector
                Eigenvector associated with λ_z.
    """

    eigenvalues, eigenvectors = np.linalg.eig(monodromy_matrix)

    # ==========================================================
    # Classify eigenvalues
    # ==========================================================

    unit_circle_eigenvalues = []
    real_eigenvalues = []

    for index, eigenvalue in enumerate(eigenvalues):

        if abs(abs(eigenvalue) - 1.0) < 0.1:
            unit_circle_eigenvalues.append((index, eigenvalue))
        else:
            real_eigenvalues.append((index, eigenvalue))

    # ==========================================================
    # Identify the vertical mode
    # ==========================================================

    minimum_distance_to_unity = np.inf
    vertical_mode_index = None

    for index, eigenvalue in unit_circle_eigenvalues:

        eigenvector = eigenvectors[:, index]

        vertical_component = (
            abs(eigenvector[2])
            + abs(eigenvector[5])
        )

        if vertical_component > 0.05:

            distance_to_unity = abs(eigenvalue - 1.0)

            if distance_to_unity < minimum_distance_to_unity:
                minimum_distance_to_unity = distance_to_unity
                vertical_mode_index = index

    # ==========================================================
    # Fallback: select the eigenvalue closest to λ = 1
    # ==========================================================

    if vertical_mode_index is None:

        for index, eigenvalue in unit_circle_eigenvalues:

            distance_to_unity = abs(eigenvalue - 1.0)

            if distance_to_unity < minimum_distance_to_unity:
                minimum_distance_to_unity = distance_to_unity
                vertical_mode_index = index

    analysis = {
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "distance_to_unity": minimum_distance_to_unity,
        "vertical_eigenvector": None,
        "beta_z": None,
        "lambda_z": None,
    }

    if vertical_mode_index is not None:

        lambda_z = eigenvalues[vertical_mode_index]

        analysis["lambda_z"] = lambda_z
        analysis["beta_z"] = np.angle(lambda_z)
        analysis["vertical_eigenvector"] = (
            eigenvectors[:, vertical_mode_index]
        )

    if verbose:

        print("\nMonodromy matrix eigenvalues:")

        for index, eigenvalue in enumerate(eigenvalues):

            marker = (
                " <-- vertical mode"
                if index == vertical_mode_index
                else ""
            )

            print(
                f"  [{index}] "
                f"{eigenvalue.real:+.6f} "
                f"{eigenvalue.imag:+.6f}i   "
                f"|λ| = {abs(eigenvalue):.6f}   "
                f"arg = {np.degrees(np.angle(eigenvalue)):+.2f}°"
                f"{marker}"
            )

        if analysis["lambda_z"] is not None:

            print(
                f"\nDistance to λ = 1 : "
                f"{minimum_distance_to_unity:.6f}"
            )

            print(
                f"β_z = "
                f"{np.degrees(analysis['beta_z']):.4f}°"
            )

    return analysis

def build_halo_seed(
    bifurcation_orbit: dict,
    vertical_perturbation: float = 1e-3,
    verbose: bool = True,
):
    """
    Construct an initial Halo-orbit seed from the Lyapunov-family
    bifurcation point.

    The initial condition is generated using the eigenvector associated
    with the vertical mode (λ ≈ 1) of the monodromy matrix.

    Returns
    -------
        (x0, z0, ydot0)
    """

    monodromy_data = bifurcation_orbit["monodromy_data"]
    vertical_eigenvector = monodromy_data["vertical_eigenvector"]

    if vertical_eigenvector is None:

        print(
            "[Warning] Vertical-mode eigenvector not found "
            "at the bifurcation point."
        )

        return (
            bifurcation_orbit["x0"],
            vertical_perturbation,
            bifurcation_orbit["ydot0"],
        )

    vertical_eigenvector = np.real(vertical_eigenvector)

    if verbose:

        print("\nVertical-mode eigenvector (real part):")

        labels = ["x", "y", "z", "xdot", "ydot", "zdot"]

        for index, (label, value) in enumerate(
            zip(labels, vertical_eigenvector)
        ):
            print(
                f"  [{index}] "
                f"{label:4s} = {value:+.6f}"
            )

        print(
            f"\nVertical position component : "
            f"{vertical_eigenvector[2]:+.6f}"
        )

        print(
            f"Vertical velocity component : "
            f"{vertical_eigenvector[5]:+.6f}"
        )

    # ==========================================================
    # Normalize using the vertical components only
    # ==========================================================

    vertical_norm = np.sqrt(
        vertical_eigenvector[2] ** 2
        + vertical_eigenvector[5] ** 2
    )

    if vertical_norm < 1e-10:

        print(
            "[Warning] Vertical eigenvector has negligible "
            "out-of-plane components."
        )

        vertical_norm = 1.0

    perturbation_scale = (
        vertical_perturbation / vertical_norm
    )

    # ==========================================================
    # Construct Halo initial guess
    # ==========================================================

    x0_seed = (
        bifurcation_orbit["x0"]
        + perturbation_scale * vertical_eigenvector[0]
    )

    z0_seed = (
        perturbation_scale * vertical_eigenvector[2]
    )

    ydot0_seed = (
        bifurcation_orbit["ydot0"]
        + perturbation_scale * vertical_eigenvector[4]
    )

    if verbose:

        print(f"\nVertical perturbation = {vertical_perturbation:.2e}")
        print(f"Scaling factor        = {perturbation_scale:.6f}")

        print("\nHalo initial guess:")

        print(
            f"  x0    = {x0_seed:.10f} "
            f"(Δ = {perturbation_scale * vertical_eigenvector[0]:+.4e})"
        )

        print(f"  z0    = {z0_seed:+.10f}")

        print(
            f"  ydot0 = {ydot0_seed:.10f} "
            f"(Δ = {perturbation_scale * vertical_eigenvector[4]:+.4e})"
        )

    return x0_seed, z0_seed, ydot0_seed

def compute_lyapunov_family(
    x0_start: float,
    ydot0_start: float,
    initial_period: float,
    mu: float,
    number_of_orbits: int = 80,
    x0_step: float = 2e-3,
    stop_at_bifurcation: bool = True,
    verbose: bool = True,
):
    """
    Computes a family of planar Lyapunov orbits while evaluating the
    monodromy matrix of each solution to detect the Halo bifurcation.

    Parameters
    ----------
    x0_start : float
        Initial x-coordinate of the first Lyapunov orbit.

    ydot0_start : float
        Initial y-velocity of the first Lyapunov orbit.

    initial_period : float
        Initial orbital period estimate.

    mu : float
        CR3BP mass parameter.

    number_of_orbits : int, optional
        Maximum number of family members to compute.

    x0_step : float, optional
        Predictor step applied to x0 during continuation.

    stop_at_bifurcation : bool, optional
        Stops the continuation when the bifurcation is detected.

    verbose : bool, optional
        Enables differential corrector output.

    Returns
    -------
    tuple[list[dict], dict | None]
        Computed Lyapunov family and the detected bifurcation orbit.
    """

    family = []
    bifurcation_orbit = None

    current_x0 = x0_start
    current_ydot0 = ydot0_start
    current_half_period = initial_period / 2.0

    previous_beta = None

    _, _, _ = find_L_points(mu)
    moon_x = 1.0 - mu

    print(f"\n{'─' * 60}")
    print(
        f"  {'Orb':>4}  {'x0':>12}  {'ydot0':>12}  {'T':>10}  "
        f"{'amp_y':>8}  {'β_z [deg]':>10}  {'dist(λ,1)':>10}"
    )
    print(f"{'─' * 60}")

    for orbit_index in range(number_of_orbits):

        full_period = 2.0 * current_half_period

        # ==========================================================
        # Propagate the complete Lyapunov orbit
        # ==========================================================
        solution = solve_ivp(
            cr3bp,
            [0.0, full_period],
            [current_x0, 0.0, 0.0, 0.0, current_ydot0, 0.0],
            args=(mu,),
            rtol=1e-12,
            atol=1e-13,
            max_step=full_period / 400.0,
        )

        y_amplitude = np.max(np.abs(solution.y[1]))

        # ==========================================================
        # Compute and analyze the monodromy matrix
        # ==========================================================
        monodromy_matrix = compute_monodromy(
            current_x0,
            current_ydot0,
            full_period,
            mu,
        )

        monodromy_data = analyze_monodromy(
            monodromy_matrix,
            verbose=False,
        )

        beta_z_deg = (
            np.degrees(monodromy_data["beta_z"])
            if monodromy_data["beta_z"] is not None
            else np.nan
        )

        distance_to_unity = monodromy_data["distance_to_unity"]

        print(
            f"  {orbit_index + 1:>4}  "
            f"{current_x0:>12.8f}  "
            f"{current_ydot0:>12.8f}  "
            f"{full_period:>10.5f}  "
            f"{y_amplitude:>8.5f}  "
            f"{beta_z_deg:>10.4f}  "
            f"{distance_to_unity:>10.6f}"
        )

        orbit_data = {
            "index": orbit_index,
            "x0": current_x0,
            "ydot0": current_ydot0,
            "half_period": current_half_period,
            "full_period": full_period,
            "trajectory": solution.y,
            "y_amplitude": y_amplitude,
            "monodromy_matrix": monodromy_matrix,
            "monodromy_data": monodromy_data,
            "beta_z": beta_z_deg,
            "distance_to_unity": distance_to_unity,
        }

        family.append(orbit_data)

        # ==========================================================
        # Bifurcation detection
        # ==========================================================
        if (
            distance_to_unity < 0.05
            and bifurcation_orbit is None
        ):
            print(
                f"\n  *** BIFURCATION DETECTED (distance criterion): "
                f"orbit {orbit_index + 1}  "
                f"distance = {distance_to_unity:.6f} ***\n"
            )
            bifurcation_orbit = orbit_data

        if (
            previous_beta is not None
            and bifurcation_orbit is None
        ):
            if previous_beta * beta_z_deg < 0.0:
                print(
                    f"\n  *** BIFURCATION DETECTED (β sign change): "
                    f"orbit {orbit_index + 1}  "
                    f"β: {previous_beta:.3f}° → {beta_z_deg:.3f}° ***\n"
                )
                bifurcation_orbit = orbit_data

        previous_beta = beta_z_deg

        if stop_at_bifurcation and bifurcation_orbit is not None:
            print("Stopping continuation at the bifurcation point.")
            break

        # ==========================================================
        # Family continuation limits
        # ==========================================================
        minimum_moon_distance = np.min(
            np.sqrt(
                (solution.y[0] - moon_x) ** 2 +
                solution.y[1] ** 2
            )
        )

        if minimum_moon_distance < 0.02:
            print(
                f"Orbit {orbit_index + 1} reached the lunar vicinity. "
                "Stopping continuation."
            )
            break

        if y_amplitude > 0.3:
            print(
                f"Maximum y-amplitude ({y_amplitude:.4f}) exceeded the "
                "continuation limit."
            )
            break

        # ==========================================================
        # Tangent predictor
        # ==========================================================
        initial_state = np.zeros(42)
        initial_state[:6] = [
            current_x0,
            0.0,
            0.0,
            0.0,
            current_ydot0,
            0.0,
        ]
        initial_state[6:] = np.eye(6).flatten()

        stm_solution = solve_ivp(
            cr3bp_with_stm,
            [0.0, current_half_period],
            initial_state,
            args=(mu,),
            rtol=1e-12,
            atol=1e-13,
        )

        half_period_stm = stm_solution.y[6:, -1].reshape(6, 6)

        if abs(half_period_stm[1, 4]) > 1e-14:
            predicted_ydot0 = (
                -half_period_stm[1, 0]
                * x0_step
                / half_period_stm[1, 4]
            )
        else:
            predicted_ydot0 = 0.0

        predicted_x0 = current_x0 + x0_step
        predicted_ydot0 += current_ydot0

        corrected_x0, corrected_ydot0, corrected_half_period = (
            differential_corrector_lyapunov(
                predicted_x0,
                predicted_ydot0,
                mu,
                full_period,
                verbose=verbose,
            )
        )

        if corrected_x0 is None:
            print(
                f"Differential corrector failed at orbit "
                f"{orbit_index + 2}. Stopping continuation."
            )
            break

        current_x0 = corrected_x0
        current_ydot0 = corrected_ydot0
        current_half_period = corrected_half_period

    print(f"{'─' * 60}")

    return family, bifurcation_orbit