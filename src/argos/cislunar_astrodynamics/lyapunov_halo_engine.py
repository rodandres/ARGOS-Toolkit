from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from numpy.linalg import svd

from scipy.integrate import solve_ivp
from scipy.integrate._ivp.ivp import OdeResult


from argos.cislunar_astrodynamics.cr3bp_engine import *

# Normalized constants for the Earth-Moon system
MU = 1.215e-2
LU_KM = 384400.0  # Distance from Earth to Moon in kilometers
TU_DAYS = 27.321661 / (2.0 * np.pi)  # Time unit in days (sideral lunar month)

@dataclass
class OrbitData:
    """
    Store computed data describing a periodic CR3BP orbit.

    The state-related quantities are expressed in normalized CR3BP units,
    while geometric distances and orbital periods are also provided in
    physical units where indicated.

    Attributes
    ----------
    x0 : float
        Initial dimensionless x-coordinate in the rotating frame.
    y0 : float
        Initial dimensionless y-coordinate in the rotating frame.
    z0 : float
        Initial dimensionless z-coordinate in the rotating frame.
    xdot0 : float
        Initial dimensionless x-velocity in the rotating frame.
    ydot0 : float
        Initial dimensionless y-velocity in the rotating frame.
    zdot0 : float
        Initial dimensionless z-velocity in the rotating frame.
    T_half : float
        Half-period in normalized CR3BP time units.
    CJ : float
        Jacobi constant of the orbit.
    amp_y : float
        In-plane y-amplitude in normalized distance units.
    amp_z : float
        Out-of-plane z-amplitude in normalized distance units.
    monodromy : np.ndarray, shape (6, 6)
        Monodromy matrix of the periodic orbit.
    trajectory : np.ndarray
        Computed orbit trajectory in normalized CR3BP state coordinates.
    u : np.ndarray
        Auxiliary orbit parameters associated with the continuation or
        correction process.
    tangent : np.ndarray or None
        Continuation tangent vector, if available.
    stability_indices : np.ndarray or None
        Stability indices computed from the orbit monodromy matrix, if
        available.
    perilune_km : float
        Minimum orbital distance from the relevant primary [km].
    apolune_km : float
        Maximum orbital distance from the relevant primary [km].
    period_days : float
        Full orbital period [days].
    """
    x0: float = 0.0
    y0: float = 0.0
    z0: float = 0.0
    xdot0: float = 0.0
    ydot0: float = 0.0
    zdot0: float = 0.0
    T_half: float = 0.0
    CJ: float = 0.0
    amp_y: float = 0.0
    amp_z: float = 0.0
    monodromy: np.ndarray = field(default_factory=lambda: np.zeros((6, 6)))
    trajectory: np.ndarray = field(default_factory=lambda: np.zeros((6, 1)))
    u: np.ndarray = field(default_factory=lambda: np.zeros(4))
    tangent: Optional[np.ndarray] = None
    stability_indices: Optional[np.ndarray] = field(default=None)
    perilune_km: float = 0.0
    apolune_km: float = 0.0
    period_days: float = 0.0

@dataclass
class PropagationResult:
    """
    Store the result of a CR3BP trajectory propagation.

    Attributes
    ----------
    state_final : np.ndarray
        Final propagated CR3BP state.
    Phi_final : np.ndarray, shape (6, 6)
        Final state transition matrix.
    trajectory : np.ndarray
        Propagated state trajectory.
    solution : OdeResult
        Complete integration result returned by ``solve_ivp``.
    """

    state_final: np.ndarray
    Phi_final: np.ndarray
    trajectory: np.ndarray
    solution: OdeResult

@dataclass
class ResidualResult:
    """
    Store the residual and sensitivity information from a periodic-orbit propagation.

    Attributes
    ----------
    F : np.ndarray
        Periodic-orbit residual vector.
    J : np.ndarray
        Jacobian or sensitivity matrix associated with the residual.
    state_cross : np.ndarray
        State evaluated at the trajectory crossing used to construct the
        residual.
    half_period : float
        Computed half-period in normalized CR3BP time units.
    Phi : np.ndarray, shape (6, 6)
        State transition matrix at the crossing.
    trajectory : np.ndarray
        Propagated trajectory used to evaluate the residual.
    solution : OdeResult
        Complete integration result returned by ``solve_ivp``.
    """

    F: np.ndarray
    J: np.ndarray
    state_cross: np.ndarray
    half_period: float
    Phi: np.ndarray
    trajectory: np.ndarray
    solution: OdeResult

@dataclass
class ContinuationStep:
    """
    Store the result of a pseudo-arclength continuation step.

    Attributes
    ----------
    predicted_state : np.ndarray
        State predicted by the continuation predictor.
    corrected_state : np.ndarray
        State obtained after applying the correction procedure.
    tangent_vector : np.ndarray
        Continuation tangent vector used for the step.
    residual_norm : float
        Norm of the corrected periodic-orbit residual.
    iterations : int
        Number of correction iterations performed.
    converged : bool
        Indicates whether the correction procedure converged.
    """

    predicted_state: np.ndarray
    corrected_state: np.ndarray
    tangent_vector: np.ndarray
    residual_norm: float
    iterations: int
    converged: bool

def build_state_vector(
    x0: float,
    z0: float,
    ydot0: float,
    half_period: float,
) -> np.ndarray:
    """
    Build the state vector used by the periodic-orbit correction process.

    Parameters
    ----------
    x0 : float
        Initial dimensionless x-coordinate in the rotating frame.
    z0 : float
        Initial dimensionless z-coordinate in the rotating frame.
    ydot0 : float
        Initial dimensionless y-velocity in the rotating frame.
    half_period : float
        Half-period of the orbit in normalized CR3BP time units.

    Returns
    -------
    np.ndarray, shape (4,)
        Reduced orbit parameter vector ordered as
        ``[x0, z0, ydot0, half_period]``.
    """

    return np.array(
        [x0, z0, ydot0, half_period],
        dtype=float,
    )

def unpack_state_vector(
    state_vector: np.ndarray,
) -> tuple[float, float, float, float]:
    """
    Extract the orbit parameters from a reduced state vector.

    Parameters
    ----------
    state_vector : np.ndarray, shape (4,)
        Reduced orbit parameter vector ordered as
        ``[x0, z0, ydot0, half_period]``.

    Returns
    -------
    tuple of float
        Values ``(x0, z0, ydot0, half_period)`` extracted from the input
        vector.
    """

    return (
        state_vector[0],
        state_vector[1],
        state_vector[2],
        state_vector[3],
    )

def differential_corrector(
    x0: float,
    ydot0: float,
    period_guess: float,
    z0: float | None = None,
    mu: float = MU,
    tol: float = 1e-12,
    max_iter: int = 50,
    verbose: bool = False,
):
    """
    Correct a periodic orbit using a differential correction method.

    The correction iteratively adjusts the selected initial conditions and
    half-period to satisfy the periodic-orbit symmetry and crossing
    conditions.

    Parameters
    ----------
    ...
    
    Returns
    -------
    tuple of float or None
        Corrected orbit parameters ``(x0, ydot0, half_period, z0)`` if the
        correction converges successfully. Returns ``(None, None, None,
        None)`` if the correction does not converge.

    Raises
    ------
    ...
        Document only exceptions that are explicitly raised by the
        implementation.    

    Notes
    -----
    Initial conditions are

        [x0, 0, z0, 0, ydot0, 0]

    with z0 = 0 for Lyapunov orbits.

    The correction is performed at the first subsequent
    crossing of the symmetry plane y = 0.

    Lyapunov constraint
    -------------------
        xdot(T/2) = 0

    Halo constraints
    ----------------
        xdot(T/2) = 0
        zdot(T/2) = 0

    The Jacobian includes the correction associated with the
    variable crossing time:

        d(v_f)/du =
            Phi[v,u]
            - a_v * Phi[y,u] / ydot_f

    The Newton correction uses the minimum-norm solution:

        Δu = -Jᵀ (J Jᵀ)^(-1) F
    """

    # ==========================================================
    # Determine orbit type
    # ==========================================================

    if z0 is None:

        # ------------------------------------------------------
        # Lyapunov
        # ------------------------------------------------------

        z0_is_free_variable = False
        n_constraints = 1

        z0 = 0.0        

        free_variable_indices = [0, 4]

    else:

        # ------------------------------------------------------
        # Halo
        # ------------------------------------------------------

        z0_is_free_variable = True
        n_constraints = 2

        free_variable_indices = [0, 2, 4]
    

    n_variables = len(free_variable_indices)

    # ==========================================================
    # Initial information
    # ==========================================================

    if verbose:

        orbit_type = "Halo" if z0_is_free_variable else "Lyapunov"

        print("\n" + "=" * 70)
        print(f"Differential Corrector: {orbit_type}")
        print("=" * 70)

        print(
            f"Initial guess: "
            f"x0 = {x0:.12f}, "
            f"z0 = {z0:.12f}, "
            f"ydot0 = {ydot0:.12f}, "
            f"T_guess = {period_guess:.12f}"
        )

    # ==========================================================
    # Integration parameters & config
    # ==========================================================

    half_period_guard = 0.05 * period_guess

    max_propagation_time = 3.0 * period_guess    

    def positive_y_crossing(t, state_stm, mu):

        if t > half_period_guard:

            return state_stm[1]

        return 1.0

    def negative_y_crossing(t, state_stm, mu):

        if t > half_period_guard:

            return state_stm[1]

        return -1.0

    positive_y_crossing.terminal = True
    positive_y_crossing.direction = +1

    negative_y_crossing.terminal = True
    negative_y_crossing.direction = -1

    # ==========================================================
    # Newton iterations
    # ==========================================================

    for iteration in range(max_iter):        

        initial_state = np.zeros(42)

        initial_state[:6] = [
            x0,
            0.0,
            z0,
            0.0,
            ydot0,
            0.0,
        ]

        initial_state[6:] = np.eye(6).flatten()        

        sol = solve_ivp(
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

        positive_crossing_time = (
            sol.t_events[0][0]
            if len(sol.t_events[0]) > 0
            else np.inf
        )

        negative_crossing_time = (
            sol.t_events[1][0]
            if len(sol.t_events[1]) > 0
            else np.inf
        )        

        if (
            np.isinf(positive_crossing_time)
            and np.isinf(negative_crossing_time)
        ):

            if verbose:

                print(
                    f"Iteration {iteration}: "
                    "no y = 0 crossing found."
                )

            return None, None, None, None

        # ======================================================
        # Select earliest crossing
        # ======================================================

        if positive_crossing_time < negative_crossing_time:

            half_period = positive_crossing_time

            state_at_half_period = sol.y_events[0][0]

        else:

            half_period = negative_crossing_time

            state_at_half_period = sol.y_events[1][0]

        # ======================================================
        # State and STM at half period
        # ======================================================

        STM_at_half_period = (
            state_at_half_period[6:]
            .reshape(6, 6)
        )

        xdot_at_half_period = state_at_half_period[3]

        ydot_at_half_period = state_at_half_period[4]

        zdot_at_half_period = state_at_half_period[5]

        # ======================================================
        # Dynamics at half period
        # ======================================================

        state_derivative = cr3bp(
            half_period,
            state_at_half_period[:6],
            mu,
        )

        xddot_at_half_period = state_derivative[3]

        zddot_at_half_period = state_derivative[5]        

        if verbose:

            print(f"\nIteration {iteration:2d}")
            print(f"  t_half  = {half_period:+.12e}")
            print(f"  xdot_f  = {xdot_at_half_period:+.12e}")
            print(f"  ydot_f  = {ydot_at_half_period:+.12e}")

            if z0_is_free_variable:
                print(f"  zdot_f  = {zdot_at_half_period:+.12e}")

            print(f"  xddot_f = {xddot_at_half_period:+.12e}")

            if z0_is_free_variable:
                print(f"  zddot_f = {zddot_at_half_period:+.12e}")

        # ======================================================
        # Convergence check
        # ======================================================

        if z0_is_free_variable:            
            converged = (abs(xdot_at_half_period) < tol and abs(zdot_at_half_period) < tol)
        else:
            converged = (abs(xdot_at_half_period) < tol)        

        if converged:

            if verbose:

                print(f"\nConverged in {iteration} iterations.")
                print(f"  x0         = {x0:.15f}")
                print(f"  z0         = {z0:.15f}")
                print(f"  ydot0      = {ydot0:.15f}")
                print(f"  half_period = {half_period:.15f}")
                print(f"  period      = {2.0 * half_period:.15f}")

            if z0_is_free_variable:
                return x0, ydot0, half_period, z0
            else:
                return x0, ydot0, half_period, None                

        # ======================================================
        # Crossing-time singularity check
        # ======================================================

        if abs(ydot_at_half_period) < 1e-14:
            if verbose:
                print("ydot ≈ 0 at symmetry-plane crossing.")
            return None, None, None, None

        jacobian = np.zeros(
            (n_constraints, n_variables)
        )

        for column, state_index in enumerate(free_variable_indices):

            jacobian[0, column] = (
                STM_at_half_period[
                    3,
                    state_index,
                ]
                -
                xddot_at_half_period
                *
                STM_at_half_period[
                    1,
                    state_index,
                ]
                /
                ydot_at_half_period
            )            

            if z0_is_free_variable:

                jacobian[1, column] = (
                    STM_at_half_period[
                        5,
                        state_index,
                    ]
                    -
                    zddot_at_half_period
                    *
                    STM_at_half_period[
                        1,
                        state_index,
                    ]
                    /
                    ydot_at_half_period
                )

        # ======================================================
        # Residual vector
        # ======================================================

        if z0_is_free_variable:

            residual = np.array(
                [
                    xdot_at_half_period,
                    zdot_at_half_period,
                ]
            )

        else:            

            residual = np.array(
                [
                    xdot_at_half_period,
                ]
            )

        gram_matrix = (jacobian @ jacobian.T)        

        determinant = np.linalg.det(gram_matrix)

        if verbose:
            print("\nJacobian:")
            print(jacobian)
            print(f"det(J J^T) = {determinant:+.6e}")

        if (not np.isfinite(determinant) or abs(determinant) < 1e-30):
            if verbose:
                print("Degenerate Jacobian.")
            return None, None, None, None        

        correction = (
            -jacobian.T
            @ np.linalg.solve(
                gram_matrix,
                residual,
            )
        )

        correction_norm = np.linalg.norm(correction)

        if verbose:
            print(f"\nCorrection = {correction}")

            print(f"||Δ|| = {correction_norm:.6e}")

        if correction_norm > 0.05:
            correction *= (0.05 / correction_norm)

            if verbose:
                print("Newton step limited to 0.05.")

        # ======================================================
        # Update free variables
        # ======================================================

        if z0_is_free_variable:
            x0 += correction[0]
            z0 += correction[1]
            ydot0 += correction[2]

        else:
            x0 += correction[0]
            ydot0 += correction[1]    

    if verbose:

        print("\nDifferential corrector did not converge.")

    return None, None, None, None

def compute_monodromy(
    x0: float,
    ydot0: float,
    period: float,
    mu: float,
):
    """
    Compute the monodromy matrix of a periodic CR3BP orbit.

    The monodromy matrix is obtained by propagating the state transition
    matrix over one complete orbital period.

    Parameters
    ----------
    x0 : float
        Initial dimensionless x-coordinate in the rotating frame.
    ydot0 : float
        Initial dimensionless y-velocity in the rotating frame.
    period : float
        Orbital period in normalized CR3BP time units.
    mu : float
        Dimensionless CR3BP mass parameter.

    Returns
    -------
    np.ndarray, shape (6, 6)
        Monodromy matrix evaluated after one complete orbital period.
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
    Analyze the eigenstructure and stability of a periodic-orbit monodromy matrix.

    The eigenvalues of the monodromy matrix are used to characterize the
    linear stability properties of the periodic orbit.

    Parameters
    ----------
    monodromy_matrix : np.ndarray, shape (6, 6)
        Monodromy matrix of the periodic orbit.
    verbose : bool, optional
        If True, print the computed eigenvalues and stability information.

    Returns
    -------
    dict
        Dictionary containing the eigenvalues, eigenvectors, and stability
        information derived from the monodromy matrix.
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

def propagate_half_period(
    state_vector: np.ndarray,
    mu: float,
    *,
    atol: float = 1e-12,
    rtol: float = 1e-12,
) -> PropagationResult:
    """
    Propagate a CR3BP orbit over one half-period.

    The state transition matrix is propagated together with the spacecraft
    state to obtain the sensitivity information required by the orbit
    correction process.

    Parameters
    ----------
    ...
    
    Returns
    -------
    PropagationResult
        Propagation result containing the final state, state transition
        matrix, trajectory, and integration solution.
    """

    x0, z0, ydot0, half_period = unpack_state_vector(state_vector)

    initial_state = np.array(
        [x0, 0.0, z0, 0.0, ydot0, 0.0]
    )

    initial_conditions = np.concatenate(
        [
            initial_state,
            np.eye(6).ravel(),
        ]
    )

    solution = solve_ivp(
        cr3bp_with_stm,
        (0.0, half_period),
        initial_conditions,
        args=(mu,),
        rtol=rtol,
        atol=atol,
        dense_output=False,
    )

    return PropagationResult(
        state_final=solution.y[:6, -1],
        Phi_final=solution.y[6:, -1].reshape(6, 6),
        trajectory=solution.y[:6],
        solution=solution,
    )

def propagate_full_period(
    state_vector: np.ndarray,
    mu: float,
    *,
    atol: float = 1e-12,
    rtol: float = 1e-12,
) -> PropagationResult:
    """
    Propagate a CR3BP orbit over one complete orbital period.

    The state transition matrix is propagated together with the spacecraft
    state to obtain the monodromy matrix of the periodic orbit.

    Parameters
    ----------
    ...
    
    Returns
    -------
    PropagationResult
        Propagation result containing the final state, state transition
        matrix, trajectory, and integration solution.
    """

    x0, z0, ydot0, half_period = unpack_state_vector(state_vector)

    initial_state = np.array(
        [x0, 0.0, z0, 0.0, ydot0, 0.0]
    )

    initial_conditions = np.concatenate(
        [
            initial_state,
            np.eye(6).ravel(),
        ]
    )

    solution = solve_ivp(
        cr3bp_with_stm,
        (0.0, 2.0 * half_period),
        initial_conditions,
        args=(mu,),
        rtol=rtol,
        atol=atol,
        dense_output=False,
    )

    return PropagationResult(
        state_final=solution.y[:6, -1],
        Phi_final=solution.y[6:, -1].reshape(6, 6),
        trajectory=solution.y[:6],
        solution=solution,
    )

def compute_residual(
    state_vector: np.ndarray,
    mu: float,
    *,
    atol: float = 1e-12,
    rtol: float = 1e-12,
) -> ResidualResult:
    """
    Compute the periodic-orbit residual and its sensitivity information.

    The orbit is propagated to the symmetry-plane crossing and the resulting
    state and state transition matrix are used to construct the residual
    required by the correction algorithm.

    Parameters
    ----------
    ...
    
    Returns
    -------
    ResidualResult
        Residual information including the residual vector, sensitivity
        matrix, crossing state, half-period, state transition matrix,
        trajectory, and integration solution.
    """

    propagation = propagate_half_period(
        state_vector,
        mu,
        atol=atol,
        rtol=rtol,
    )

    final_state = propagation.state_final
    state_transition_matrix = propagation.Phi_final

    ydot = final_state[4]

    state_derivative = cr3bp(
        0.0,
        final_state,
        mu,
    )

    xddot = state_derivative[3]
    zddot = state_derivative[5]

    residual_vector = np.array(
        [
            final_state[1],
            final_state[3],
            final_state[5],
        ]
    )

    jacobian = np.zeros((3, 4))

    free_variable_indices = [0, 2, 4]

    for column, stm_column in enumerate(free_variable_indices):
        jacobian[0, column] = state_transition_matrix[1, stm_column]
        jacobian[1, column] = state_transition_matrix[3, stm_column]
        jacobian[2, column] = state_transition_matrix[5, stm_column]

    jacobian[:, 3] = [
        ydot,
        xddot,
        zddot,
    ]

    return ResidualResult(
        F=residual_vector,
        J=jacobian,
        state_cross=final_state,
        half_period=state_vector[3],
        Phi=state_transition_matrix,
        trajectory=propagation.trajectory,
        solution=propagation.solution,
    )

def newton_minimum_norm(
    initial_state_vector: np.ndarray,
    mu: float,
    *,
    tol: float = 1e-12,
    max_iter: int = 30,
    verbose: bool = True,
) -> np.ndarray:
    """
    Compute a minimum-norm Newton correction for a nonlinear system.

    The correction is obtained using the singular value decomposition of the
    system Jacobian.

    Parameters
    ----------
    ...
    
    Returns
    -------
    np.ndarray
        Minimum-norm correction vector.
    """

    state_vector = initial_state_vector.copy()

    for iteration in range(max_iter):

        residual = compute_residual(state_vector, mu)

        residual_norm = np.linalg.norm(residual.F)

        if verbose:
            print(
                f"[Seed] Iteration {iteration:2d} | "
                f"||F|| = {residual_norm:.3e}"
            )

        if residual_norm < tol:
            return state_vector

        correction = -np.linalg.pinv(residual.J) @ residual.F

        state_vector += correction

    raise RuntimeError(
        "Initial seed failed to converge in newton_minimum_norm(). "
        "Check the initial guess."
    )

def compute_stability_indices(
    monodromy_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute stability indices from a periodic-orbit monodromy matrix.

    Parameters
    ----------
    monodromy_matrix : np.ndarray, shape (6, 6)
        Monodromy matrix of the periodic orbit.

    Returns
    -------
    tuple of np.ndarray
        Stability indices and the corresponding eigenvalues of the
        monodromy matrix.
    """

    eigenvalues = np.linalg.eigvals(monodromy_matrix)

    trivial_indices = np.argsort(
        np.abs(eigenvalues - 1.0)
    )[:2]

    remaining_indices = [
        index
        for index in range(6)
        if index not in trivial_indices
    ]

    reciprocal_pairs = []
    used_indices = set()

    for index in remaining_indices:

        if index in used_indices:
            continue

        eigenvalue = eigenvalues[index]

        closest_index = None
        minimum_error = np.inf

        for candidate in remaining_indices:

            if candidate == index or candidate in used_indices:
                continue

            error = abs(
                eigenvalues[candidate]
                - 1.0 / eigenvalue
            )

            if error < minimum_error:
                minimum_error = error
                closest_index = candidate

        reciprocal_pairs.append(
            (
                eigenvalue,
                eigenvalues[closest_index],
            )
        )

        used_indices.update(
            {
                index,
                closest_index,
            }
        )

    stability_indices = np.array(
        [
            0.5 * (eigenvalue + 1.0 / eigenvalue)
            for eigenvalue, _ in reciprocal_pairs
        ]
    )

    return stability_indices, eigenvalues

def compute_orbit_extrema(
    state_vector: np.ndarray,
    mu: float,
    *,
    n_samples: int = 1500,
) -> tuple[float, float]:
    """
    Compute the minimum and maximum orbital distances.

    The normalized CR3BP trajectory is converted to physical distance using
    the configured characteristic length unit.

    Parameters
    ----------
    ...
    
    Returns
    -------
    tuple of float
        Minimum and maximum orbital distances from the relevant primary
        ``(perilune_km, apolune_km)`` [km].
    """

    x0, z0, ydot0, half_period = unpack_state_vector(state_vector)

    initial_state = np.array(
        [
            x0,
            0.0,
            z0,
            0.0,
            ydot0,
            0.0,
        ]
    )

    solution = solve_ivp(
        cr3bp,
        (0.0, 2.0 * half_period),
        initial_state,
        args=(mu,),
        rtol=1e-11,
        atol=1e-11,
        dense_output=True,
    )

    time_samples = np.linspace(
        0.0,
        2.0 * half_period,
        n_samples,
    )

    trajectory = solution.sol(time_samples)

    moon_position = np.array(
        [
            1.0 - mu,
            0.0,
            0.0,
        ]
    )

    lunar_distances = (
        np.linalg.norm(
            trajectory[:3].T - moon_position,
            axis=1,
        )
        * LU_KM
    )

    perilune_distance = float(np.min(lunar_distances))
    apolune_distance = float(np.max(lunar_distances))

    return perilune_distance, apolune_distance

def compute_initial_tangent(
    jacobian: np.ndarray,
    orientation_hint: Optional[float] = None,
) -> np.ndarray:
    """
    Compute the initial tangent vector for pseudo-arclength continuation.

    Parameters
    ----------
    ...
    
    Returns
    -------
    np.ndarray
        Initial normalized tangent vector in the continuation parameter
        space.
    """

    _, _, right_singular_vectors = svd(jacobian)

    tangent = right_singular_vectors[-1].copy()
    tangent /= np.linalg.norm(tangent)

    if (
        orientation_hint is not None
        and np.sign(tangent[1]) != np.sign(orientation_hint)
    ):
        tangent *= -1.0

    return tangent

def update_tangent(
    jacobian: np.ndarray,
    previous_tangent: np.ndarray,
) -> np.ndarray:
    """
    Update the continuation tangent vector between neighboring solutions.

    Parameters
    ----------
    ...
    
    Returns
    -------
    np.ndarray
        Updated normalized tangent vector.
    """

    bordered_system = np.zeros((4, 4))
    bordered_system[:3, :] = jacobian
    bordered_system[3, :] = previous_tangent

    right_hand_side = np.zeros(4)
    right_hand_side[3] = 1.0

    tangent = np.linalg.solve(
        bordered_system,
        right_hand_side,
    )

    tangent /= np.linalg.norm(tangent)

    if np.dot(tangent, previous_tangent) < 0.0:
        tangent *= -1.0

    return tangent

def predictor(
    state_vector: np.ndarray,
    tangent_vector: np.ndarray,
    continuation_step: float,
) -> np.ndarray:
    """
    Predict the next solution along a continuation branch.

    Parameters
    ----------
    ...
    
    Returns
    -------
    np.ndarray
        Predicted state for the next continuation step.
    """
    return state_vector + continuation_step * tangent_vector

def compute_augmented_system(
    state_vector: np.ndarray,
    predicted_state: np.ndarray,
    tangent_vector: np.ndarray,
    mu: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the augmented system used by pseudo-arclength continuation.

    The augmented system combines the periodic-orbit residual with the
    pseudo-arclength constraint.

    Parameters
    ----------
    ...
    
    Returns
    -------
    tuple of np.ndarray
        Augmented residual vector and augmented Jacobian matrix.
    """
    residual = compute_residual(state_vector, mu)

    augmented_residual = np.zeros(4)
    augmented_residual[:3] = residual.F
    augmented_residual[3] = np.dot(
        tangent_vector,
        state_vector - predicted_state,
    )

    augmented_jacobian = np.zeros((4, 4))
    augmented_jacobian[:3, :] = residual.J
    augmented_jacobian[3, :] = tangent_vector

    return augmented_residual, augmented_jacobian

def pseudo_arclength_corrector(
    predicted_state: np.ndarray,
    tangent_vector: np.ndarray,
    mu: float,
    *,
    tol: float = 1e-11,
    max_iter: int = 15,
    verbose: bool = False,
) -> ContinuationStep:
    """
    Correct a predicted solution using pseudo-arclength continuation.

    Parameters
    ----------
    ...
    
    Returns
    -------
    ContinuationStep
        Result of the continuation correction, including the predicted and
        corrected states, tangent vector, residual norm, iteration count,
        and convergence status.
    """
    corrected_state = predicted_state.copy()

    for iteration in range(max_iter):

        augmented_residual, augmented_jacobian = compute_augmented_system(
            corrected_state,
            predicted_state,
            tangent_vector,
            mu,
        )

        residual_norm = np.linalg.norm(augmented_residual)

        if verbose:
            print(
                f"  Newton {iteration:2d} |G| = {residual_norm:.3e}"
            )

        if residual_norm < tol:

            residual = compute_residual(corrected_state, mu)
            updated_tangent = update_tangent(
                residual.J,
                tangent_vector,
            )

            return ContinuationStep(
                predicted_state=predicted_state,
                corrected_state=corrected_state,
                tangent_vector=updated_tangent,
                residual_norm=residual_norm,
                iterations=iteration,
                converged=True,
            )

        correction = np.linalg.solve(
            augmented_jacobian,
            -augmented_residual,
        )

        corrected_state += correction

    return ContinuationStep(
        predicted_state=predicted_state,
        corrected_state=corrected_state,
        tangent_vector=tangent_vector,
        residual_norm=residual_norm,
        iterations=max_iter,
        converged=False,
    )

def compute_lyapunov_family(
    x0_start: float,
    ydot0_start: float,
    initial_period: float,
    mu: float,
    number_of_orbits: int = 80,
    x0_step: float = 2e-3,
    stop_at_bifurcation: bool = True,
    verbose: bool = True,
) -> tuple[list[OrbitData], OrbitData | None]:
    """
    Compute a family of planar Lyapunov periodic orbits.

    The family is generated using pseudo-arclength continuation, starting
    from an initial corrected orbit and progressively computing neighboring
    solutions.

    Parameters
    ----------
    ...
    
    Returns
    -------
    tuple
        A tuple ``(family, bifurcation_orbit)`` where ``family`` is a list
        of ``OrbitData`` objects containing the computed Lyapunov orbits,
        and ``bifurcation_orbit`` is an ``OrbitData`` object identifying a
        detected bifurcation orbit, or ``None`` if no bifurcation is found.
    """

    family = []
    bifurcation_orbit = None

    current_x0 = x0_start
    current_ydot0 = ydot0_start
    current_half_period = initial_period / 2.0

    previous_beta = None

    _, _, _ = find_L_points(mu)
    moon_x = 1.0 - mu

    if verbose:
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

        if verbose:
            print(
                f"  {orbit_index + 1:>4}  "
                f"{current_x0:>12.8f}  "
                f"{current_ydot0:>12.8f}  "
                f"{full_period:>10.5f}  "
                f"{y_amplitude:>8.5f}  "
                f"{beta_z_deg:>10.4f}  "
                f"{distance_to_unity:>10.6f}"
            )

        orbit_data = OrbitData(
            x0=current_x0,
            ydot0=current_ydot0,
            T_half=current_half_period,
            trajectory=solution.y,
            monodromy=monodromy_matrix,            
        )

        family.append(orbit_data)

        # ==========================================================
        # Bifurcation detection
        # ==========================================================
        if (distance_to_unity < 0.05 and bifurcation_orbit is None):
            if verbose:
                print(
                    f"\n  *** BIFURCATION DETECTED (distance criterion): "
                    f"orbit {orbit_index + 1}  "
                    f"distance = {distance_to_unity:.6f} ***\n"
                )
            bifurcation_orbit = orbit_data
        if (previous_beta is not None and bifurcation_orbit is None):
            if previous_beta * beta_z_deg < 0.0:
                if verbose:
                    print(
                        f"\n  *** BIFURCATION DETECTED (β sign change): "
                        f"orbit {orbit_index + 1}  "
                        f"β: {previous_beta:.3f}° → {beta_z_deg:.3f}° ***\n"
                    )
                bifurcation_orbit = orbit_data

        previous_beta = beta_z_deg

        if stop_at_bifurcation and bifurcation_orbit is not None:
            if verbose: print("Stopping continuation at the bifurcation point.")
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
        if verbose:
            print(f"Minimum distance to the Moon: {minimum_moon_distance:.6f}")

        if minimum_moon_distance < 0.02:
            if verbose:
                print(
                f"Orbit {orbit_index + 1} reached the lunar vicinity. "
                f"Stopping continuation."
            )
            break

        if y_amplitude > 0.3:
            if verbose:
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

        corrected_x0, corrected_ydot0, corrected_half_period, _ = (
            differential_corrector(
                x0=predicted_x0,
                ydot0=predicted_ydot0,
                mu=mu,
                period_guess=full_period,
                verbose=verbose,
            )
        )

        if corrected_x0 is None:
            if verbose:
                print(
                    f"Differential corrector failed at orbit "
                    f"{orbit_index + 2}. Stopping continuation."
                )
            break

        current_x0 = corrected_x0
        current_ydot0 = corrected_ydot0
        current_half_period = corrected_half_period

    if verbose: print(f"{'─' * 60}")

    return family, bifurcation_orbit

def build_halo_seed(
    bifurcation_orbit: OrbitData,
    vertical_perturbation: float = 1e-3,
    verbose: bool = True,
) -> tuple[float, float, float]:
    """
    Construct an initial seed for a halo orbit.

    The seed is generated by perturbing a Lyapunov orbit using the vertical
    component of the corresponding linearized eigenvector.

    Parameters
    ----------
    ...
    
    Returns
    -------
    tuple of float
        Initial halo-orbit parameters ``(x0, z0, ydot0)`` in normalized
        CR3BP units.
    """

    monodromy = bifurcation_orbit.monodromy

    # ==========================================================
    # Validate monodromy matrix
    # ==========================================================

    if monodromy is None or monodromy.shape != (6, 6):

        if verbose:
            print(
                "[Warning] Invalid monodromy matrix at "
                "the bifurcation point."
            )

        return (
            bifurcation_orbit.x0,
            vertical_perturbation,
            bifurcation_orbit.ydot0,
        )

    # ==========================================================
    # Compute eigenvalues/eigenvectors
    # ==========================================================

    eigenvalues, eigenvectors = np.linalg.eig(monodromy)

    # ==========================================================
    # Find the vertical mode
    #
    # We look for an eigenvalue close to lambda = 1 whose
    # eigenvector has a significant out-of-plane component.
    # ==========================================================

    candidates = []

    for i, eigenvalue in enumerate(eigenvalues):

        eigenvector = eigenvectors[:, i]

        vertical_component = np.sqrt(
            np.abs(eigenvector[2]) ** 2
            + np.abs(eigenvector[5]) ** 2
        )

        eigenvalue_distance = np.abs(eigenvalue - 1.0)

        candidates.append(
            (
                eigenvalue_distance,
                -vertical_component,
                i,
            )
        )

    # Prefer eigenvalues close to 1, while requiring a
    # meaningful vertical component.
    candidates.sort()

    vertical_eigenvector = None
    selected_index = None

    for _, _, index in candidates:

        eigenvector = eigenvectors[:, index]

        vertical_norm = np.sqrt(
            np.abs(eigenvector[2]) ** 2
            + np.abs(eigenvector[5]) ** 2
        )

        if vertical_norm > 1e-10:

            vertical_eigenvector = eigenvector
            selected_index = index
            break

    if vertical_eigenvector is None:

        if verbose:
            print(
                "[Warning] Vertical-mode eigenvector not found "
                "at the bifurcation point."
            )

        return (
            bifurcation_orbit.x0,
            vertical_perturbation,
            bifurcation_orbit.ydot0,
        )

    # ==========================================================
    # Use real part
    # ==========================================================

    vertical_eigenvector = np.real(vertical_eigenvector)

    if verbose:

        print("\nVertical-mode eigenvector (real part):")

        labels = [
            "x",
            "y",
            "z",
            "xdot",
            "ydot",
            "zdot",
        ]

        for index, (label, value) in enumerate(
            zip(labels, vertical_eigenvector)
        ):

            print(
                f"  [{index}] "
                f"{label:4s} = {value:+.6f}"
            )

        print(
            f"\nSelected eigenvalue : "
            f"{eigenvalues[selected_index]:+.8f}"
        )

        print(
            f"Vertical position component : "
            f"{vertical_eigenvector[2]:+.6f}"
        )

        print(
            f"Vertical velocity component : "
            f"{vertical_eigenvector[5]:+.6f}"
        )

    # ==========================================================
    # Normalize using vertical components only
    # ==========================================================

    vertical_norm = np.sqrt(
        vertical_eigenvector[2] ** 2
        + vertical_eigenvector[5] ** 2
    )

    if vertical_norm < 1e-10:

        if verbose:
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
        bifurcation_orbit.x0
        + perturbation_scale
        * vertical_eigenvector[0]
    )

    z0_seed = (
        perturbation_scale
        * vertical_eigenvector[2]
    )

    ydot0_seed = (
        bifurcation_orbit.ydot0
        + perturbation_scale
        * vertical_eigenvector[4]
    )

    # ==========================================================
    # Diagnostics
    # ==========================================================

    if verbose:

        print(
            f"\nVertical perturbation = "
            f"{vertical_perturbation:.2e}"
        )

        print(
            f"Scaling factor        = "
            f"{perturbation_scale:.6f}"
        )

        print("\nHalo initial guess:")

        print(
            f"  x0    = {x0_seed:.10f} "
            f"(Δ = "
            f"{perturbation_scale * vertical_eigenvector[0]:+.4e})"
        )

        print(
            f"  z0    = {z0_seed:+.10f}"
        )

        print(
            f"  ydot0 = {ydot0_seed:.10f} "
            f"(Δ = "
            f"{perturbation_scale * vertical_eigenvector[4]:+.4e})"
        )

    return x0_seed, z0_seed, ydot0_seed

def build_halo_orbit(
    state_vector: np.ndarray,
    tangent_vector: Optional[np.ndarray],
    mu: float,
) -> OrbitData:
    """
    Build a corrected halo orbit and compute its associated orbit data.

    Parameters
    ----------
    ...
    
    Returns
    -------
    OrbitData
        Computed halo-orbit data, including the initial state, period,
        Jacobi constant, trajectory, monodromy matrix, and stability
        information.
    """
    residual = compute_residual(state_vector, mu)
    full_period = propagate_full_period(state_vector, mu)

    stability_index, _ = compute_stability_indices(full_period.Phi_final)

    perilune_km, apolune_km = compute_orbit_extrema(
        state_vector,
        mu,
    )

    initial_state = np.array(
        [
            state_vector[0],
            0.0,
            state_vector[1],
            0.0,
            state_vector[2],
            0.0,
        ]
    )

    jacobi = jacobi_constant(initial_state, mu)

    return OrbitData(
        x0=state_vector[0],
        z0=state_vector[1],
        ydot0=state_vector[2],
        T_half=state_vector[3],
        CJ=jacobi,
        amp_y=float(np.max(np.abs(residual.trajectory[1]))),
        amp_z=abs(state_vector[1]),
        monodromy=full_period.Phi_final,
        trajectory=residual.trajectory,
        u=state_vector.copy(),
        tangent=(
            None
            if tangent_vector is None
            else tangent_vector.copy()
        ),
        stability_indices=stability_index,
        perilune_km=perilune_km,
        apolune_km=apolune_km,
        period_days=2.0 * state_vector[3] * TU_DAYS,
    )

def compute_halo_family(
    initial_state: np.ndarray,
    mu: float,
    *,
    initial_step_size: float = 5e-4,
    minimum_step_size: float = 1e-8,
    maximum_step_size: float = 5e-3,
    n_steps: int = 60,
    max_steps: int = 20000,
    target_newton_iterations: int = 4,
    tol: float = 1e-11,
    orientation_hint: Optional[float] = None,
    target_perilune_km: Optional[float] = None,
    verbose: bool = True,
    verbose_every: int = 25,
) -> List[OrbitData]:
    """
    Compute a family of halo periodic orbits.

    The family is generated by continuing corrected halo-orbit solutions
    across the selected continuation parameter.

    Parameters
    ----------
    ...
    
    Returns
    -------
    list of OrbitData
        Computed halo orbits belonging to the family.
    """
    initial_state = newton_minimum_norm(
        initial_state,
        mu,
        tol=tol,
        verbose=verbose,
    )

    initial_residual = compute_residual(initial_state, mu)

    tangent_vector = compute_initial_tangent(
        initial_residual.J,
        orientation_hint=orientation_hint,
    )

    family = [
        build_halo_orbit(
            initial_state,
            tangent_vector,
            mu,
        )
    ]

    current_state = initial_state
    current_tangent = tangent_vector
    step_size = initial_step_size

    step_budget = (
        max_steps
        if target_perilune_km is not None
        else min(n_steps, max_steps)
    )

    accepted_steps = 0

    while accepted_steps < step_budget:

        converged = False
        continuation_step = None

        while step_size >= minimum_step_size:

            predicted_state = predictor(
                current_state,
                current_tangent,
                step_size,
            )

            continuation_step = pseudo_arclength_corrector(
                predicted_state,
                current_tangent,
                mu,
                tol=tol,
                verbose=False,
            )

            if continuation_step.converged:
                converged = True
                break

            step_size *= 0.5

        if not converged:
            if verbose:
                print(
                    f"[step {accepted_steps}] Continuation terminated: "
                    f"no convergence with step size "
                    f"{minimum_step_size:.1e}."
                )
            break

        orbit = build_halo_orbit(
            continuation_step.corrected_state,
            continuation_step.tangent_vector,
            mu,
        )

        family.append(orbit)

        near_target = (
            target_perilune_km is not None
            and orbit.perilune_km < 1.5 * target_perilune_km
        )

        if verbose and (
            accepted_steps % verbose_every == 0
            or near_target
        ):
            print(
                f"[step {accepted_steps:4d}] "
                f"iters={continuation_step.iterations:2d}  "
                f"ds={step_size:.2e}  "
                f"z0={orbit.z0:+.6f}  "
                f"T={orbit.period_days:6.2f} d  "
                f"r_p={orbit.perilune_km:8.1f} km  "
                f"r_a={orbit.apolune_km:8.1f} km  "
                f"nu={np.round(orbit.stability_indices.real, 3)}"
            )

        if continuation_step.iterations <= target_newton_iterations:
            step_size = min(
                1.3 * step_size,
                maximum_step_size,
            )
        elif continuation_step.iterations > target_newton_iterations + 3:
            step_size = max(
                0.7 * step_size,
                minimum_step_size,
            )

        current_state = continuation_step.corrected_state
        current_tangent = continuation_step.tangent_vector

        accepted_steps += 1

        if (
            target_perilune_km is not None
            and orbit.perilune_km <= target_perilune_km
        ):
            if verbose:
                print(
                    f"[step {accepted_steps:4d}] Target perilune reached "
                    f"(r_p={orbit.perilune_km:.1f} km <= "
                    f"{target_perilune_km:.1f} km)."
                )
            break

    return family