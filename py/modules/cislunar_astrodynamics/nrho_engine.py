from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from numpy.linalg import svd
from scipy.integrate import solve_ivp
from scipy.integrate._ivp.ivp import OdeResult
from scipy.optimize import fsolve

from py.modules.cislunar_astrodynamics.cr3bp_engine import *

LU_KM = 384400.0                        # distance unit - Earth-> Moon (km)
TU_DAYS = 27.321661 / (2.0 * np.pi)     # time unit unidad -> sideral lunar month (days)

def build_state_vector(
    x0: float,
    z0: float,
    ydot0: float,
    half_period: float,
) -> np.ndarray:
    """
    Builds the continuation state vector.

    Parameters
    ----------
    x0 : float
        Initial x-coordinate.

    z0 : float
        Initial z-coordinate.

    ydot0 : float
        Initial y-velocity.

    half_period : float
        Half-period of the periodic orbit.

    Returns
    -------
    np.ndarray
        Continuation state vector.
    """

    return np.array(
        [x0, z0, ydot0, half_period],
        dtype=float,
    )


def unpack_state_vector(
    state_vector: np.ndarray,
) -> tuple[float, float, float, float]:
    """
    Unpacks the continuation state vector.

    Parameters
    ----------
    state_vector : np.ndarray
        Continuation state vector.

    Returns
    -------
    tuple[float, float, float, float]
        x0, z0, ydot0, and half-period.
    """

    return (
        state_vector[0],
        state_vector[1],
        state_vector[2],
        state_vector[3],
    )

@dataclass
class PropagationResult:
    """
    Stores the results of a state and STM propagation.

    Attributes
    ----------
    state_final : np.ndarray
        Final spacecraft state at the end of the propagation.

    Phi_final : np.ndarray
        Final State Transition Matrix (STM).

    trajectory : np.ndarray
        State trajectory over the propagation interval.

    solution : OdeResult
        Raw integration result returned by the ODE solver.
    """

    state_final: np.ndarray
    Phi_final: np.ndarray
    trajectory: np.ndarray
    solution: OdeResult

def propagate_half_period(
    state_vector: np.ndarray,
    mu: float,
    *,
    atol: float = 1e-12,
    rtol: float = 1e-12,
) -> PropagationResult:
    """
    Propagates a periodic orbit and its State Transition Matrix (STM)
    over half of the orbital period.

    Parameters
    ----------
    state_vector : np.ndarray
        Continuation state vector
        [x0, z0, ydot0, half_period].

    mu : float
        CR3BP mass parameter.

    atol : float, optional
        Absolute integration tolerance.

    rtol : float, optional
        Relative integration tolerance.

    Returns
    -------
    PropagationResult
        Propagation results containing the final state, final STM,
        trajectory, and complete ODE solution.
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
    Propagates a periodic orbit and its State Transition Matrix (STM)
    over one complete orbital period.

    The monodromy matrix is obtained through direct integration over the
    full orbital period rather than by reconstructing it from the half-period
    STM using symmetry relations. Although this approximately doubles the
    propagation cost, it avoids potential sign errors in the symmetry
    transformation and is performed only once per converged orbit.

    Parameters
    ----------
    state_vector : np.ndarray
        Continuation state vector
        [x0, z0, ydot0, half_period].

    mu : float
        CR3BP mass parameter.

    atol : float, optional
        Absolute integration tolerance.

    rtol : float, optional
        Relative integration tolerance.

    Returns
    -------
    PropagationResult
        Propagation results containing the final state, final STM,
        trajectory, and complete ODE solution.
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

@dataclass
class ResidualResult:
    """
    Stores the residual vector and Jacobian associated with the
    differential correction process.

    Attributes
    ----------
    F : np.ndarray
        Residual vector evaluated at the half-period crossing.

    J : np.ndarray
        Jacobian matrix of the residual vector.

    state_cross : np.ndarray
        State vector at the symmetry-plane crossing.

    half_period : float
        Half-period corresponding to the evaluated orbit.

    Phi : np.ndarray
        State Transition Matrix (STM) at the half-period crossing.

    trajectory : np.ndarray
        State trajectory over the propagated half-period.

    solution : OdeResult
        Raw integration result returned by the ODE solver.
    """

    F: np.ndarray
    J: np.ndarray
    state_cross: np.ndarray
    half_period: float
    Phi: np.ndarray
    trajectory: np.ndarray
    solution: OdeResult


def compute_residual(
    state_vector: np.ndarray,
    mu: float,
    *,
    atol: float = 1e-12,
    rtol: float = 1e-12,
) -> ResidualResult:
    """
    Computes the residual vector and its Jacobian for the
    pseudo-arclength differential corrector.

    Parameters
    ----------
    state_vector : np.ndarray
        Continuation state vector
        [x0, z0, ydot0, half_period].

    mu : float
        CR3BP mass parameter.

    atol : float, optional
        Absolute integration tolerance.

    rtol : float, optional
        Relative integration tolerance.

    Returns
    -------
    ResidualResult
        Residual vector, Jacobian, propagated state,
        State Transition Matrix, trajectory, and solver output.
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
    Refines the initial continuation seed using a minimum-norm Newton
    iteration.

    The initial solution must satisfy F(u) ≈ 0 before computing the first
    continuation tangent. The tangent vector represents the local geometry
    of the solution manifold and is therefore meaningful only when
    evaluated at a converged solution.

    Since the correction problem consists of three equations and four
    unknowns, the system is underdetermined. Rather than fixing one
    variable, the Moore-Penrose pseudoinverse is used to compute the
    minimum-norm Newton step, yielding the closest solution to the initial
    guess.

    Parameters
    ----------
    initial_state_vector : np.ndarray
        Initial continuation state vector.

    mu : float
        CR3BP mass parameter.

    tol : float, optional
        Convergence tolerance on the residual norm.

    max_iter : int, optional
        Maximum number of Newton iterations.

    verbose : bool, optional
        Enables iteration logging.

    Returns
    -------
    np.ndarray
        Corrected continuation state vector.

    Raises
    ------
    RuntimeError
        If the Newton iteration fails to converge.
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
    Computes the Howell (1984) stability indices from the monodromy matrix.

    The two non-trivial reciprocal eigenvalue pairs are identified after
    removing the trivial pair located near λ = 1, which is associated with
    time invariance of the periodic orbit.

    The stability indices are defined as

        ν_i = 0.5 * (λ_i + 1 / λ_i)

    for each reciprocal eigenvalue pair.

    Parameters
    ----------
    monodromy_matrix : np.ndarray
        Monodromy matrix of the periodic orbit.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Stability indices and the complete set of eigenvalues.
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
    Computes the perilune and apolune distances of a periodic orbit.

    The orbit is propagated over one complete period without integrating
    the State Transition Matrix (STM), making this computation
    significantly less expensive than a full monodromy propagation.

    The minimum lunar distance provides a physically meaningful measure
    of the continuation progress toward the NRHO family.

    Parameters
    ----------
    state_vector : np.ndarray
        Continuation state vector
        [x0, z0, ydot0, half_period].

    mu : float
        CR3BP mass parameter.

    n_samples : int, optional
        Number of uniformly distributed samples used to evaluate the orbit.

    Returns
    -------
    tuple[float, float]
        Perilune and apolune distances, in kilometers.
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
    Computes the initial pseudo-arclength continuation tangent.

    The tangent is obtained as the null-space vector of the Jacobian using
    Singular Value Decomposition (SVD).

    Parameters
    ----------
    jacobian : np.ndarray
        Residual Jacobian evaluated at the converged solution.

    orientation_hint : float, optional
        Desired sign of the z-direction component used to enforce a
        consistent continuation direction.

    Returns
    -------
    np.ndarray
        Unit tangent vector.
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
    Updates the continuation tangent using Keller's bordered system
    formulation (Keller, 1977).

    Parameters
    ----------
    jacobian : np.ndarray
        Residual Jacobian at the current solution.

    previous_tangent : np.ndarray
        Tangent vector from the previous continuation step.

    Returns
    -------
    np.ndarray
        Updated unit tangent vector.
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
    Euler predictor for pseudo-arclength continuation.

    Parameters
    ----------
    state_vector : np.ndarray
        Current solution vector.

    tangent_vector : np.ndarray
        Unit tangent vector at the current solution.

    continuation_step : float
        Pseudo-arclength continuation step.

    Returns
    -------
    np.ndarray
        Predicted solution vector.
    """
    return state_vector + continuation_step * tangent_vector

@dataclass
class ContinuationStep:
    """
    Stores the result of a single pseudo-arclength continuation step.

    Parameters
    ----------
    predicted_state : np.ndarray
        State vector obtained from the Euler predictor.

    corrected_state : np.ndarray
        State vector after Newton correction.

    tangent_vector : np.ndarray
        Unit tangent vector at the corrected solution.

    residual_norm : float
        Euclidean norm of the residual after convergence.

    iterations : int
        Number of Newton iterations performed.

    converged : bool
        Indicates whether the corrector converged.
    """

    predicted_state: np.ndarray
    corrected_state: np.ndarray
    tangent_vector: np.ndarray
    residual_norm: float
    iterations: int
    converged: bool


def compute_augmented_system(
    state_vector: np.ndarray,
    predicted_state: np.ndarray,
    tangent_vector: np.ndarray,
    mu: float,
):
    """
    Assemble the augmented pseudo-arclength continuation system.

    Parameters
    ----------
    state_vector : np.ndarray
        Current solution vector.

    predicted_state : np.ndarray
        Euler predictor obtained from the previous continuation step.

    tangent_vector : np.ndarray
        Unit tangent vector defining the pseudo-arclength constraint.

    mu : float
        CR3BP mass parameter.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
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
    Newton corrector for pseudo-arclength continuation.

    Parameters
    ----------
    predicted_state : np.ndarray
        State vector obtained from the Euler predictor.

    tangent_vector : np.ndarray
        Unit tangent vector defining the pseudo-arclength constraint.

    mu : float
        CR3BP mass parameter.

    tol : float, optional
        Convergence tolerance for the augmented residual norm.

    max_iter : int, optional
        Maximum number of Newton iterations.

    verbose : bool, optional
        If True, prints iteration information.

    Returns
    -------
    ContinuationStep
        Result of the continuation step.
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

@dataclass
class HaloOrbit:
    """
    Stores the properties of a converged Halo orbit.

    Parameters
    ----------
    x0 : float
        Initial x-coordinate.

    z0 : float
        Initial z-coordinate.

    ydot0 : float
        Initial y-velocity.

    T_half : float
        Half-period of the orbit in normalized CR3BP units.

    CJ : float
        Jacobi constant.

    amp_y : float
        Maximum excursion along the y-axis.

    amp_z : float
        Maximum excursion along the z-axis.

    monodromy : np.ndarray
        Monodromy matrix computed over one full orbital period.

    trajectory : np.ndarray
        State history over one orbital period.

    u : np.ndarray
        Continuation state vector
        ``[x0, z0, ydot0, T_half]``.

    tangent : np.ndarray, optional
        Unit tangent vector associated with the continuation branch.

    stability_indices : np.ndarray, optional
        Howell stability indices computed from the monodromy matrix.

    perilune_km : float, optional
        Minimum distance to the Moon during the orbit, in kilometers.

    apolune_km : float, optional
        Maximum distance to the Moon during the orbit, in kilometers.

    period_days : float, optional
        Orbital period expressed in days.
    """

    x0: float
    z0: float
    ydot0: float
    T_half: float
    CJ: float
    amp_y: float
    amp_z: float
    monodromy: np.ndarray
    trajectory: np.ndarray
    u: np.ndarray
    tangent: Optional[np.ndarray] = None
    stability_indices: Optional[np.ndarray] = field(default=None)
    perilune_km: float = 0.0
    apolune_km: float = 0.0
    period_days: float = 0.0


def build_halo_orbit(
    state_vector: np.ndarray,
    tangent_vector: Optional[np.ndarray],
    mu: float,
) -> HaloOrbit:
    """
    Build a HaloOrbit object from a converged continuation state.

    Parameters
    ----------
    state_vector : np.ndarray
        Continuation state vector
        ``[x0, z0, ydot0, T_half]``.

    tangent_vector : np.ndarray, optional
        Unit tangent vector associated with the continuation branch.

    mu : float
        CR3BP mass parameter.

    Returns
    -------
    HaloOrbit
        Halo orbit containing the propagated trajectory, stability
        properties, and geometric characteristics.
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

    return HaloOrbit(
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

def continue_family(
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
) -> List[HaloOrbit]:
    """
    Generate a continuous family of Halo orbits using pseudo-arclength
    continuation.

    The algorithm consists of:

    1. Minimum-norm Newton correction of the initial seed.
    2. Euler predictor.
    3. Newton corrector applied to the augmented system.
    4. Keller tangent update.
    5. Adaptive continuation step-size control.

    The continuation may terminate after a fixed number of accepted steps
    or when a target perilune distance is reached.

    Parameters
    ----------
    initial_state : np.ndarray
        Initial continuation state vector
        ``[x0, z0, ydot0, T_half]``.

    mu : float
        CR3BP mass parameter.

    initial_step_size : float, optional
        Initial pseudo-arclength step size.

    minimum_step_size : float, optional
        Minimum allowable continuation step size.

    maximum_step_size : float, optional
        Maximum allowable continuation step size.

    n_steps : int, optional
        Maximum number of continuation steps when no target perilune is
        specified.

    max_steps : int, optional
        Absolute upper limit on continuation steps.

    target_newton_iterations : int, optional
        Desired number of Newton iterations used for adaptive step-size
        control.

    tol : float, optional
        Newton convergence tolerance.

    orientation_hint : float, optional
        Preferred orientation of the initial tangent vector.

    target_perilune_km : float, optional
        Target perilune distance in kilometers.

    verbose : bool, optional
        If True, prints continuation progress.

    verbose_every : int, optional
        Interval between progress messages.

    Returns
    -------
    List[HaloOrbit]
        Converged Halo orbit family.
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