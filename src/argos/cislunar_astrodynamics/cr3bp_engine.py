import numpy as np
from scipy.optimize import fsolve

def jacobi_constant(state: np.ndarray, mu: float) -> float:
    """
    Compute the Jacobi constant for a state in the CR3BP.

    The Jacobi constant is the conserved quantity associated with the
    Circular Restricted Three-Body Problem in the rotating frame.

    Parameters
    ----------
    state : np.ndarray, shape (6,)
        CR3BP state vector in the rotating frame, ordered as
        ``[x, y, z, x_dot, y_dot, z_dot]``.
    mu : float
        Dimensionless CR3BP mass parameter.

    Returns
    -------
    float
        Jacobi constant of the given state.
    """

    x, y, z, xd, yd, zd = state
    r1 = np.sqrt((x + mu)**2 + y**2 + z**2)
    r2 = np.sqrt((x - (1 - mu))**2 + y**2 + z**2)
    omega = 0.5 * (x**2 + y**2) + (1 - mu) / r1 + mu / r2
    v_square = xd**2 + yd**2 + zd**2
    return 2 * omega - v_square

def effective_potential(x: float, y: float, mu: float) -> float:
    """
    Compute the CR3BP effective potential.

    Parameters
    ----------
    x : float
        x-coordinate.
    y : float
        y-coordinate.
    mu : float
        CR3BP mass parameter.

    Returns
    -------
    float
        Effective potential.
    """
    r1 = np.sqrt((x + mu)**2 + y**2)
    r2 = np.sqrt((x - (1 - mu))**2 + y**2)

    omega = 1/2 * (x**2 + y**2) + ((1 - mu) / r1) + (mu / r2)
    return omega

def cr3bp(t: float, state: np.ndarray, mu: float) -> np.ndarray:
    """
    Compute the equations of motion for the Circular Restricted Three-Body Problem.

    The equations are expressed in the normalized rotating reference frame
    of the CR3BP.

    Parameters
    ----------
    t : float
        Current normalized time. The value is required by ODE solvers but
        does not explicitly appear in the autonomous equations.
    state : np.ndarray, shape (6,)
        CR3BP state vector in the rotating frame, ordered as
        ``[x, y, z, x_dot, y_dot, z_dot]``.
    mu : float
        Dimensionless CR3BP mass parameter.

    Returns
    -------
    np.ndarray, shape (6,)
        Time derivative of the state vector, ordered consistently with
        ``state``.
    """
    x, y, z, x_dot, y_dot, z_dot = state
    r1 = np.sqrt((x + mu)**2 + y**2 + z**2)
    r2 = np.sqrt((x - (1 - mu))**2 + y**2 + z**2)    

    omega_x = x - ((1 - mu) * (x + mu) / r1**3) - (mu * (x - (1 - mu)) / r2**3)
    omega_y = y - ((1 - mu) * y / r1**3) - (mu * y / r2**3)
    omega_z = - (1 - mu) * (z / r1**3) - (mu * (z / r2**3))
    
    x_ddot = 2 * y_dot + omega_x
    y_ddot = -2 * x_dot + omega_y
    z_ddot = omega_z

    return np.array([x_dot, y_dot, z_dot, x_ddot, y_ddot, z_ddot])

def cr3bp_with_stm(t: float, state_stm: np.ndarray, mu: float) -> np.ndarray:
    """
    Compute the CR3BP equations of motion together with the STM dynamics.

    The input contains the six-dimensional CR3BP state followed by the
    flattened 6x6 state transition matrix (STM). The STM dynamics are
    obtained from the CR3BP Jacobian.

    Parameters
    ----------
    t : float
        Current normalized time. The value is required by the ODE solver.
    state_stm : np.ndarray, shape (42,)
        Augmented state containing the CR3BP state followed by the
        flattened 6x6 state transition matrix.
    mu : float
        Dimensionless CR3BP mass parameter.

    Returns
    -------
    np.ndarray, shape (42,)
        Time derivative of the augmented state, containing the CR3BP state
        derivative followed by the flattened STM derivative.
    """
    state = state_stm[:6]
    stm = state_stm[6:].reshape(6, 6)

    state_dot = cr3bp(t, state, mu)
    jacobian = cr3bp_jacobian(*state[:3], mu) # jacobiano 6x6
    stm_dot = jacobian @ stm # Φ̇ = A·Φ

    return np.concatenate([state_dot, stm_dot.flatten()])

def lagrange_eq(x: float, mu: float) -> float:
    """
    Evaluate the scalar equilibrium equation for collinear Lagrange points.

    The equation is evaluated along the x-axis of the normalized rotating
    CR3BP frame and is used to determine the locations of L1, L2, and L3.

    Parameters
    ----------
    x : float
        Dimensionless x-coordinate along the rotating frame x-axis.
    mu : float
        Dimensionless CR3BP mass parameter.

    Returns
    -------
    float
        Value of the equilibrium equation at the specified x-coordinate.
    """
    r1 = abs(x + mu)
    r2 = abs(x - (1 - mu))
    return x - (1 - mu) * (x + mu) / r1**3 - mu * (x - (1 - mu)) / r2**3

def find_L_points(mu: float) -> tuple:
    """
    Compute the locations of the three collinear Lagrange points.

    The locations of L1, L2, and L3 are obtained by solving the
    corresponding scalar equilibrium equation numerically.

    Parameters
    ----------
    mu : float
        Dimensionless CR3BP mass parameter.

    Returns
    -------
    tuple of float
        Dimensionless x-coordinates of L1, L2, and L3, respectively.
    """
    L1 = fsolve(lagrange_eq, 0.5, args=(mu))[0]
    L2 = fsolve(lagrange_eq, 1.5, args=(mu))[0]
    L3 = fsolve(lagrange_eq, -1.5, args=(mu))[0]
    return L1, L2, L3

def eigenvalues_on_L_points(mu: float) -> tuple:
    """
    Compute the eigenvalues and eigenvectors of the CR3BP Jacobian at L1, L2, and L3.

    The linearized dynamics are evaluated at each collinear Lagrange point
    using the CR3BP Jacobian.

    Parameters
    ----------
    mu : float
        Dimensionless CR3BP mass parameter.

    Returns
    -------
    tuple
        Three ``(eigenvalues, eigenvectors)`` pairs corresponding to L1,
        L2, and L3, respectively. Each eigenvalue array has shape ``(6,)``
        and each eigenvector matrix has shape ``(6, 6)``.
    """
    L1, L2, L3 = find_L_points(mu)
    jacobian_L1 = cr3bp_jacobian(L1, 0, 0, mu)
    jacobian_L2 = cr3bp_jacobian(L2, 0, 0, mu)
    jacobian_L3 = cr3bp_jacobian(L3, 0, 0, mu)

    eigvals_L1, eigvecs_L1 = np.linalg.eig(jacobian_L1)
    eigvals_L2, eigvecs_L2 = np.linalg.eig(jacobian_L2)
    eigvals_L3, eigvecs_L3 = np.linalg.eig(jacobian_L3)

    return (
        (eigvals_L1, eigvecs_L1),
        (eigvals_L2, eigvecs_L2),
        (eigvals_L3, eigvecs_L3)
    )

def pseudopotential_second_derivatives(x: float, y: float, z: float, mu: float):
    """
    Compute the second derivatives of the CR3BP effective potential.

    Parameters
    ----------
    x : float
        Dimensionless x-coordinate in the rotating frame.
    y : float
        Dimensionless y-coordinate in the rotating frame.
    z : float
        Dimensionless z-coordinate in the rotating frame.
    mu : float
        Dimensionless CR3BP mass parameter.

    Returns
    -------
    tuple of float
        Second derivatives of the effective potential in the order
        ``(omega_xx, omega_yy, omega_zz, omega_xy, omega_xz, omega_yz)``.
    """
    r1 = np.sqrt((x + mu)**2 + y**2 + z**2)
    r2 = np.sqrt((x - (1 - mu))**2 + y**2 + z**2)    

    omega_xx = 1 \
        - ((1 - mu) * (1/r1**3 - 3*(x + mu)**2 / r1**5)) \
        - (mu * (1/r2**3 - 3*(x - (1 - mu))**2 / r2**5))

    omega_yy = 1 \
        - ((1 - mu) * (1/r1**3 - 3*y**2 / r1**5)) \
        - (mu * (1/r2**3 - 3*y**2 / r2**5))

    omega_zz = \
        - (1 - mu) * (1/r1**3 - 3*z**2 / r1**5) \
        - mu * (1/r2**3 - 3*z**2 / r2**5)

    omega_xy = 3 * (
        (1 - mu) * (x + mu) * y / r1**5 +
        mu * (x - (1 - mu)) * y / r2**5
    )

    omega_xz = 3 * (
        (1 - mu) * (x + mu) * z / r1**5 +
        mu * (x - (1 - mu)) * z / r2**5
    )

    omega_yz = 3 * (
        (1 - mu) * y * z / r1**5 +
        mu * y * z / r2**5
    )

    return omega_xx, omega_yy, omega_zz, omega_xy, omega_xz, omega_yz

def cr3bp_jacobian(x: float, y: float, z: float, mu: float) -> np.ndarray:
    """"
    Compute the Jacobian matrix of the CR3BP equations of motion.

    The Jacobian represents the linearized dynamics around the specified
    state position in the normalized rotating reference frame.

    Parameters
    ----------
    x : float
        Dimensionless x-coordinate in the rotating frame.
    y : float
        Dimensionless y-coordinate in the rotating frame.
    z : float
        Dimensionless z-coordinate in the rotating frame.
    mu : float
        Dimensionless CR3BP mass parameter.

    Returns
    -------
    np.ndarray, shape (6, 6)
        Jacobian matrix of the CR3BP equations of motion.
    """
    omega_xx, omega_yy, omega_zz, omega_xy, omega_xz, omega_yz = pseudopotential_second_derivatives(x, y, z, mu)        

    jacobian = np.zeros((6,6))

    jacobian[0, 3] = 1.0
    jacobian[1, 4] = 1.0
    jacobian[2, 5] = 1.0

    jacobian[3, 0] = omega_xx
    jacobian[3, 1] = omega_xy
    jacobian[3, 2] = omega_xz
    jacobian[3, 4] = 2.0

    jacobian[4, 0] = omega_xy
    jacobian[4, 1] = omega_yy
    jacobian[4, 2] = omega_yz
    jacobian[4, 3] = -2.0

    jacobian[5, 0] = omega_xz
    jacobian[5, 1] = omega_yz
    jacobian[5, 2] = omega_zz

    return jacobian