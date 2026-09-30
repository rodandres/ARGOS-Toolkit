import numpy as np

def quaternion_dynamics(        
        t: float,
        state: np.ndarray,
        inertia_matrix: np.ndarray,
        inverse_inertia_matrix: np.ndarray,
        applied_torque: np.ndarray,
        disturbance_torque: np.ndarray,
    ):
        """
        Compute the spacecraft rotational equations of motion.

        The input state contains the attitude quaternion followed by angular
        velocity. The quaternion is normalized before computing its derivative.

        Parameters
        ----------
        t : float
            Current integration time [s].
        state : np.ndarray, shape (7,)
            Rotational state ordered as quaternion followed by angular velocity
            [rad/s].
        inertia_matrix : np.ndarray, shape (3, 3)
            Spacecraft inertia tensor [kg·m²].
        inverse_inertia_matrix : np.ndarray, shape (3, 3)
            Inverse spacecraft inertia tensor [kg⁻¹·m⁻²].
        applied_torque : np.ndarray, shape (3,)
            Applied control torque [N·m].
        disturbance_torque : np.ndarray, shape (3,)
            Environmental disturbance torque [N·m].

        Returns
        -------
        np.ndarray, shape (7,)
            Time derivative of the rotational state. The first four elements
            contain the quaternion derivative and the last three contain the
            angular acceleration [rad/s²].
        """       

        q = state[:4].copy()
        omega = state[4:].copy()

        # Normalize quaternion
        q /= np.linalg.norm(q)

        I = inertia_matrix
        I_inv = inverse_inertia_matrix


        omega_dot = I_inv @ (applied_torque + disturbance_torque - np.cross(omega, I @ omega))

        q_x, q_y, q_z, qw = q
        omega_x, omega_y, omega_z = omega
        q_dot = 0.5 * np.array([
            qw * omega_x + q_y * omega_z - q_z * omega_y,
            qw * omega_y + q_z * omega_x - q_x * omega_z,
            qw * omega_z + q_x * omega_y - q_y * omega_x,
            -q_x * omega_x - q_y * omega_y - q_z * omega_z
        ])

        return np.concatenate((q_dot, omega_dot))

def cr3bp(t, state, mu, length_factor, time_factor):
    """
    Compute translational dynamics using the circular restricted three-body problem.

    The input state is converted to normalized CR3BP units before evaluating
    the equations of motion. The resulting velocity and acceleration are
    converted back to SI units.

    Parameters
    ----------
    t : float
        Current integration time [s].
    state : np.ndarray, shape (6,)
        Translational state ordered as position [m] followed by velocity
        [m/s].
    mu : float
        Dimensionless CR3BP mass parameter.
    length_factor : float
        Characteristic length used for normalization [m].
    time_factor : float
        Characteristic time used for normalization [s].

    Returns
    -------
    tuple of np.ndarray
        Velocity vector [m/s] and acceleration vector [m/s²].
    """
    x, y, z, x_dot, y_dot, z_dot = state

    # Normalize the position and velocity vectors
    x /= length_factor
    y /= length_factor
    z /= length_factor
    x_dot /= length_factor / time_factor 
    y_dot /= length_factor / time_factor
    z_dot /= length_factor / time_factor

    t_int = t / time_factor  # Normalize time

    r1 = np.sqrt((x + mu)**2 + y**2 + z**2)
    r2 = np.sqrt((x - (1 - mu))**2 + y**2 + z**2)    

    omega_x = x - ((1 - mu) * (x + mu) / r1**3) - (mu * (x - (1 - mu)) / r2**3)
    omega_y = y - ((1 - mu) * y / r1**3) - (mu * y / r2**3)
    omega_z = - (1 - mu) * (z / r1**3) - (mu * (z / r2**3))
    
    x_ddot = 2 * y_dot + omega_x
    y_ddot = -2 * x_dot + omega_y
    z_ddot = omega_z

    # Rescale the velocity and acceleration back to original units
    x_dot *= length_factor / time_factor
    y_dot *= length_factor / time_factor
    z_dot *= length_factor / time_factor
    x_ddot *= length_factor / time_factor**2
    y_ddot *= length_factor / time_factor**2
    z_ddot *= length_factor / time_factor**2

    velocity = np.array([x_dot, y_dot, z_dot])
    acceleration = np.array([x_ddot, y_ddot, z_ddot])

    return velocity, acceleration

def rel2bp(t, state, mu):
    """
    Compute translational dynamics using the relative two-body model.

    The acceleration is computed from the central gravitational parameter
    using the standard two-body point-mass equation.

    Parameters
    ----------
    t : float
        Current integration time [s].
    state : np.ndarray, shape (6,)
        Translational state ordered as position [m] followed by velocity
        [m/s].
    mu : float
        Gravitational parameter of the central body [m³/s²].

    Returns
    -------
    tuple of np.ndarray
        Velocity vector [m/s] and gravitational acceleration vector [m/s²].
    """
    r = state[:3].copy()
    v = state[3:].copy()
    
    dvdt = -mu * r / np.linalg.norm(r)**3

    velocity = np.array([v[0], v[1], v[2]])
    acceleration = np.array([dvdt[0], dvdt[1], dvdt[2]])

    return velocity, acceleration

def newton(t, state, mass, applied_force, disturbance_force, orbital_model=None, *args):
        """
        Compute the total translational equations of motion.

        The total acceleration is obtained by combining the selected dynamics
        model with externally applied and disturbance forces.

        Parameters
        ----------
        t : float
            Current integration time [s].
        state : np.ndarray, shape (6,)
            Translational state ordered as position [m] followed by velocity
            [m/s].
        mass : float
            Spacecraft mass [kg].
        applied_force : np.ndarray, shape (3,)
            Applied spacecraft force [N].
        disturbance_force : np.ndarray, shape (3,)
            External disturbance force [N].
        orbital_model : callable, optional
            Function defining the orbital dynamics model. If None, no orbital
            dynamics are applied. The function should accept the same parameters
            as this function and return a tuple of velocity and acceleration.

        

        Returns
        -------
        np.ndarray, shape (6,)
            Translational state derivative, consisting of velocity [m/s] and
            acceleration [m/s²].
        """

        if orbital_model is not None:
            _, a_model = orbital_model(t, state, *args[0])
        else:            
            a_model = np.zeros(3)
        
        if mass <= 0:
            a_external = np.zeros(3)
        else:
            a_external = (applied_force + disturbance_force) / mass
        
        a_total = a_model + a_external
        
        v_total = state[3:6]

        acceleration = a_total
        velocity = v_total

        return np.concatenate((velocity, acceleration))