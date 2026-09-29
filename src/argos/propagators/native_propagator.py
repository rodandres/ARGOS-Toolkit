from argos.propagators.propagator_base import TranslationalPropagatorBase, RotationalPropagatorBase
import numpy as np

from argos.solvers.solvers_base import solve

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from argos.core.spacecraft import Spacecraft
    from argos.general.dataclasses import SimulationData
    from argos.enviroments.environment_base import EnvironmentBase

class NativeRotationalPropagator(RotationalPropagatorBase):
    """
    Propagate spacecraft rotational dynamics using a native numerical solver.

    The rotational state consists of the attitude quaternion and angular
    velocity. Quaternion dynamics are integrated together with the rigid-body
    rotational equations of motion.

    Parameters
    ----------
    integration_method : str, optional
        Numerical integration method used by the solver.
        Defaults to ``"NATIVE_RK45"``.
    """
    def __init__(self, integration_method: str = "NATIVE_RK45"):
        super().__init__(integration_method)    

    def quaternion_dynamics(
        self,
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


    def propagate(
        self,
        spacecraft: Spacecraft,
        simulation_data: SimulationData,
        environment: EnvironmentBase,
    ):
        """
        Propagate the spacecraft rotational state over the current time step.

        The attitude quaternion and angular velocity are integrated from the
        current spacecraft state to the end of the current propagation interval.
        The resulting quaternion is normalized after integration.

        Parameters
        ----------
        spacecraft : Spacecraft
            Spacecraft whose rotational state is propagated.
        simulation_data : SimulationData
            Current simulation data and maximum simulation time.
        environment : EnvironmentBase
            Environment model providing the disturbance torque.
        """
        
        q = spacecraft.spacecraft_data.true_state.attitude
        omega = spacecraft.spacecraft_data.true_state.angular_velocity

        I = spacecraft.inertia_tensor
        I_inv = spacecraft.inertia_tensor_inv
        applied_torque = spacecraft.spacecraft_data.current_torque_exerted
        disturbance_torque = environment.get_perturbation_torque()

        # Define the state vector
        state = np.concatenate((q, omega))

        t_end = min(spacecraft.spacecraft_data.t + spacecraft.spacecraft_data.current_propagation_dt, simulation_data.max_sim_time)

        sol = solve(self.quaternion_dynamics,
                    [spacecraft.spacecraft_data.t, t_end],
                    state,
                    self.integration_method,
                    args=(I, I_inv, applied_torque, disturbance_torque), h0=spacecraft.spacecraft_data.current_propagation_dt, h_adaptative=False
                    )            

        state_end = sol.y[:, -1]

        spacecraft.spacecraft_data.true_state.attitude = state_end[0:4]
        spacecraft.spacecraft_data.true_state.attitude /= np.linalg.norm(spacecraft.spacecraft_data.true_state.attitude)  # Normalize quaternion
        spacecraft.spacecraft_data.true_state.angular_velocity = state_end[4:7]
        spacecraft.spacecraft_data.true_state.angular_acceleration = self.quaternion_dynamics(t_end, state_end, I, I_inv, applied_torque, disturbance_torque)[4:7]

    def initialize(self, simulation_data):
        """
        Initialize the native rotational propagator.

        Parameters
        ----------
        simulation_data : SimulationData
            Current simulation data used during initialization.
        """
        # Implement any initialization logic needed for the native rotational propagator
        pass

SUPPORTED_DYNAMICS = ["REL2BP", "CR3BP", "NEWTON"]

class NativeTranslationalPropagator(TranslationalPropagatorBase):
    """
    Propagate spacecraft translational dynamics using a native numerical solver.

    The propagator supports two-body relative dynamics, the circular
    restricted three-body problem, and a pure Newtonian force model with
    externally applied forces.

    Parameters
    ----------
    dynamics : str, optional
        Translational dynamics model. Supported values are ``"REL2BP"``,
        ``"CR3BP"``, and ``"NEWTON"``. Defaults to ``"NEWTON"``.
    integration_method : str, optional
        Numerical integration method used by the solver.
        Defaults to ``"NATIVE_RK45"``.
    """

    def __init__(self, dynamics: str = "NEWTON", integration_method: str = "NATIVE_RK45"):
        super().__init__(integration_method)
        self.set_dynamic_model(dynamics)

    def set_dynamic_model(self, dynamics: str):
        """
        Select the translational dynamics model.

        Parameters
        ----------
        dynamics : str
            Dynamics model to use. Supported values are ``"REL2BP"``,
            ``"CR3BP"``, and ``"NEWTON"``.

        Raises
        ------
        ValueError
            If ``dynamics`` is not one of the supported dynamics models.
        """
        if dynamics not in SUPPORTED_DYNAMICS:
            raise ValueError(f"Unsupported dynamics model: {dynamics}. Supported models are: {SUPPORTED_DYNAMICS}")

        if dynamics == "REL2BP":
            self.dynamics_function = self.rel2bp
            mu = 6.67430e-11 * 5.972e24 # For LEO
            self.arguments  = (mu, )
        elif dynamics == "CR3BP":
            self.dynamics_function = self.cr3bp
            mu = 1.215e-2 # For Earth-Moon system - CR3BP
            length_factor = 384400.0e3 # Earth-Moon distance in m
            time_factor = 27.321661 / (2.0 * np.pi) * 24 * 3600 # Sideral lunar month in seconds
            self.arguments  = (mu, length_factor, time_factor, )
        elif dynamics == "NEWTON":
            self.dynamics_function = None
            self.arguments  = ()
        
        
    def cr3bp(self, t, state, mu, length_factor, time_factor):
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

    def rel2bp(self, t, state, mu):
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

    def dynamics(self, t, state, mass, applied_force, disturbance_force):
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

        Returns
        -------
        np.ndarray, shape (6,)
            Translational state derivative, consisting of velocity [m/s] and
            acceleration [m/s²].
        """

        if self.dynamics_function is not None:
            _, a_model = self.dynamics_function(t, state, *self.arguments )
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

    def propagate(
        self,
        spacecraft,
        simulation_data,
        environment,
    ):
        """
        Propagate the spacecraft translational state over the current time step.

        The current position and velocity are integrated using the configured
        dynamics model and applied spacecraft force. The resulting state and
        acceleration are written back to the spacecraft true state.

        Parameters
        ----------
        spacecraft : Spacecraft
            Spacecraft whose translational state is propagated.
        simulation_data : SimulationData
            Current simulation data and maximum simulation time.
        environment : EnvironmentBase
            Environment model associated with the simulation.
        """
    
        position = spacecraft.spacecraft_data.true_state.position
        velocity = spacecraft.spacecraft_data.true_state.velocity

        mass = spacecraft.mass
        applied_force = spacecraft.spacecraft_data.current_force_exerted

        # Define the state vector
        state = np.concatenate((position, velocity))

        t_end = min(spacecraft.spacecraft_data.t + spacecraft.spacecraft_data.current_propagation_dt, simulation_data.max_sim_time)                            

        sol = solve(self.dynamics,
                    [spacecraft.spacecraft_data.t, t_end],
                    state,
                    self.integration_method,
                    args=(mass, applied_force, np.zeros(3), ), h0=spacecraft.spacecraft_data.current_propagation_dt, h_adaptative=True
                    )

        state_end = sol.y[:, -1]

        spacecraft.spacecraft_data.true_state.position = state_end[0:3]
        spacecraft.spacecraft_data.true_state.velocity = state_end[3:6]
        spacecraft.spacecraft_data.true_state.acceleration = self.dynamics(t_end, state_end, mass, applied_force, np.zeros(3))[3:6]