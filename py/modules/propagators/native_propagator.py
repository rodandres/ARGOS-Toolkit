from py.modules.propagators.propagator_base import TranslationalPropagatorBase, RotationalPropagatorBase
import numpy as np

from py.modules.solvers.solvers_base import solve

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from py.modules.new.spacecraft import Spacecraft
    from py.general.dataclasses import SimulationData
    from py.modules.enviroments.environment_base import EnvironmentBase

class NativeRotationalPropagator(RotationalPropagatorBase):

    def __init__(self, integration_method: str = "NATIVE_RK45"):
        super().__init__(integration_method)    

    def quaternion_dynamics(self, t, state, inertia_matrix, inverse_inertia_matrix, applied_torque, disturbance_torque):        

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



    def propagate(self, spacecraft: Spacecraft, simulation_data: SimulationData, environment: EnvironmentBase):
        
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
        Initialize the rotational propagator with the given simulation data.

        Parameters:
        simulation_data: The data required to initialize the rotational propagator.
        """
        # Implement any initialization logic needed for the native rotational propagator
        pass

class NativeTranslationalPropagator(TranslationalPropagatorBase):

    def __init__(self, integration_method: str = "NATIVE_RK45"):
        super().__init__(integration_method)

    def translational_dynamics(self, t, state, mass, applied_force, disturbance_force):
        position = state[:3].copy()
        velocity = state[3:].copy()

        acceleration = (applied_force + disturbance_force) / mass

        return np.concatenate((velocity, acceleration))

    def cr3bp(self, t, state, mu):
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


    def rel2bp(self, t, state, mu):
        r = state[:3].copy()
        v = state[3:].copy()

        drdt = v
        dvdt = -mu * r / np.linalg.norm(r)**3

        return np.concatenate((drdt, dvdt))


    def propagate(self, spacecraft, simulation_data, environment):
    
        position = spacecraft.spacecraft_data.true_state.position
        velocity = spacecraft.spacecraft_data.true_state.velocity

        mass = spacecraft.mass
        applied_force = spacecraft.spacecraft_data.current_force_exerted            

        # Define the state vector
        state = np.concatenate((position, velocity))

        t_end = min(spacecraft.spacecraft_data.t + spacecraft.spacecraft_data.current_propagation_dt, simulation_data.max_sim_time)                

        mu = 6.67430e-11 * 5.972e24 # For LEO
        mu = 1.215e-2 # For Earth-Moon system - CR3BP

        sol = solve(self.cr3bp,
                    [spacecraft.spacecraft_data.t, t_end],
                    state,
                    self.integration_method,
                    args=(mu, ), h0=spacecraft.spacecraft_data.current_propagation_dt, h_adaptative=True
                    )

        state_end = sol.y[:, -1]

        spacecraft.spacecraft_data.true_state.position = state_end[0:3]
        spacecraft.spacecraft_data.true_state.velocity = state_end[3:6]