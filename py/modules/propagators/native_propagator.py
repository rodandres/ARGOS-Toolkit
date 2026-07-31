from py.modules.propagators.propagator_base import TranslationalPropagatorBase, RotationalPropagatorBase
import numpy as np

from py.modules.solvers.solvers_base import solve

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


    def propagate(self, simulation_data, environment):

        for spacecraft in simulation_data.spacecrafts:
            q = spacecraft.spacecraft_data.true_q
            omega = spacecraft.spacecraft_data.true_omega

            I = spacecraft.inertia_tensor
            I_inv = spacecraft.inertia_tensor_inv
            applied_torque = spacecraft.spacecraft_data.current_torque_exerted
            disturbance_torque = environment.get_perturbation_torque()

            # Define the state vector
            state = np.concatenate((q, omega))

            t_end = min(simulation_data.t + simulation_data.dt_propagation, simulation_data.max_sim_time)

            sol = solve(self.quaternion_dynamics,
                        [simulation_data.t, t_end],
                        state,
                        self.integration_method,
                        args=(I, I_inv, applied_torque, disturbance_torque), h0=simulation_data.dt_propagation, h_adaptative=False
                        )            

            state_end = sol.y[:, -1]

            spacecraft.spacecraft_data.true_q = state_end[0:4]
            spacecraft.spacecraft_data.true_q /= np.linalg.norm(spacecraft.spacecraft_data.true_q)  # Normalize quaternion
            spacecraft.spacecraft_data.true_omega = state_end[4:7]
            spacecraft.spacecraft_data.true_alpha = self.quaternion_dynamics(t_end, state_end, I, I_inv, applied_torque, disturbance_torque)[4:7]

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

    def rel2bp(self, t, state, mu):
        r = state[:3].copy()
        v = state[3:].copy()

        drdt = v
        dvdt = -mu * r / np.linalg.norm(r)**3

        return np.concatenate((drdt, dvdt))


    def propagate(self, simulation_data, environment):

        for spacecraft in simulation_data.spacecrafts:
            position = spacecraft.spacecraft_data.true_pos
            velocity = spacecraft.spacecraft_data.true_vel

            mass = spacecraft.mass
            applied_force = spacecraft.spacecraft_data.current_force_exerted            

            # Define the state vector
            state = np.concatenate((position, velocity))

            t_end = min(simulation_data.t + simulation_data.dt_propagation, simulation_data.max_sim_time)
    
            mu = 6.67430e-11 * 5.972e24

            sol = solve(self.rel2bp,
                        [simulation_data.t, t_end],
                        state,
                        self.integration_method,
                        args=(mu, ), h0=simulation_data.dt_propagation, h_adaptative=True
                        )

            state_end = sol.y[:, -1]

            spacecraft.spacecraft_data.true_pos = state_end[0:3]
            spacecraft.spacecraft_data.true_vel = state_end[3:6]
            
