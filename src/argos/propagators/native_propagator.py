from argos.propagators.propagator_base import TranslationalPropagatorBase, RotationalPropagatorBase
import numpy as np

from argos.solvers.solvers_base import solve
from argos.propagators.dynamic_models import (
    quaternion_dynamics,
    cr3bp,
    rel2bp,
    newton
)

from argos import _cpp

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

        self.dynamic_model = quaternion_dynamics

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

        sol = solve(self.dynamic_model,
                    [spacecraft.spacecraft_data.t, t_end],
                    state,
                    self.integration_method,
                    args=(I, I_inv, applied_torque, disturbance_torque), h0=spacecraft.spacecraft_data.current_propagation_dt, h_adaptative=False
                    )            

        state_end = sol.y[:, -1]

        spacecraft.spacecraft_data.true_state.attitude = state_end[0:4]
        spacecraft.spacecraft_data.true_state.attitude /= np.linalg.norm(spacecraft.spacecraft_data.true_state.attitude)  # Normalize quaternion
        spacecraft.spacecraft_data.true_state.angular_velocity = state_end[4:7]
        spacecraft.spacecraft_data.true_state.angular_acceleration = self.dynamic_model(t_end, state_end, I, I_inv, applied_torque, disturbance_torque)[4:7]

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
    model : str, optional
        Orbital dynamics model. Supported values are ``"REL2BP"``,
        ``"CR3BP"``.
    integration_method : str, optional
        Numerical integration method used by the solver.
        Defaults to ``"NATIVE_RK45"``.
    """

    def __init__(self, orbital_model: str | None = None, integration_method: str = "NATIVE_RK45"):
        super().__init__(integration_method)

        self.dynamic_model = None
        self.orbital_model = None
        self.orbital_model_arguments = ()
        self._set_orbital_model(orbital_model)        
        self._set_dynamics()


    def _set_orbital_model(self, orbital_model: str):
        """
        Select the translational dynamics model.

        Parameters
        ----------
        orbital_model : str
            Dynamics model to use. Supported values are ``"REL2BP"`` and,
            ``"CR3BP"``

        Raises
        ------
        ValueError
            If ``orbital_nodel`` is not one of the supported dynamics models.
        """
        if orbital_model not in SUPPORTED_DYNAMICS and orbital_model is not None:            
            raise ValueError(f"Unsupported dynamics model: {orbital_model}. Supported models are: {SUPPORTED_DYNAMICS}")

        if orbital_model == "REL2BP":
            self.orbital_model = rel2bp
            if self.using_cpp: self.orbital_model = "REL2BP"
            mu = 6.67430e-11 * 5.972e24 # For LEO
            self.orbital_model_arguments = (mu, )
        elif orbital_model == "CR3BP":
            self.orbital_model = cr3bp
            if self.using_cpp: self.orbital_model = "CR3BP"
            mu = 1.215e-2 # For Earth-Moon system - CR3BP
            length_factor = 384400.0e3 # Earth-Moon distance in m
            time_factor = 27.321661 / (2.0 * np.pi) * 24 * 3600 # Sideral lunar month in seconds
            self.orbital_model_arguments  = (mu, length_factor, time_factor, )
        
    def _set_dynamics(self):
        if self.using_cpp:
            self.dynamic_model = "NEWTON"
            
        else:
            self.dynamic_model = newton    

    def evaluate_dynamics(
        self,
        t_end,
        state_end,
        mass,
        applied_force,
        disturbance_force,
    ):

        if self.using_cpp:

            if self.orbital_model == "CR3BP":
                mu, length_factor, time_factor = self.orbital_model_arguments

            elif self.orbital_model == "REL2BP":
                mu = self.orbital_model_arguments[0]
                length_factor = 1.0
                time_factor = 1.0

            else:
                mu = 0.0
                length_factor = 1.0
                time_factor = 1.0

            derivative = _cpp.evaluate_translational_dynamics(
                state_end,
                t_end,
                mass,
                applied_force,
                disturbance_force,
                self.dynamic_model,
                self.orbital_model,
                mu,
                length_factor,
                time_factor,
            )

        else:

            derivative = self.dynamic_model(
                t_end,
                state_end,
                mass,
                applied_force,
                disturbance_force,
                self.orbital_model,
                self.orbital_model_arguments,
            )

        return derivative


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

        sol = solve(self.dynamic_model,
                    [spacecraft.spacecraft_data.t, t_end],
                    state,
                    self.integration_method,
                    args=(mass, applied_force, np.zeros(3), self.orbital_model, (self.orbital_model_arguments) ), h0=spacecraft.spacecraft_data.current_propagation_dt, h_adaptative=True
                    )

        state_end = sol.y[:, -1]

        spacecraft.spacecraft_data.true_state.position = state_end[0:3]
        spacecraft.spacecraft_data.true_state.velocity = state_end[3:6]

        
        spacecraft.spacecraft_data.true_state.acceleration = (
            self.evaluate_dynamics(
                t_end,
                state_end,
                mass,
                applied_force,
                np.zeros(3),
            )[3:6]
        )