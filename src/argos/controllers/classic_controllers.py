from argos.controllers.controller_base import ControllerBase
import numpy as np
from argos.math import quaternion_error as quat_error

from argos.general.dataclasses import ControlOutput

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from argos.general.dataclasses import ControlOutput, GuidanceOutput, NavigationOutput
    from argos.actuators.actuators_base import ActuatorBase

class PDController(ControllerBase):
    def __init__(self,
                 attitude_proportional_gain,
                 attitude_derivative_gain,
                 translational_proportional_gain,
                 translational_derivative_gain,
                 minimum_torque=None,
                 maximum_torque=None):
        
        self.attitude_proportional_gain = attitude_proportional_gain
        self.attitude_derivative_gain = attitude_derivative_gain
        self.translational_proportional_gain = translational_proportional_gain
        self.translational_derivative_gain = translational_derivative_gain
        self.minimum_torque = minimum_torque
        self.maximum_torque = maximum_torque

        super().__init__()

    def _check_initialization(self):
        if not isinstance(self.attitude_proportional_gain, (int, float)):
            raise TypeError("attitude_proportional_gain must be a numeric value.")

        if not isinstance(self.attitude_derivative_gain, (int, float)):
            raise TypeError("attitude_derivative_gain must be a numeric value.")

        if not isinstance(self.translational_proportional_gain, (int, float)):
            raise TypeError("translational_proportional_gain must be a numeric value.")

        if not isinstance(self.translational_derivative_gain, (int, float)):
            raise TypeError("translational_derivative_gain must be a numeric value.")

        if self.minimum_torque is not None and not isinstance(
            self.minimum_torque, (int, float)
        ):
            raise TypeError("minimum_torque must be a numeric value or None.")

        if self.maximum_torque is not None and not isinstance(
            self.maximum_torque, (int, float)
        ):
            raise TypeError("maximum_torque must be a numeric value or None.")

        if (
            self.minimum_torque is not None
            and self.maximum_torque is not None
            and self.minimum_torque > self.maximum_torque
        ):
            raise ValueError(
                "minimum_torque cannot be greater than maximum_torque."
            )

    def compute_control(self, navigation_output: NavigationOutput, guidance_output: GuidanceOutput) -> ControlOutput:
        # ==========================================================
        # State retrieval
        # ==========================================================
    
        actual_quaternion = navigation_output.spacecraft_state.attitude.astype(float).copy()
        actual_angular_velocity = navigation_output.spacecraft_state.angular_velocity.astype(float).copy()    
    
        guidance_output_quaternion = guidance_output.state.attitude.astype(float).copy()    
    
        # ==========================================================
        # Quaternion attitude error
        # ==========================================================
    
        actual_quaternion /= np.linalg.norm(actual_quaternion)
    
        quaternion_error = quat_error(
            guidance_output_quaternion,
            actual_quaternion,
        )
    
        quaternion_vector = quaternion_error[:3]
        quaternion_scalar = quaternion_error[3]
    
        # Always follow the shortest rotation.
        if quaternion_scalar < 0.0:
    
            quaternion_vector *= -1.0
    
        # ==========================================================
        # PD control law
        # ==========================================================
    
        commanded_torque = (
            -self.attitude_proportional_gain * quaternion_vector
            -self.attitude_derivative_gain * actual_angular_velocity
        )
    
        # ==========================================================
        # Torque saturation
        # ==========================================================
    
        if (
            self.minimum_torque is not None
            and self.maximum_torque is not None
        ):
            commanded_torque = np.clip(
                commanded_torque,
                self.minimum_torque,
                self.maximum_torque,
            )
    
    
        actual_position = (
            navigation_output.spacecraft_state.position
            .astype(float)
            .copy()
        )
    
        actual_velocity = (
            navigation_output.spacecraft_state.velocity
            .astype(float)
            .copy()
        )
    
        guidance_output_position = (
            guidance_output.state.position
            .astype(float)
            .copy()
        )
    
        guidance_output_velocity = (
            guidance_output.state.velocity
            .astype(float)
            .copy()
        )
    
        # ==========================================================
        # Position and velocity errors
        # ==========================================================
    
        position_error = (
            guidance_output_position
            - actual_position
        )
    
        velocity_error = (
            guidance_output_velocity
            - actual_velocity
        )
    
        # ==========================================================
        # PD control law
        # ==========================================================
    
        commanded_force = (
            self.translational_proportional_gain * position_error
            + self.translational_derivative_gain * velocity_error
        )
    
        return ControlOutput(
            force=commanded_force,
            torque=commanded_torque,
        )


class PDAttitudeController(ControllerBase):
    """
    Proportional-Derivative spacecraft attitude controller based on
    quaternion feedback.
    """

    def __init__(
        self,
        proportional_gain: float,
        derivative_gain: float,
        minimum_torque: float | None = None,
        maximum_torque: float | None = None,
    ):
        self.proportional_gain = proportional_gain
        self.derivative_gain = derivative_gain

        self.minimum_torque = minimum_torque
        self.maximum_torque = maximum_torque

        super().__init__()

    def _check_initialization(self):
        if not isinstance(self.proportional_gain, (int, float)):
            raise TypeError("proportional_gain must be a numeric value.")

        if not isinstance(self.derivative_gain, (int, float)):
            raise TypeError("derivative_gain must be a numeric value.")

        if self.minimum_torque is not None and not isinstance(
            self.minimum_torque, (int, float)
        ):
            raise TypeError("minimum_torque must be a numeric value or None.")

        if self.maximum_torque is not None and not isinstance(
            self.maximum_torque, (int, float)
        ):
            raise TypeError("maximum_torque must be a numeric value or None.")

        if (
            self.minimum_torque is not None
            and self.maximum_torque is not None
            and self.minimum_torque > self.maximum_torque
        ):
            raise ValueError(
                "minimum_torque cannot be greater than maximum_torque."
            )    

    def compute_control(self, navigation_output: NavigationOutput, guidance_output: GuidanceOutput) -> ControlOutput:
        """
        Compute the control torque required to track the guidance_output attitude.
        """

        # ==========================================================
        # State retrieval
        # ==========================================================

        actual_quaternion = navigation_output.spacecraft_state.attitude.astype(float).copy()
        actual_angular_velocity = navigation_output.spacecraft_state.angular_velocity.astype(float).copy()     

        # TODO:
        # Replace the true spacecraft state by the estimated state once
        # the navigation filter is integrated.

        guidance_output_quaternion = guidance_output.state.attitude.astype(float).copy()

        # ==========================================================
        # Quaternion attitude error
        # ==========================================================

        actual_quaternion /= np.linalg.norm(actual_quaternion)

        quaternion_error = quat_error(
            guidance_output_quaternion,
            actual_quaternion,
        )

        quaternion_vector = quaternion_error[:3]
        quaternion_scalar = quaternion_error[3]

        # Always follow the shortest rotation.
        if quaternion_scalar < 0.0:

            quaternion_vector *= -1.0

        # ==========================================================
        # PD control law
        # ==========================================================

        commanded_torque = (
            -self.proportional_gain * quaternion_vector
            -self.derivative_gain * actual_angular_velocity
        )

        # ==========================================================
        # Torque saturation
        # ==========================================================

        if (
            self.minimum_torque is not None
            and self.maximum_torque is not None
        ):
            commanded_torque = np.clip(
                commanded_torque,
                self.minimum_torque,
                self.maximum_torque,
            )



        return ControlOutput(
            force=np.zeros(3),
            torque=commanded_torque
        )