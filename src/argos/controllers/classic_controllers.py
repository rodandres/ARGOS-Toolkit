from argos.controllers.controller_base import ControllerBase
import numpy as np
from argos.math import quaternion_error as quat_error

from argos.general.dataclasses import ControlOutput

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from argos.general.dataclasses import ControlOutput, GuidanceOutput, NavigationOutput
    from argos.actuators.actuators_base import ActuatorBase

class PDController(ControllerBase):
    """
    Proportional-Derivative controller for spacecraft translation and attitude.

    The controller computes translational force from position and velocity
    errors and rotational torque from quaternion attitude error and angular
    velocity.

    Parameters
    ----------
    Kp_rotational : float | np.ndarray
        Proportional gain applied to the quaternion attitude error.
        If a scalar is provided, the same gain is applied to all three axes.
        If an array is provided, it should be of shape (3,) or (3, 3) to specify different gains for each axis.
    Kd_rotational : float | np.ndarray
        Derivative gain applied to the spacecraft angular velocity.
        If a scalar is provided, the same gain is applied to all three axes.
        If an array is provided, it should be of shape (3,) or (3, 3) to specify different gains for each axis.
    Kp_translational : float | np.ndarray
        Proportional gain applied to the position error.
        If a scalar is provided, the same gain is applied to all three axes.
        If an array is provided, it should be of shape (3,) or (3, 3) to specify different gains for each axis.
    Kd_translational : float | np.ndarray
        Derivative gain applied to the velocity error.
        If a scalar is provided, the same gain is applied to all three axes.
        If an array is provided, it should be of shape (3,) or (3, 3) to specify different gains for each axis.
    minimum_torque : float, optional
        Minimum torque command applied during torque saturation.
    maximum_torque : float, optional
        Maximum torque command applied during torque saturation.
    """
    def __init__(self,
                 Kp_rotational: float | np.ndarray,
                 Kd_rotational: float | np.ndarray,
                 Kp_translational: float | np.ndarray,
                 Kd_translational: float | np.ndarray,
                 minimum_torque=None,
                 maximum_torque=None):

        if isinstance(Kp_rotational, (int, float)):
            Kp_rotational = np.array([Kp_rotational] * 3)
        if isinstance(Kd_rotational, (int, float)):
            Kd_rotational = np.array([Kd_rotational] * 3)
        if isinstance(Kp_translational, (int, float)):
            Kp_translational = np.array([Kp_translational] * 3)
        if isinstance(Kd_translational, (int, float)):
            Kd_translational = np.array([Kd_translational] * 3)
        
        self.Kp_rotational = Kp_rotational
        self.Kd_rotational = Kd_rotational
        self.Kp_translational = Kp_translational
        self.Kd_translational = Kd_translational
        self.minimum_torque = minimum_torque
        self.maximum_torque = maximum_torque

        super().__init__()

    def _check_initialization(self):
        if not isinstance(self.Kp_rotational, (int, float)):
            if not isinstance(self.Kp_rotational, np.ndarray):
                raise TypeError("Kp_rotational must be a numeric value or a numpy array.")            

        if not isinstance(self.Kd_rotational, (int, float)):
            if not isinstance(self.Kd_rotational, np.ndarray):
                raise TypeError("Kd_rotational must be a numeric value or a numpy array.")

        if not isinstance(self.Kp_translational, (int, float)):
            if not isinstance(self.Kp_translational, np.ndarray):
                raise TypeError("Kp_translational must be a numeric value or a numpy array.")

        if not isinstance(self.Kd_translational, (int, float)):
            if not isinstance(self.Kd_translational, np.ndarray):
                raise TypeError("Kd_translational must be a numeric value or a numpy array.")

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
        Compute translational force and attitude-control torque.

        The translational control command is computed from the position and
        velocity errors between the guidance state and the navigation state.
        The attitude control command is computed from the quaternion attitude
        error and the spacecraft angular velocity.

        Parameters
        ----------
        navigation_output : NavigationOutput
            Navigation solution containing the current spacecraft state.
        guidance_output : GuidanceOutput
            Guidance output containing the desired spacecraft state.

        Returns
        -------
        ControlOutput
            Control command containing the commanded force [N] and torque [N·m].
        """
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
            -self.Kp_rotational * quaternion_vector
            -self.Kd_rotational * actual_angular_velocity
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
            self.Kp_translational * position_error
            + self.Kd_translational * velocity_error
        )
    
        return ControlOutput(
            force=commanded_force,
            torque=commanded_torque,
        )


class PDAttitudeController(ControllerBase):
    """
    Proportional-Derivative spacecraft attitude controller based on quaternion feedback.

    The controller computes a commanded torque from the quaternion attitude
    error and the spacecraft angular velocity. Optional torque saturation
    limits can be applied to the resulting command.

    Parameters
    ----------
    Kp : float | np.ndarray
        Proportional gain applied to the quaternion attitude error.
        If a scalar is provided, the same gain is applied to all three axes.
        If an array is provided, it should be of shape (3,) or (3, 3) to specify different gains for each axis.
    Kd : float | np.ndarray
        Derivative gain applied to the spacecraft angular velocity.
        If a scalar is provided, the same gain is applied to all three axes.
        If an array is provided, it should be of shape (3,) or (3, 3) to specify different gains for each axis.
    minimum_torque : float, optional
        Minimum torque command applied during torque saturation.
    maximum_torque : float, optional
        Maximum torque command applied during torque saturation.
    """

    def __init__(
        self,
        Kp: float | np.ndarray,
        Kd: float | np.ndarray,
        minimum_torque: float | None = None,
        maximum_torque: float | None = None,
    ):
        self.Kp = Kp
        self.Kd = Kd

        if isinstance(Kp, (int, float)):
            self.Kp = np.array([Kp] * 3)
        if isinstance(Kd, (int, float)):
            self.Kd = np.array([Kd] * 3)

        self.minimum_torque = minimum_torque
        self.maximum_torque = maximum_torque

        super().__init__()

    def _check_initialization(self):
        
        if not isinstance(self.Kp, np.ndarray):
            raise TypeError("Kp must be a numeric value or a numpy array.")            

        
        if not isinstance(self.Kd, np.ndarray):
            raise TypeError("Kd must be a numeric value or a numpy array.")

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
        Compute the commanded attitude-control torque.

        The attitude error is computed from the desired and current
        quaternions. The resulting proportional-derivative control law produces
        a torque command while the translational force is set to zero.

        Parameters
        ----------
        navigation_output : NavigationOutput
            Navigation solution containing the current spacecraft attitude and
            angular velocity.
        guidance_output : GuidanceOutput
            Guidance output containing the desired spacecraft attitude.

        Returns
        -------
        ControlOutput
            Control command containing zero translational force [N] and the
            commanded attitude-control torque [N·m].
        """

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
            -self.Kp * quaternion_vector
            -self.Kd * actual_angular_velocity
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