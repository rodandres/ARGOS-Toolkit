from py.modules.controllers.controller_base import ControllerBase
import numpy as np
from py.general.data_classes_declaration import SimSharedData
from py.modules.math import quaternion_error as quat_error

from py.general.dataclasses import ControlOutput


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

    def compute_control_old(
        self,
        shared_data: SimSharedData,
    ) -> np.ndarray:
        """
        Compute the control torque required to track the reference attitude.
        """

        # ==========================================================
        # State retrieval
        # ==========================================================

        actual_quaternion = shared_data.actual_q
        actual_angular_velocity = shared_data.actual_omega

        # TODO:
        # Replace the true spacecraft state by the estimated state once
        # the navigation filter is integrated.

        reference_quaternion = shared_data.reference_q

        # ==========================================================
        # Quaternion attitude error
        # ==========================================================

        actual_quaternion /= np.linalg.norm(actual_quaternion)

        quaternion_error = quat_error(
            reference_quaternion,
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

        return commanded_torque

    def compute_control(self, estimated_state, reference):
        """
        Compute the control torque required to track the reference attitude.
        """

        # ==========================================================
        # State retrieval
        # ==========================================================

        actual_quaternion = estimated_state.spacecraft_state.attitude.astype(float).copy()
        actual_angular_velocity = estimated_state.spacecraft_state.angular_velocity.astype(float).copy()     

        # TODO:
        # Replace the true spacecraft state by the estimated state once
        # the navigation filter is integrated.

        reference_quaternion = reference.state.attitude.astype(float).copy()

        # ==========================================================
        # Quaternion attitude error
        # ==========================================================

        actual_quaternion /= np.linalg.norm(actual_quaternion)

        quaternion_error = quat_error(
            reference_quaternion,
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