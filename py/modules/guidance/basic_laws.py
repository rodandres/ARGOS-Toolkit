import numpy as np
from py.modules.guidance.guidance_base import GuidanceBase
from py.general.dataclasses import GuidanceReference

class ConstantReferenceGuidance(GuidanceBase):
    def __init__(self,
                 desired_pos: np.ndarray | None=None, desired_vel: np.ndarray | None=None, desired_accel: np.ndarray | None=None,
                 desired_quat: np.ndarray | None=None, desired_ang_vel: np.ndarray | None=None, desired_ang_accel: np.ndarray | None=None):
        self.desired_pos = desired_pos
        self.desired_vel = desired_vel
        self.desired_accel = desired_accel
        self.desired_quat = desired_quat
        self.desired_ang_vel = desired_ang_vel
        self.desired_ang_accel = desired_ang_accel

    def _check_initialization(self):
        if all(x is None for x in (
            self.desired_pos,
            self.desired_vel,
            self.desired_accel,
            self.desired_quat,
            self.desired_ang_vel,
            self.desired_ang_accel,
        )):
            raise ValueError(
                "At least one desired state must be provided for ConstantReferenceGuidance."
            )
        
        for attr in (
            "desired_pos",
            "desired_vel",
            "desired_accel",
            "desired_ang_vel",
            "desired_ang_accel",
        ):
            value = getattr(self, attr)

            if value is None:
                setattr(self, attr, np.zeros(3, dtype=float))
                continue

            if not isinstance(value, np.ndarray):
                raise TypeError(f"{attr} must be a numpy.ndarray.")

            if value.shape != (3,):
                raise ValueError(
                    f"{attr} must have shape (3,), but got {value.shape}."
                )

        if self.desired_quat is None:
            self.desired_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        else:
            if not isinstance(self.desired_quat, np.ndarray):
                raise TypeError("desired_quat must be a numpy.ndarray.")

            if self.desired_quat.shape != (4,):
                raise ValueError(
                    f"desired_quat must have shape (4,), but got {self.desired_quat.shape}."
                )

    def compute_reference(self, navigation_estimated_data, simulation_data):
        """
        Compute the guidance reference based on navigation estimated data and simulation data.

        Args:
            navigation_estimated_data: The estimated state of the spacecraft from the navigation system.
            simulation_data: The current state of the simulation.
        Returns:
            A GuidanceReference object containing the desired position, velocity, acceleration, quaternion, angular velocity, and angular acceleration.
        """

        return GuidanceReference(
            position = self.desired_pos,
            velocity = self.desired_vel,
            acceleration = self.desired_accel,
            attitude = self.desired_quat,
            angular_velocity = self.desired_ang_vel,
            angular_acceleration = self.desired_ang_accel
        )

class CustomGuidanceLaw(GuidanceBase):
    def __init__(self, custom_reference_function):
        self.custom_reference_function = custom_reference_function

    def _check_initialization(self):
        if not callable(self.custom_reference_function):
            raise ValueError("custom_reference_function must be callable.")

    def compute_reference(self, navigation_estimated_data, simulation_data):
        """
        Compute the guidance reference using a custom function.

        Args:
            navigation_estimated_data: The estimated state of the spacecraft from the navigation system.
            simulation_data: The current state of the simulation.
        Returns:
            A GuidanceReference object containing the desired position, velocity, acceleration, quaternion, angular velocity, and angular acceleration.
        """
        return self.custom_reference_function(navigation_estimated_data, simulation_data)