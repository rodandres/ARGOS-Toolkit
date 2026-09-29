import numpy as np
from argos.guidance.guidance_base import GuidanceBase
from argos.general.dataclasses import GuidanceOutput, SimulationData, StateVariables

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from argos.navigation.navigation_base import NavigationOutput

class ConstantReferenceGuidance(GuidanceBase):
    """
    Guidance law that provides a constant desired spacecraft state.

    The desired translational and rotational state remains constant for all
    guidance evaluations. Any state component not explicitly provided is
    initialized to its corresponding zero value, with the attitude
    quaternion defaulting to the identity quaternion.

    Parameters
    ----------
    desired_pos : np.ndarray, shape (3,), optional
        Desired position [m].
    desired_vel : np.ndarray, shape (3,), optional
        Desired velocity [m/s].
    desired_accel : np.ndarray, shape (3,), optional
        Desired acceleration [m/s²].
    desired_quat : np.ndarray, shape (4,), optional
        Desired attitude quaternion.
    desired_ang_vel : np.ndarray, shape (3,), optional
        Desired angular velocity [rad/s].
    desired_ang_accel : np.ndarray, shape (3,), optional
        Desired angular acceleration [rad/s²].

    Raises
    ------
    ValueError
        If no desired state component is provided, or if an input array has
        an invalid shape.
    TypeError
        If a provided state component is not a NumPy array.
    """
    def __init__(self,
                 desired_pos: np.ndarray | None=None, desired_vel: np.ndarray | None=None, desired_accel: np.ndarray | None=None,
                 desired_quat: np.ndarray | None=None, desired_ang_vel: np.ndarray | None=None, desired_ang_accel: np.ndarray | None=None):
        self.desired_pos = desired_pos
        self.desired_vel = desired_vel
        self.desired_accel = desired_accel
        self.desired_quat = desired_quat
        self.desired_ang_vel = desired_ang_vel
        self.desired_ang_accel = desired_ang_accel

        self._check_initialization()

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

    def compute_reference(self, navigation_data: NavigationOutput, simulation_data: SimulationData):
        """
        Compute the constant guidance reference.

        Parameters
        ----------
        navigation_data : NavigationOutput
            Current navigation output for the spacecraft.
        simulation_data : SimulationData
            Current simulation data.

        Returns
        -------
        GuidanceOutput
            Guidance output containing the configured desired position,
            velocity, acceleration, attitude, angular velocity, and angular
            acceleration.
        """

        state = StateVariables(
            position = self.desired_pos,
            velocity = self.desired_vel,
            acceleration = self.desired_accel,
            attitude = self.desired_quat,
            angular_velocity = self.desired_ang_vel,
            angular_acceleration = self.desired_ang_accel
        )

        return GuidanceOutput(
            state=state   
        )

class CustomGuidanceLaw(GuidanceBase):
    """
    Guidance law that delegates reference computation to a user-defined function.

    Parameters
    ----------
    custom_reference_function : callable
        Function used to compute the guidance reference. It receives the
        navigation output and simulation data and must return a
        ``GuidanceOutput`` instance.

    Raises
    ------
    ValueError
        If ``custom_reference_function`` is not callable.
    """
    def __init__(self, custom_reference_function):
        self.custom_reference_function = custom_reference_function
        super().__init__()

    def _check_initialization(self):
        if not callable(self.custom_reference_function):
            raise ValueError("custom_reference_function must be callable.")

    def compute_reference(self, navigation_data: NavigationOutput, simulation_data: SimulationData):
        """
        Compute the guidance reference using the user-defined function.

        Parameters
        ----------
        navigation_data : NavigationOutput
            Current navigation output for the spacecraft.
        simulation_data : SimulationData
            Current simulation data.

        Returns
        -------
        GuidanceOutput
            Guidance output returned by the user-defined function.

        Raises
        ------
        TypeError
            If the custom reference function does not return a
            ``GuidanceOutput`` instance.
        """
        reference = self.custom_reference_function(navigation_data, simulation_data)

        if not isinstance(reference, GuidanceOutput):
            raise TypeError("The custom_reference_function must return a GuidanceOutput object.")
        
        return reference


class GuidancePlaceholder(GuidanceBase):
    """
    Placeholder guidance law for unsupported or unimplemented guidance logic.
    """
    def _check_initialization(self):
        pass  # No specific initialization checks for the placeholder

    def compute_reference(self, navigation_data, simulation_data):
        """
        Placeholder method for computing a guidance reference.

        Parameters
        ----------
        navigation_data : NavigationOutput
            Current navigation output for the spacecraft.
        simulation_data : SimulationData
            Current simulation data.

        Raises
        ------
        NotImplementedError
            Always raised because the guidance computation is not implemented.
        """
        raise NotImplementedError("This is a placeholder method. Please implement your own guidance law.")

    