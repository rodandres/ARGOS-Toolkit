import warnings
import numpy as np
from argos.sensors.sensor_base import SensorBase
from argos.general_tools import _as_3d_array

from argos.sensors.error_models import *
from argos.frames.frames import BODY_frame, NONE_frame

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from argos.general.dataclasses import SimulationData, SpacecraftData    

class Accelerometer(SensorBase):
    """
    Model a spacecraft accelerometer with configurable measurement errors.

    The accelerometer computes specific force, transforms it from the
    inertial frame to the sensor frame, and applies the configured bias,
    white-noise, random-walk, saturation, and quantization models.

    NOTE: The current accelerometer implementation assumes that the angular velocity
    and angular acceleration provided by the propagation state correspond to the
    body's rotation relative to an inertial frame. It does not explicitly account
    for the corrections required if these quantities are defined relative to a
    non-inertial propagation frame.

    Parameters
    ----------
    position : np.ndarray or float, optional
        Accelerometer position relative to the spacecraft body frame [m].
    rotation : np.ndarray or float, optional
        Accelerometer orientation relative to the spacecraft body frame [deg].
    bias : np.ndarray or float, optional
        Constant accelerometer bias.
    noise_std : np.ndarray or float, optional
        Standard deviation of white measurement noise.
    random_walk_std : np.ndarray or float, optional
        Random-walk noise parameter.
    sample_rate_freq : float, optional
        Sensor sampling frequency [Hz].
    saturation_limit : np.ndarray or float, optional
        Symmetric measurement saturation limit.
    quantization_bits : int or None, optional
        Number of quantization bits.
    name : str, optional
        Sensor identifier.
    verbose : bool, optional
        If True, print sensor information during initialization.
    """
    def __init__(self,
                 position: np.ndarray | float = 0.0,
                 rotation: np.ndarray | float = 0.0,
                 bias: np.ndarray | float = 0.0,                 
                 noise_std: np.ndarray | float = 0.0,
                 random_walk_std: np.ndarray | float = 0.0,
                 sample_rate_freq: float = 100.0,
                 saturation_limit: np.ndarray | float = np.inf,
                 quantization_bits: int | None = 16,
                 name: str = None,
                 verbose: bool = False):
                
        error_models = (
            Bias(_as_3d_array(bias, "bias")),
            WhiteNoise(_as_3d_array(noise_std, "noise_std")),
            RandomWalk(_as_3d_array(random_walk_std, "random_walk_std")),
            Saturation(_as_3d_array(saturation_limit, "saturation_limit")),
            Quantization(_as_3d_array(saturation_limit, "saturation_limit"), quantization_bits),
            )
        
        super().__init__(
            type="Accelerometer",
            position=position,
            rotation=rotation,
            sample_rate_freq=sample_rate_freq,
            error_models=error_models,
            name=name,
            verbose=verbose)        

    def compute_specific_accel(self) -> np.ndarray:
        """
        Compute the specific acceleration measured in the body frame.

        Specific acceleration is computed based on exerted forces from actuators and
        perturbations.

        Returns
        -------
        np.ndarray, shape (3,)
            Specific acceleration [m/s²].
        """        

        exerted_data = self.spacecraft_data.exerted_data
        exerted_force = exerted_data.force
        exerted_frame = exerted_data.frame

        perturbation_data = self.spacecraft_data.perturbation_data
        perturbation_force = perturbation_data.force
        perturbation_frame = perturbation_data.frame

        # Transform to body frame:
        exerted_force_body_frame = self.spacecraft_transform_manager.transform_vector(exerted_force, exerted_frame, BODY_frame)
        perturbation_force_body_frame = self.spacecraft_transform_manager.transform_vector(perturbation_force, perturbation_frame, BODY_frame)
        
        # Compute specific acceleration in the body frame
        sc_mass = self.parent_spacecraft.mass

        if sc_mass <= 0:
            warnings.warn(
                f"Spacecraft mass is non-positive ({sc_mass} kg). "
                "Specific acceleration will be set to zero."
            )
            return np.zeros(3)

        specific_accel_in_body = (exerted_force_body_frame + perturbation_force_body_frame) / sc_mass
        
        return specific_accel_in_body
           
    def get_ideal_measurement(
        self,
    ) -> np.ndarray:
        """
        Compute the ideal accelerometer measurement.

        Parameters
        ----------
        spacecraft_data : SpacecraftData
            Current spacecraft state.
        simulation_data : SimulationData
            Current simulation data.

        Returns
        -------
        np.ndarray, shape (3,)
            Specific acceleration expressed in the sensor frame [m/s²].
        """
        specific_accel_in_body = self.compute_specific_accel()

        # Transform the specific acceleration from the body frame to the sensor frame
        true_state_data = self.spacecraft_data.true_state
        rotational_propagation_frame = true_state_data.rotational_frame

        if rotational_propagation_frame != NONE_frame:
            # Transform specific acceleration from the propagation frame to the body frame
            omega_body = self.spacecraft_transform_manager.transform_vector(self.spacecraft_data.true_state.angular_velocity, rotational_propagation_frame, BODY_frame)
            alpha_body = self.spacecraft_transform_manager.transform_vector(self.spacecraft_data.true_state.angular_acceleration, rotational_propagation_frame, BODY_frame)

            tangential_accel_body = np.cross(alpha_body, self.position)
            centripetal_accel_body = np.cross(omega_body, np.cross(omega_body, self.position))

            specific_accel_in_body_from_propagation = specific_accel_in_body + tangential_accel_body + centripetal_accel_body
        else:
            specific_accel_in_body_from_propagation = specific_accel_in_body

        corrected_accel = self.spacecraft_transform_manager.transform_vector(specific_accel_in_body_from_propagation, BODY_frame, self.frame)
        
        return corrected_accel

class Gyroscope(SensorBase):
    """
    Model a spacecraft gyroscope with configurable measurement errors.

    The gyroscope measures spacecraft angular velocity and transforms it
    into the sensor frame before applying the configured measurement error
    models.

    Parameters
    ----------
    position : np.ndarray or float, optional
        Gyroscope position relative to the spacecraft body frame [m].
    rotation : np.ndarray or float, optional
        Gyroscope orientation relative to the spacecraft body frame [deg].
    bias : np.ndarray or float, optional
        Constant gyroscope bias.
    noise_std : np.ndarray or float, optional
        Standard deviation of white measurement noise.
    random_walk_std : np.ndarray or float, optional
        Random-walk noise parameter.
    sample_rate_freq : float, optional
        Sensor sampling frequency [Hz].
    saturation_limit : np.ndarray or float, optional
        Symmetric measurement saturation limit.
    quantization_bits : int, optional
        Number of quantization bits.
    name : str, optional
        Sensor identifier.
    verbose : bool, optional
        If True, print sensor information during initialization.
    """
    def __init__(self,
                 position: np.ndarray | float = 0.0,
                 rotation: np.ndarray | float = 0.0,
                 bias: np.ndarray | float = 0.0,                 
                 noise_std: np.ndarray | float = 0.0,
                 random_walk_std: np.ndarray | float = 0.0,
                 sample_rate_freq: float = 100.0,
                 saturation_limit: np.ndarray | float = np.inf,
                 quantization_bits: int = 16,
                 name: str = None,
                 verbose: bool = False):                

        error_models = (
            Bias(_as_3d_array(bias, "bias")),
            WhiteNoise(_as_3d_array(noise_std, "noise_std")),
            RandomWalk(_as_3d_array(random_walk_std, "random_walk_std")),
            Saturation(_as_3d_array(saturation_limit, "saturation_limit")),
            Quantization(_as_3d_array(saturation_limit, "saturation_limit"), quantization_bits),
            )
        
        super().__init__(
            type="Gyroscope",
            position=position,
            rotation=rotation,
            sample_rate_freq=sample_rate_freq,
            error_models=error_models,
            name=name,
            verbose=verbose)

    def get_ideal_measurement(
        self        
    ) -> np.ndarray:
        """
        Compute the ideal gyroscope measurement.        

        Returns
        -------
        np.ndarray, shape (3,)
            Spacecraft angular velocity expressed in the sensor frame [rad/s].
        """

        true_state = self.spacecraft_data.true_state
        rotational_frame = true_state.rotational_frame

        if rotational_frame == NONE_frame:
            return np.zeros(3)

        omega_sensor = self.spacecraft_transform_manager.transform_vector(
            true_state.angular_velocity,
            rotational_frame,
            self.frame,
        )

        return omega_sensor