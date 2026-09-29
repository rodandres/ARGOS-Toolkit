import numpy as np
from argos.sensors.sensor_base import SensorBase
from argos.general_tools import _as_3d_array

from argos.sensors.error_models import *

class Accelerometer(SensorBase):
    """
    Model a spacecraft accelerometer with configurable measurement errors.

    The accelerometer computes specific force, transforms it from the
    inertial frame to the sensor frame, and applies the configured bias,
    white-noise, random-walk, saturation, and quantization models.

    Parameters
    ----------
    sensor_pos : np.ndarray or float, optional
        Accelerometer position relative to the spacecraft body frame [m].
    sensor_rotation : np.ndarray or float, optional
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
                 sensor_pos: np.ndarray | float = 0.0,
                 sensor_rotation: np.ndarray | float = 0.0,
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
            sensor_type="Accelerometer",
            sensor_pos=sensor_pos,
            sensor_rotation=sensor_rotation,
            sample_rate_freq=sample_rate_freq,
            error_models=error_models,
            name=name,
            verbose=verbose)        

    def compute_specific_accel(self, spacecraft_data, simulation_data) -> np.ndarray:
        """
        Compute the specific acceleration measured by the accelerometer.

        Specific acceleration is computed as spacecraft acceleration minus the
        gravitational acceleration provided by the simulation.

        Parameters
        ----------
        spacecraft_data : SpacecraftData
            Current spacecraft state.
        simulation_data : SimulationData
            Current simulation data containing gravitational acceleration.

        Returns
        -------
        np.ndarray, shape (3,)
            Specific acceleration [m/s²].
        """
        spacecraft_accel = spacecraft_data.true_accel
        gravity_accel = simulation_data.gravity_accel

        specific_accel = spacecraft_accel - gravity_accel
        return specific_accel
    
    def transform_to_sensor_frame(
        self,
        accel,
        spacecraft_data,
        simulation_data,
    ) -> np.ndarray:
        """
        Transform an acceleration vector into the accelerometer frame.

        The transformation accounts for the spacecraft attitude and the
        accelerometer position relative to the spacecraft center of mass.

        Parameters
        ----------
        accel : np.ndarray, shape (3,)
            Acceleration vector in the inertial frame [m/s²].
        spacecraft_data : SpacecraftData
            Current spacecraft rotational state.
        simulation_data : SimulationData
            Current simulation data containing the inertial-to-body DCM.

        Returns
        -------
        np.ndarray, shape (3,)
            Acceleration expressed in the sensor frame [m/s²].
        """
        
        # Correction of the accel in the inertial frame to the body frame
        accel_body = simulation_data.DCM_inertial_to_body @ accel

        # Correction of the accel in the body frame to the sensor frame
        accel_body += (np.cross(spacecraft_data.true_alpha, self.sensor_pos) + np.cross(spacecraft_data.true_omega, np.cross(spacecraft_data.true_omega, self.sensor_pos)))
        

        return self.DCM_sensor_to_body.T @ accel_body
            
    def get_ideal_measurement(
        self,
        spacecraft_data,
        simulation_data,
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
        specific_accel = self.compute_specific_accel(spacecraft_data, simulation_data)
        corrected_accel = self.transform_to_sensor_frame(specific_accel, spacecraft_data, simulation_data)
        return corrected_accel

class Gyroscope(SensorBase):
    """
    Model a spacecraft gyroscope with configurable measurement errors.

    The gyroscope measures spacecraft angular velocity and transforms it
    into the sensor frame before applying the configured measurement error
    models.

    Parameters
    ----------
    sensor_pos : np.ndarray or float, optional
        Gyroscope position relative to the spacecraft body frame [m].
    sensor_rotation : np.ndarray or float, optional
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
                 sensor_pos: np.ndarray | float = 0.0,
                 sensor_rotation: np.ndarray | float = 0.0,
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
            sensor_type="Gyroscope",
            sensor_pos=sensor_pos,
            sensor_rotation=sensor_rotation,
            sample_rate_freq=sample_rate_freq,
            error_models=error_models,
            name=name,
            verbose=verbose)

    def transform_to_sensor_frame(self, gyro, simulation_data) -> np.ndarray:
        """
        Transform angular velocity into the gyroscope frame.

        Parameters
        ----------
        gyro : np.ndarray, shape (3,)
            Angular velocity in the inertial frame [rad/s].
        simulation_data : SimulationData
            Current simulation data containing the inertial-to-body DCM.

        Returns
        -------
        np.ndarray, shape (3,)
            Angular velocity expressed in the sensor frame [rad/s].
        """
        # Correction of the gyro in the inertial frame to the body frame
        gyro_body = simulation_data.DCM_inertial_to_body @ gyro        
    
        return self.DCM_sensor_to_body.T @ gyro_body

    def get_ideal_measurement(
        self,
        spacecraft_data,
        simulation_data,
    ) -> np.ndarray:
        """
        Compute the ideal gyroscope measurement.

        Parameters
        ----------
        spacecraft_data : SpacecraftData
            Current spacecraft state.
        simulation_data : SimulationData
            Current simulation data.

        Returns
        -------
        np.ndarray, shape (3,)
            Spacecraft angular velocity expressed in the sensor frame [rad/s].
        """
        # Gyroscope ideally measures the angular velocity of the spacecraft in the body frame
        return self.transform_to_sensor_frame(spacecraft_data.true_omega, simulation_data)