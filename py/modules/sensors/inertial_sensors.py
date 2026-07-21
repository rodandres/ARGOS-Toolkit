import numpy as np
from py.modules.sensors.sensor_base import SensorBase
from py.modules.general_tools import _as_3d_array

from py.modules.sensors.error_models import *

class Accelerometer(SensorBase):

    def __init__(self,
                 sensor_pos: np.ndarray | float = 0.0,
                 sensor_rotation: np.ndarray | float = 0.0,
                 bias: np.ndarray | float = 0.0,                 
                 noise_std: np.ndarray | float = 0.0,
                 random_walk_std: np.ndarray | float = 0.0,
                 sample_rate_freq: float = 100.0,
                 saturation_limit: np.ndarray | float = np.inf,
                 quantization_levels: int | None = 2**16,
                 verbose: bool = False):
                
        error_models = (
            Bias(_as_3d_array(bias, "bias")),
            WhiteNoise(_as_3d_array(noise_std, "noise_std")),
            RandomWalk(_as_3d_array(random_walk_std, "random_walk_std")),
            Saturation(_as_3d_array(saturation_limit, "saturation_limit")),
            Quantization(quantization_levels),
            )
        
        super().__init__(
            sensor_type="Accelerometer",
            sensor_pos=sensor_pos,
            sensor_rotation=sensor_rotation,
            sample_rate_freq=sample_rate_freq,
            error_models=error_models,
            verbose=verbose)        

    def compute_specific_accel(self, spacecraft_data, simulation_data) -> np.ndarray:
        spacecraft_accel = spacecraft_data.true_accel
        gravity_accel = simulation_data.gravity_accel

        specific_accel = spacecraft_accel - gravity_accel
        return specific_accel
    
    def transform_to_sensor_frame(self, accel, spacecraft_data, simulation_data) -> np.ndarray:
        
        # Correction of the accel in the inertial frame to the body frame
        accel_body = simulation_data.DCM_inertial_to_body @ accel

        # Correction of the accel in the body frame to the sensor frame
        accel_body += (np.cross(spacecraft_data.true_alpha, self.sensor_pos) + np.cross(spacecraft_data.true_omega, np.cross(spacecraft_data.true_omega, self.sensor_pos)))
        

        return self.DCM_sensor_to_body.T @ accel_body
            
    def get_ideal_measurement(self, spacecraft_data, simulation_data) -> np.ndarray:
        specific_accel = self.compute_specific_accel(spacecraft_data, simulation_data)
        corrected_accel = self.transform_to_sensor_frame(specific_accel, spacecraft_data, simulation_data)
        return corrected_accel

class Gyroscope(SensorBase):
    
    def __init__(self,
                 sensor_pos: np.ndarray | float = 0.0,
                 sensor_rotation: np.ndarray | float = 0.0,
                 bias: np.ndarray | float = 0.0,                 
                 noise_std: np.ndarray | float = 0.0,
                 random_walk_std: np.ndarray | float = 0.0,
                 sample_rate_freq: float = 100.0,
                 saturation_limit: np.ndarray | float = np.inf,
                 quantization_levels: int | None = 2**16,
                 verbose: bool = False):                

        error_models = (
            Bias(_as_3d_array(bias, "bias")),
            WhiteNoise(_as_3d_array(noise_std, "noise_std")),
            RandomWalk(_as_3d_array(random_walk_std, "random_walk_std")),
            Saturation(_as_3d_array(saturation_limit, "saturation_limit")),
            Quantization(quantization_levels),
            )
        
        super().__init__(
            sensor_type="Gyroscope",
            sensor_pos=sensor_pos,
            sensor_rotation=sensor_rotation,
            sample_rate_freq=sample_rate_freq,
            error_models=error_models,
            verbose=verbose)

    def transform_to_sensor_frame(self, gyro, simulation_data) -> np.ndarray:
        # Correction of the gyro in the inertial frame to the body frame
        gyro_body = simulation_data.DCM_inertial_to_body @ gyro        
    
        return self.DCM_sensor_to_body.T @ gyro_body

    def get_ideal_measurement(self, spacecraft_data, simulation_data) -> np.ndarray:
        # Gyroscope ideally measures the angular velocity of the spacecraft in the body frame
        return self.transform_to_sensor_frame(spacecraft_data.true_omega, simulation_data)