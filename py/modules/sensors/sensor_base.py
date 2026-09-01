from abc import ABC, abstractmethod

import numpy as np
from py.modules.math import quaternion_from_euler, quaternion_to_DCM
from py.modules.general_tools import _as_3d_array

class SensorBase(ABC):
    """
    Abstract interface for spacecraft sensors.
    """
    def __init__(self,
                 sensor_type: str,
                 sensor_pos: np.ndarray,
                 sensor_rotation: np.ndarray,
                 sample_rate_freq: float = 100.0,
                 error_models: tuple = (),
                 verbose: bool = False):
        
        
        self.type = sensor_type
        self.position = _as_3d_array(sensor_pos, "sensor_pos") # Position of the sensor in the spacecraft body frame (numpy array of shape (3,))
        self.rotation = _as_3d_array(sensor_rotation, "sensor_rotation") # Rotation of the sensor in the spacecraft body frame (numpy array of shape (3,))
        self.sample_rate_sec = 1.0 / sample_rate_freq
        self.error_models = error_models  # Tuple of error models to apply to the sensor measurement
        self.verbose = verbose

        sensor_pos = _as_3d_array(sensor_pos, "sensor_pos")
        sensor_rotation = _as_3d_array(sensor_rotation, "sensor_rotation")
        sensor_rotation_rad = np.radians(sensor_rotation)

        self.rotation_quat = quaternion_from_euler(sensor_rotation_rad[0], sensor_rotation_rad[1], sensor_rotation_rad[2])  # Quaternion representing the sensor rotation in the body frame
        self.DCM_sensor_to_body = quaternion_to_DCM(self.rotation_quat)  # Direction Cosine Matrix from sensor frame to body frame

        self.old_measurement = np.zeros(3)
        self.last_measurement_time = -self.sample_rate_sec  # Initialize to ensure the first measurement is taken at t=0

        self.measurement = np.zeros(3)

        if self.verbose:
            self.print_info()

    def print_info(self):
        """
        Print the sensor information.
        """
        print("="*50)
        print(f"Sensor Type: {self.type}")
        print(f"Sensor Position (Body Frame): {self.position}")
        print(f"Sensor Rotation (Body Frame): {self.rotation}")
        print(f"Sample Rate (Hz): {1.0 / self.sample_rate_sec}")
        print(f"Error Models: {[type(model).__name__ for model in self.error_models]}")
        print("="*50)

    def get_type(self) -> str:
        """
        Get the type of the sensor.

        Returns
        -------
        str
            Sensor type.
        """        
        return self.type
    
    def update(self, spacecraft_data, simulation_data):
        """
        Review and update the sensor state based on the spacecraft data and simulation data.

        Parameters
        ----------
        spacecraft_data : SpacecraftData
            The current state of the spacecraft.
        simulation_data : SimulationData
            The current state of the simulation.            
        """

        if self.verbose: print("Sensor Update: ", self.type, " at time: ", spacecraft_data.t)
        if spacecraft_data.t - self.last_measurement_time >= self.sample_rate_sec:
            if self.verbose: print("Sensor Sampling: ", self.type, " at time: ", spacecraft_data.t)

            measurement = self.get_ideal_measurement(spacecraft_data, simulation_data)
            
            for error_model in self.error_models:
                measurement = error_model.apply(measurement, dt=spacecraft_data.current_propagation_dt)
    
            self.old_measurement = measurement
            self.last_measurement_time = spacecraft_data.t
    
            self.measurement = measurement
        else:
            if self.verbose: print("Sensor Not Sampling: ", self.type, " at time: ", spacecraft_data.t)
            self.should_sample = False
            self.measurement = self.old_measurement
        
    def get_measurement(self) -> np.ndarray:
        return self.measurement
        
        

    @abstractmethod
    def get_ideal_measurement(self, spacecraft_data, simulation_data) -> np.ndarray:
        """
        Get the ideal sensor measurement without any noise or errors.

        Parameters
        ----------
        spacecraft_data : SpacecraftData
            The current state of the spacecraft.
        simulation_data : SimulationData
            The current state of the simulation.

        Returns
        -------
        np.ndarray
            Ideal sensor measurement.
        """
        pass