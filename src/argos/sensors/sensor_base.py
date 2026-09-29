from abc import ABC, abstractmethod

import numpy as np
from argos.math import quaternion_from_euler, quaternion_to_DCM
from argos.general_tools import _as_3d_array
from argos.faults.fault_manager import FaultInjector

class SensorBase(ABC):
    """
    Abstract base class for spacecraft sensors.

    A sensor generates an ideal measurement from spacecraft and simulation
    data, applies configured measurement error models, and then applies any
    active fault injection before storing the resulting measurement.

    Parameters
    ----------
    type : str
        Sensor type identifier.
    position : np.ndarray
        Sensor position relative to the spacecraft body frame [m].
    rotation : np.ndarray
        Sensor orientation relative to the spacecraft body frame [deg].
    sample_rate_freq : float, optional
        Sensor sampling frequency [Hz]. Defaults to 100 Hz.
    error_models : tuple, optional
        Sequence of sensor error models applied to each measurement.
    name : str, optional
        Sensor identifier.
    verbose : bool, optional
        If True, print sensor information and update status.
    """
    def __init__(self,
                 type: str,
                 position: np.ndarray,
                 rotation: np.ndarray,
                 sample_rate_freq: float = 100.0,
                 error_models: tuple = (),
                 name: str = None,
                 verbose: bool = False):        
        
        self.type = type
        self.position = _as_3d_array(position, "position") # Position of the sensor in the spacecraft body frame (numpy array of shape (3,))
        self.rotation = _as_3d_array(rotation, "rotation") # Rotation of the sensor in the spacecraft body frame (numpy array of shape (3,))
        self.sample_rate_sec = 1.0 / sample_rate_freq
        self.error_models = error_models  # Tuple of error models to apply to the sensor measurement

        self.name = name

        self.verbose = verbose

        position = _as_3d_array(position, "position")
        rotation = _as_3d_array(rotation, "rotation")
        rotation_rad = np.radians(rotation)

        self.rotation_quat = quaternion_from_euler(rotation_rad[0], rotation_rad[1], rotation_rad[2])  # Quaternion representing the sensor rotation in the body frame
        self.DCM_sensor_to_body = quaternion_to_DCM(self.rotation_quat)  # Direction Cosine Matrix from sensor frame to body frame

        self.old_measurement = np.zeros(3)
        self.last_measurement_time = -self.sample_rate_sec  # Initialize to ensure the first measurement is taken at t=0

        self.measurement = np.zeros(3)

        self.fault_injector = FaultInjector()

        if self.verbose:
            self.print_info()

    def print_info(self):
        """
        Print the sensor configuration and current setup information.
        """
        print("="*50)
        print(f"Sensor Name: {self.name}")
        print(f"Sensor Type: {self.type}")        
        print(f"Sensor Position (Body Frame): {self.position}")
        print(f"Sensor Rotation (Body Frame): {self.rotation}")
        print(f"Sample Rate (Hz): {1.0 / self.sample_rate_sec}")
        print(f"Error Models: {[type(model).__name__ for model in self.error_models]}")
        print("="*50)    

    def get_type(self) -> str: # NOTE: To be deleted
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
        Update the sensor measurement according to its sampling rate.

        When a measurement is due, the ideal measurement is generated, sensor
        error models are applied sequentially, and active fault modes are then
        applied. Between sampling instants, the previous measurement is
        retained.

        Parameters
        ----------
        spacecraft_data : SpacecraftData
            Current spacecraft state and simulation timing information.
        simulation_data : SimulationData
            Current simulation data.
        """

        if self.verbose: print("Sensor Update: ", self.type, " at time: ", spacecraft_data.t)
        if spacecraft_data.t - self.last_measurement_time >= self.sample_rate_sec:
            if self.verbose: print("Sensor Sampling: ", self.type, " at time: ", spacecraft_data.t)

            measurement = self.get_ideal_measurement(spacecraft_data, simulation_data)
            
            for error_model in self.error_models:
                measurement = error_model.apply(measurement, dt=spacecraft_data.current_propagation_dt)


            context = {
                "spacecraft_data": spacecraft_data,
                "simulation_data": simulation_data,
                "type": self.type,                
            }

            measurement = self.fault_injector.apply(measurement,
                                                    context=context)


            self.old_measurement = measurement
            self.last_measurement_time = spacecraft_data.t
    
            self.measurement = measurement
        else:
            if self.verbose: print("Sensor Not Sampling: ", self.type, " at time: ", spacecraft_data.t)
            self.should_sample = False
            self.measurement = self.old_measurement
        
    def get_measurement(self) -> np.ndarray: # NOTE: To be deleted
        return self.measurement
        
        

    @abstractmethod
    def get_ideal_measurement(self, spacecraft_data, simulation_data) -> np.ndarray:
        """
        Compute the ideal sensor measurement before error and fault injection.

        Parameters
        ----------
        spacecraft_data : SpacecraftData
            Current spacecraft state.
        simulation_data : SimulationData
            Current simulation data.

        Returns
        -------
        np.ndarray
            Ideal sensor measurement.

        Raises
        ------
        NotImplementedError
            Subclasses must implement this method.
        """
        pass