from abc import ABC, abstractmethod

import numpy as np
from argos.math import quaternion_from_euler
from argos.general_tools import _as_3d_array
from argos.faults.fault_manager import FaultInjector

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from argos.general.dataclasses import SimulationData, SpacecraftData
    from argos.core.spacecraft import Spacecraft
    from argos.frames.transform_manager import TransformsManager

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

        self.frame = None
        
        rotation_rad = np.radians(rotation)

        self.rotation_quat = quaternion_from_euler(rotation_rad[0], rotation_rad[1], rotation_rad[2])  # Quaternion representing the sensor rotation in the body frame

        self.old_measurement = np.zeros(3)
        self.last_measurement_time = -self.sample_rate_sec  # Initialize to ensure the first measurement is taken at t=0

        self.measurement = np.zeros(3)

        self.fault_injector = FaultInjector()

        self.parent_spacecraft:Spacecraft = None  # Reference to the parent spacecraft, set during spacecraft initialization
        self.spacecraft_data: SpacecraftData = None  # Reference to the spacecraft data object, set during spacecraft initialization
        self.spacecraft_transform_manager: TransformsManager = None  # Reference to the spacecraft transform manager, set during spacecraft initialization
        self.simulation_data: SimulationData = None  # Reference to the simulation data object, set during spacecraft initialization

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
    
    def update(self):
        """
        Update the sensor measurement according to its sampling rate.

        When a measurement is due, the ideal measurement is generated, sensor
        error models are applied sequentially, and active fault modes are then
        applied. Between sampling instants, the previous measurement is
        retained.        
        """        

        if self.verbose: print("Sensor Update: ", self.type, " at time: ", self.spacecraft_data.t)
        if self.spacecraft_data.t - self.last_measurement_time >= self.sample_rate_sec:
            if self.verbose: print("Sensor Sampling: ", self.type, " at time: ", self.spacecraft_data.t)

            measurement = self.get_ideal_measurement()
            for error_model in self.error_models:
                measurement = error_model.apply(measurement, dt=self.spacecraft_data.current_propagation_dt)


            context = {
                "spacecraft_data": self.spacecraft_data,
                "simulation_data": self.simulation_data,
                "type": self.type,                
            }

            measurement = self.fault_injector.apply(measurement,
                                                    context=context)
    
            self.old_measurement = measurement
            self.last_measurement_time = self.spacecraft_data.t
    
            self.measurement = measurement
        else:
            if self.verbose: print("Sensor Not Sampling: ", self.type, " at time: ", self.spacecraft_data.t)
            self.should_sample = False
            self.measurement = self.old_measurement
        
    def get_measurement(self) -> np.ndarray: # NOTE: To be deleted
        return self.measurement
        
        

    @abstractmethod
    def get_ideal_measurement(self) -> np.ndarray:
        """
        Compute the ideal sensor measurement before error and fault injection.        

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