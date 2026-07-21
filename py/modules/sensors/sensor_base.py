from abc import ABC, abstractmethod

import numpy as np


class SensorBase(ABC):
    """
    Abstract interface for spacecraft sensors.
    """
    def __init__(self, verbose: bool = False):
        self.verbose = verbose

        self.sensor_type = ""  # Placeholder for sensor type, to be defined in subclasses

    def get_type(self) -> str:
        """
        Get the type of the sensor.

        Returns
        -------
        str
            Sensor type.
        """        
        return self.sensor_type

    @abstractmethod
    def get_output(self,true_state: np.ndarray, true_ref_state: np.ndarray | None = None) -> np.ndarray:
        """
        Get the sensor output, which may include noise or other effects.

        Parameters
        ----------
        true_state : np.ndarray
            True state vector.

        Returns
        -------
        np.ndarray
            Sensor output.
        """
        return self.measure(true_state)
