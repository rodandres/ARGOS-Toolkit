from abc import ABC, abstractmethod

from py.modules.sensors.sensor_base import SensorBase
from py.general.dataclasses import EstimationOutput

class NavigationBase(ABC):

    def __init__(self):
        self._check_initialization()

    @abstractmethod
    def _check_initialization(self):
        """
        Check if the navigation system has been properly initialized.
        Raises an exception if not initialized.
        """
        pass

    @abstractmethod
    def estimate(self, sensors: list[SensorBase]) -> EstimationOutput:

        """
        Abstract method to estimate the spacecraft's state based on sensor data.

        Args:
            sensors (list[SensorsBase]): A list of sensor objects providing data for estimation.

        Returns:
            estimationOutput: An object containing the estimated state of the spacecraft.
        """
        pass