from abc import ABC, abstractmethod

from py.modules.sensors.sensor_base import SensorBase
from py.general.dataclasses import EstimationOutput

class NavigationBase(ABC):

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