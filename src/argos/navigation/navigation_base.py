from abc import ABC, abstractmethod

from argos.sensors.sensor_base import SensorBase
from argos.general.dataclasses import NavigationOutput

class NavigationBase(ABC):
    """
    Abstract base class for spacecraft navigation models.

    Navigation models process sensor measurements to estimate the spacecraft
    state and, when applicable, the state of a reference or target
    spacecraft.
    """
    def __init__(self):
        self._check_initialization()

    @abstractmethod
    def _check_initialization(self):
        """
        Validate the navigation model initialization.

        Subclasses implement this method to verify that their configuration is
        valid.

        Raises
        ------
        Exception
            Subclasses may raise an appropriate exception when the navigation
            model configuration is invalid.
        """
        pass

    @abstractmethod
    def estimate(self, sensors: list[SensorBase]) -> NavigationOutput:
        """
        Estimate spacecraft state from sensor measurements.

        Parameters
        ----------
        sensors : list of SensorBase
            Sensors available to the navigation model.

        Returns
        -------
        NavigationOutput
            Estimated spacecraft state and, when provided by the navigation
            model, the reference or target state.

        Raises
        ------
        NotImplementedError
            Subclasses must implement this method.
        """
        pass