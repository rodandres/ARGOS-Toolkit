from abc import ABC, abstractmethod
from copy import deepcopy
import numpy as np
from argos.math import quaternion_error as quat_error

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from argos.general.dataclasses import ControlOutput, GuidanceOutput, NavigationOutput
    from argos.actuators.actuators_base import ActuatorBase


class ControllerBase(ABC):
    def __init__(self):
        self._check_initialization()

    @abstractmethod
    def _check_initialization(self):
        """
        Check if the controller has been properly initialized.
        Raises an exception if not initialized.
        """
        pass
    
    """
    Abstract interface for spacecraft control laws.
    """
    def copy(self):
         return deepcopy(self)

   
    @abstractmethod
    def compute_control(self, navigation_output: NavigationOutput, guidance_output: GuidanceOutput) -> ControlOutput:
        """
        Compute the commanded control torque.

        Parameters
        ----------
        simulation_data : SimulationData
            Current simulation data.

        Returns
        -------
        ControlOutput
            Commanded control torque expressed in the body frame.
        """
        pass

class ControlAllocatorBase(ABC):
    def __init__(self):
        self._check_initialization()

    def copy(self):
         return deepcopy(self)


    @abstractmethod
    def _check_initialization(self):
        """
        Check if the control allocator has been properly initialized.
        Raises an exception if not initialized.
        """
        pass

    def set_actuators(self, actuators: list[ActuatorBase]):
        """
        Add actuators to the control allocator.

        Parameters
        ----------
        actuators : list[ActuatorBase]
            List of actuator instances to be added.
        """
        self.actuators = actuators

    @abstractmethod
    def allocate(self, control_output: ControlOutput):
        pass