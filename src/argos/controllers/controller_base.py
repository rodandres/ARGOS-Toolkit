from abc import ABC, abstractmethod
from copy import deepcopy
import numpy as np
from argos.math import quaternion_error as quat_error

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from argos.general.dataclasses import ControlOutput, GuidanceOutput, NavigationOutput
    from argos.actuators.actuators_base import ActuatorBase


class ControllerBase(ABC):
    """
    Abstract base class for spacecraft controllers.

    Controllers receive navigation and guidance information and compute a
    control command for the spacecraft.
    """
    def __init__(self):
        self._check_initialization()

    @abstractmethod
    def _check_initialization(self):
        """
        Validate the controller initialization.

        Raises
        ------
        Exception
            Subclasses should raise an appropriate exception when their
            configuration is invalid.
        """
        pass
        
    def copy(self):
        """
        Create an independent copy of the controller.

        Returns
        -------
        ControllerBase
            Deep copy of the current controller instance.
        """
        return deepcopy(self)

   
    @abstractmethod
    def compute_control(self, navigation_output: NavigationOutput, guidance_output: GuidanceOutput) -> ControlOutput:
        """
        Compute the control command from navigation and guidance information.

        Parameters
        ----------
        navigation_output : NavigationOutput
            Current navigation solution for the spacecraft.
        guidance_output : GuidanceOutput
            Desired spacecraft state or guidance information.

        Returns
        -------
        ControlOutput
            Computed spacecraft control command.

        Raises
        ------
        NotImplementedError
            Subclasses must implement this method.
        """
        pass

class ControlAllocatorBase(ABC):
    """
    Abstract base class for spacecraft control allocators.

    Control allocators convert a high-level control command into commands
    for the actuators available on a spacecraft.
    """
    def __init__(self):
        self._check_initialization()

    def copy(self):
        """
        Create an independent copy of the control allocator.

        Returns
        -------
        ControlAllocatorBase
            Deep copy of the current control allocator instance.
        """
        return deepcopy(self)


    @abstractmethod    
    def _check_initialization(self):
        """
        Validate the control allocator initialization.

        Subclasses implement this method to verify that their configuration is
        valid.
        """
        pass

    def set_actuators(self, actuators: list[ActuatorBase]):
        """
        Set the actuators available to the control allocator.

        Parameters
        ----------
        actuators : list of ActuatorBase
            Actuator instances that can receive commands from the allocator.
        """
        self.actuators = actuators

    @abstractmethod
    def allocate(self, control_output: ControlOutput):
        """
        Allocate a control command among the available actuators.

        Parameters
        ----------
        control_output : ControlOutput
            High-level force and torque command to be converted into actuator
            commands.

        Raises
        ------
        NotImplementedError
            Subclasses must implement this method.
        """
        pass