from abc import ABC, abstractmethod
import numpy as np
from py.general.data_classes_declaration import SimSharedData
from py.modules.math import quaternion_error as quat_error

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from py.general.dataclasses import SimulationData, ControlOutput, GuidanceReference, EstimationOutput
    from py.modules.actuators.actuators_base import ActuatorBase


class ControllerBase(ABC):
    """
    Abstract interface for spacecraft control laws.
    """
    
    def compute_control_old(
        self,
        shared_data: SimSharedData,
    ) -> np.ndarray:
        """
        Compute the commanded control torque.

        Parameters
        ----------
        shared_data : SimSharedData
            Shared simulation data.

        Returns
        -------
        np.ndarray
            Commanded control torque expressed in the body frame.
        """
        pass

    @abstractmethod
    def compute_control(self, estimated_state: EstimationOutput, reference: GuidanceReference) -> ControlOutput:
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