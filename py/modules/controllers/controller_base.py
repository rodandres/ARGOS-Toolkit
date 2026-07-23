from abc import ABC, abstractmethod
import numpy as np
from py.general.data_classes_declaration import SimSharedData
from py.modules.math import quaternion_error as quat_error

class ControllerBase(ABC):
    """
    Abstract interface for spacecraft control laws.
    """

    @abstractmethod
    def compute_control(
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
