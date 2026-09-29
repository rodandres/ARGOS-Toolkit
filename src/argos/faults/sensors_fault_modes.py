import numpy as np
from argos.faults.fault_manager import FaultMode

class SensorStuck(FaultMode):
    """
    Fault mode that holds a sensor output at a fixed value.

    When activated, the first value received by the fault mode is stored and
    subsequently returned for all following applications until the fault is
    deactivated.
    """
    def __init__(self):
        super().__init__()
        self.stuck_value = None

    def on_activate(self):
        """
        Reset the stored sensor value when the fault is activated.
        """
        self.stuck_value = None

    def on_deactivate(self):
        """
        Clear the stored sensor value when the fault is deactivated.
        """
        self.stuck_value = None

    def apply(self, value, context=None):
        """
        Apply the stuck-sensor fault to an input value.

        The first value received after activation is stored and returned for
        subsequent calls until the fault is deactivated.

        Parameters
        ----------
        value : np.ndarray
            Sensor value to be affected by the fault.
        context, optional
            Additional context available to the fault mode. It is not used by
            this fault mode.

        Returns
        -------
        np.ndarray
            Copy of the stored sensor value.
        """

        if self.stuck_value is None:
            self.stuck_value = np.copy(value)

        return np.copy(self.stuck_value)

    