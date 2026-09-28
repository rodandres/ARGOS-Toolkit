import numpy as np
from py.modules.faults.fault_manager import FaultMode

class SensorStuck(FaultMode):

    def __init__(self):
        super().__init__()
        self.stuck_value = None

    def on_activate(self):
        self.stuck_value = None

    def on_deactivate(self):
        self.stuck_value = None

    def apply(self, value, context=None):

        if self.stuck_value is None:
            self.stuck_value = np.copy(value)

        return np.copy(self.stuck_value)

    