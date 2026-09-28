import numpy as np
from py.modules.enviroments.environment_base import EnvironmentBase

class ClassicalEnvironment(EnvironmentBase):
    def __init__(self):
        super().__init__()

    def get_perturbation_torque(self):
        return np.array([0.0, 0.0, 0.0])  # No perturbation torque in classical environment