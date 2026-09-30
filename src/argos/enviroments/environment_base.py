from abc import ABC, abstractmethod

class EnvironmentBase(ABC):
    @abstractmethod
    def get_perturbation_torque(self):
        pass