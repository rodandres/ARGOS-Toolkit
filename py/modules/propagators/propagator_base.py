from abc import ABC, abstractmethod

class TranslationalPropagatorBase(ABC):

    def __init__(self, integration_method: str = "NATIVE_RK45"):        
        self.integration_method = integration_method

    @abstractmethod
    def propagate(self, spacecraft_data, simulation_data, environment):
        """
        Propagate the translational state forward in time by dt.

        Parameters:
        simulation_data: The current simulation data.

        Returns:
        The new translational state after propagation.
        """
        pass

    def initialize(self, simulation_data):
        """
        Initialize the translational propagator with the given simulation data.

        Parameters:
        simulation_data: The data required to initialize the translational propagator.
        """
        pass

class RotationalPropagatorBase(ABC):

    def __init__(self, integration_method: str = "NATIVE_RK45"):
            self.integration_method = integration_method

    @abstractmethod
    def propagate(self, spacecraft_data, simulation_data, environment):
        pass


    def initialize(self, simulation_data):
        """
        Initialize the rotational propagator with the given simulation data.

        Parameters:
        simulation_data: The data required to initialize the rotational propagator.
        """
        pass