from abc import ABC, abstractmethod

from py.general.dataclasses import EstimationOutput, SimulationData, GuidanceReference

class GuidanceBase(ABC):

    def __init__(self):
        self._check_initialization()

    @abstractmethod
    def _check_initialization(self):
        """
        Check if the guidance system has been properly initialized.
        Raises an exception if not initialized.
        """        

    @abstractmethod
    def compute_reference(self, navigation_estimated_data: EstimationOutput, simulation_data: SimulationData)->GuidanceReference:
        """
        Abstract method to compute the guidance reference based on navigation estimated data and simulation data.

        Args:
            navigation_estimated_data: The estimated state of the spacecraft from the navigation system.
            simulation_data: The current state of the simulation.
        
        """
        pass