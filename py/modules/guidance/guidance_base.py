from abc import ABC, abstractmethod

from py.general.dataclasses import EstimationOutput, SimulationData, GuidanceReference

class GuidanceBase(ABC):    

    @abstractmethod
    def compute_reference(self, navigation_estimated_data: EstimationOutput, simulation_data: SimulationData)->GuidanceReference:
        """
        Abstract method to compute the guidance reference based on navigation estimated data and simulation data.

        Args:
            navigation_estimated_data: The estimated state of the spacecraft from the navigation system.
            simulation_data: The current state of the simulation.
        
        """
        pass