from abc import ABC, abstractmethod
from copy import deepcopy

class TranslationalPropagatorBase(ABC):
    """
    Abstract base class for spacecraft translational propagators.

    Translational propagators advance the spacecraft position and velocity
    according to a selected dynamics model and applied forces.

    Parameters
    ----------
    integration_method : str, optional
        Numerical integration method used by the propagator.
        Defaults to ``"NATIVE_RK45"``.
    """
    def __init__(self, integration_method: str = "NATIVE_RK45"):        
        self.integration_method = integration_method
        self.using_cpp = False

        if "cpp" in self.integration_method.lower():
            self.using_cpp = True
    

    def copy(self):
        """
        Create an independent copy of the propagator.

        Returns
        -------
        TranslationalPropagatorBase
            Deep copy of the current propagator.
        """
        return deepcopy(self)

    @abstractmethod
    def propagate(self, spacecraft_data, simulation_data, environment):
        """
        Propagate the spacecraft translational state forward in time.

        Parameters
        ----------
        spacecraft_data : SpacecraftData
            Current spacecraft state and simulation timing information.
        simulation_data : SimulationData
            Current simulation data.
        environment : EnvironmentBase
            Environment model used to determine external effects.

        Raises
        ------
        NotImplementedError
            Subclasses must implement the propagation method.
        """
        pass

    def initialize(self, simulation_data):
        """
        Initialize the translational propagator.

        Parameters
        ----------
        simulation_data : SimulationData
            Current simulation data used to initialize the propagator.
        """
        pass

class RotationalPropagatorBase(ABC):
    """
    Abstract base class for spacecraft rotational propagators.

    Rotational propagators advance the spacecraft attitude and angular
    velocity according to the spacecraft inertia and applied torques.

    Parameters
    ----------
    integration_method : str, optional
        Numerical integration method used by the propagator.
        Defaults to ``"NATIVE_RK45"``.
    """
    def __init__(self, integration_method: str = "NATIVE_RK45"):
        self.integration_method = integration_method

        self.using_cpp = False
        if "cpp" in self.integration_method.lower():
            self.using_cpp = True
        

    def copy(self):
        """
        Create an independent copy of the propagator.

        Returns
        -------
        RotationalPropagatorBase
            Deep copy of the current propagator.
        """
        return deepcopy(self)

    @abstractmethod
    def propagate(self, spacecraft_data, simulation_data, environment):
        """
        Propagate the spacecraft rotational state forward in time.

        Parameters
        ----------
        spacecraft_data : SpacecraftData
            Current spacecraft state and simulation timing information.
        simulation_data : SimulationData
            Current simulation data.
        environment : EnvironmentBase
            Environment model used to determine external effects.

        Raises
        ------
        NotImplementedError
            Subclasses must implement the propagation method.
        """
        pass


    def initialize(self, simulation_data):
        """
        Initialize the rotational propagator.

        Parameters
        ----------
        simulation_data : SimulationData
            Current simulation data used to initialize the propagator.
        """
        pass