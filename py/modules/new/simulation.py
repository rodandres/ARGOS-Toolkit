# NOTAS DE IMPLEMENTACION:
# Se debe añadir una verificacion antes de correr la sim garantizando que se tengan spacecrafts
# Se debe añadir una verificacion o manejo de los propagadores en caso de ser None

from dataclasses import dataclass, field
from pathlib import Path
import sys
from tabnanny import verbose

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from py.modules.new.spacecraft import Spacecraft
from py.general.dataclasses import SimulationData
from py.general.data_save import SimulationHistory

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from py.modules.propagators.propagator_base import TranslationalPropagatorBase, RotationalPropagatorBase
    from py.modules.enviroments.environment_base import EnvironmentBase

class Simulation:
    def __init__(self, max_sim_time: float, dt_propagation: float,
                 environment: EnvironmentBase,
                 rotational_propagator_engine: RotationalPropagatorBase | None=None,
                 translational_propagator_engine: TranslationalPropagatorBase | None=None,
                 verbose: bool = False):        

        simulation_data = SimulationData(
            max_sim_time=max_sim_time,
            dt_propagation=dt_propagation,            
            spacecrafts=[],
            dt_master=dt_propagation  # Initialize dt_master with dt_propagation; will be updated later
            )

        self.simulation_data = simulation_data

        self.rotational_propagator_engine = rotational_propagator_engine
        self.translational_propagator_engine = translational_propagator_engine

        self.environment = environment

        self.simulation_history = SimulationHistory()

        self.verbose = verbose

    def add_spacecraft(self,
                       mass: float,
                       initial_state: np.ndarray, # pos, vel, quat, omega in inertial frame (numpy array of shape (13,))
                       inertia_tensor: np.ndarray,
                       actuators: list, # NOTE: Add the type of the list
                       sensors: list, # NOTE: Add the type of the list
                       mission_manager, # NOTE: Add the type of the mission manager
                       name: str |None=None,
                       verbose=False):
        

        # Check mass type
        if not isinstance(mass, (int, float)):
            raise TypeError("Spacecraft mass must be a number (int or float).")
        
        # Check inertia tensor type and shape
        if not isinstance(inertia_tensor, np.ndarray):
            raise TypeError("Spacecraft inertia tensor must be a numpy ndarray.")

        if inertia_tensor.shape != (3, 3):
            raise ValueError("Spacecraft inertia tensor must be a 3x3 matrix.")

        # Check actuators, sensors, controllers, and guidance types
        # TODO: Implement type checks for actuators, sensors, controllers, and guidance if specific types are defined.

        # Check initial state type and shape
        if not isinstance(initial_state, np.ndarray):
            raise TypeError("Initial state must be a numpy ndarray.")
        if initial_state.shape != (13,):
            raise ValueError("Initial state must be a numpy array of shape (13,) representing pos, vel, quat, omega in inertial frame.")


        if name is None:
            name = f"Spacecraft_{len(self.simulation_data.spacecrafts)+1}"

        spacecraft = Spacecraft(
            name=name,
            mass=mass,
            initial_state=initial_state,
            inertia_tensor=inertia_tensor,
            actuators=actuators,
            sensors=sensors,            
            mission_manager=mission_manager,            
            verbose=verbose
        )

        self.simulation_data.spacecrafts.append(spacecraft)

        if self.verbose:
            print(f"Spacecraft '{name}' added to the simulation.")

    def __set_dt_master(self): # POSSIBLE BUG: dts must be with a minimum common multiple
        dts = [self.simulation_data.dt_propagation]  # Start with the master propagation time step
        for spacecraft in self.simulation_data.spacecrafts:
            dt_nav, dt_guid, dt_control = spacecraft.get_gnc_dts()
            dts.append(dt_nav)
            dts.append(dt_guid)
            dts.append(dt_control)

        self.simulation_data.dt_master = min(dts)

    def __init_simulation(self):
        if self.verbose:
            print("Initializing simulation...")
        
        self.__set_dt_master()
    
        if self.verbose:
            print("Simulation initialized.")
    
    def simulate(self):
        self.__init_simulation()

        if self.verbose:
            print("Starting simulation...")

        while self.simulation_data.t < self.simulation_data.max_sim_time:
            
            self.simulation_data.t = self.simulation_data.tick * self.simulation_data.dt_master

            #if self.verbose: print(f"Simulation time: {self.simulation_data.t:.2f} seconds")

            for spacecraft in self.simulation_data.spacecrafts:
                spacecraft.compute_tick_step(self.simulation_data)

            if self.simulation_data.tick % int(self.simulation_data.dt_propagation / self.simulation_data.dt_master) == 0:
                if self.translational_propagator_engine is not None:
                    self.translational_propagator_engine.propagate(self.simulation_data, self.environment)  # Assuming environment is not needed for translational propagation
                if self.rotational_propagator_engine is not None:                    
                    self.rotational_propagator_engine.propagate(self.simulation_data, self.environment)  # Assuming environment is not needed for rotational propagation

            # Save the current state to history
            self.simulation_history.record(self.simulation_data)
            self.simulation_data.tick += 1        
        
        if self.verbose: print("Simulation completed.")

        self.simulation_history.finalize()  # Finalize the history after the simulation is complete

        return self.simulation_history