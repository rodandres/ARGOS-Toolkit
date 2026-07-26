from dataclasses import dataclass, field
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from py.modules.new.spacecraft import Spacecraft
from py.general.dataclasses import SimulationData

class Simulation:
    def __init__(self, max_sim_time: float, dt_propagation: float, verbose: bool = False):        

        simulation_data = SimulationData(
            max_sim_time=max_sim_time,
            dt_propagation=dt_propagation,
            verbose=verbose
        )

        self.simulation_data = simulation_data

        
    def add_spacecraft(self,
                       spacecraft_mass: float,
                       spacecraft_inertia_tensor: np.ndarray,
                       spacecraft_actuators: list, # NOTE: Add the type of the list
                       spacecraft_sensors: list, # NOTE: Add the type of the list
                       spacecraft_controllers: list, # NOTE: Add the type of the list
                       spacecraft_guidance: list, # NOTE: Add the type of the list
                       spacecraft_dt_nav: float,
                       spacecraft_dt_guid: float,
                       spacecraft_dt_control: float,
                       initial_state: np.ndarray, # pos, vel, quat, omega in inertial frame (numpy array of shape (13,))
                       name: str |None=None,
                       verbose=False):
        

        # Check mass type
        if not isinstance(spacecraft_mass, (int, float)):
            raise TypeError("Spacecraft mass must be a number (int or float).")
        
        # Check inertia tensor type and shape
        if not isinstance(spacecraft_inertia_tensor, np.ndarray):
            raise TypeError("Spacecraft inertia tensor must be a numpy ndarray.")

        if spacecraft_inertia_tensor.shape != (3, 3):
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
            mass=spacecraft_mass,
            inertia_tensor=spacecraft_inertia_tensor,
            actuators=spacecraft_actuators,
            sensors=spacecraft_sensors,
            controllers=spacecraft_controllers,
            guidance=spacecraft_guidance,
            dt_nav=spacecraft_dt_nav,
            dt_guid=spacecraft_dt_guid,
            dt_control=spacecraft_dt_control,
            initial_state=initial_state,
            verbose=verbose
        )

        self.simulation_data.spacecrafts[name] = {
            "spacecraft": spacecraft
        }

        if self.simulation_data.verbose:
            print(f"Spacecraft '{name}' added to the simulation.")

    def __set_dt_master(self): # POSSIBLE BUG: dts must be with a minimum common multiple
        dts = [self.simulation_data.dt_propagation]  # Start with the master propagation time step
        for spacecraft in self.simulation_data.spacecrafts.values():
            dt_nav, dt_guid, dt_control = spacecraft.get_gnc_dts()
            dts.append((dt_nav, dt_guid, dt_control))

        self.simulation_data.dt_master = min(dts)

    def __set_solver(self):
        # Placeholder for solver selection logic
        # This method can be expanded to select and configure the appropriate numerical solver based on simulation settings.
        pass
        
    def __init_simulation(self):
        if self.simulation_data.verbose:
            print("Initializing simulation...")
        

        self.__set_dt_master()
        self.__set_solver()
    

        if self.simulation_data.verbose:
            print("Simulation initialized.")

    def __propagate_dynamics(self):
        # NOTE: This method should implement the dynamics propagation logic for the simulation.
        pass

    def simulate(self):
        self.__init_simulation()

        if self.simulation_data.verbose:
            print("Starting simulation...")

        while self.simulation_data.t < self.simulation_data.max_sim_time:
            
            self.simulation_data.t = self.simulation_data.tick * self.simulation_data.dt_master

            for spacecraft in self.simulation_data.spacecrafts:
                spacecraft.compute_tick_step(self.simulation_data)

            if self.simulation_data.tick % self.simulation_data.dt_propagation == 0:
                self.__propagate_dynamics()

            self.simulation_data.tick += 1

        # TODO: Add logging and history recording here if needed.
        

        if self.simulation_data.verbose:
            print("Simulation completed.")
