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
                       mass: float | None=None, # NOTE: Add the type of the mass               
                       initial_position: np.ndarray | None=None, # NOTE: Add the type of the initial position
                       initial_velocity: np.ndarray | None=None, # NOTE: Add the type of the initial
                       initial_attitude: np.ndarray | None=None, # NOTE: Add the type of the initial attitude
                       initial_angular_velocity: np.ndarray | None=None, # NOTE: Add the type of
                       inertia_tensor: np.ndarray | None=None,
                       actuators: list | None=None, # NOTE: Add the type of the list
                       sensors: list | None=None, # NOTE: Add the type of the list
                       mission_manager: object | None=None, # NOTE: Add the type of the mission manager
                       target_name: str | None=None, # NOTE: Add the type of the target name
                       name: str |None=None,
                       verbose=False):
        

        if mass is None:
            if verbose: print("Mass not provided. Using default mass of 0 kg.")
            mass = 0.0  # Default mass

        if initial_position is None:
            if verbose: print("Initial position not provided. Using default position of [0, 0, 0].")
            initial_position = np.zeros(3)  # Default position

        if initial_velocity is None:
            if verbose: print("Initial velocity not provided. Using default velocity of [0, 0, 0].")
            initial_velocity = np.zeros(3)  # Default velocity

        if initial_attitude is None:
            if verbose: print("Initial attitude not provided. Using default quaternion of [0, 0, 0, 1].")
            initial_attitude = np.array([0, 0, 0, 1])  # Default quaternion

        if initial_angular_velocity is None:
            if verbose: print("Initial angular velocity not provided. Using default angular velocity of [0, 0, 0].")
            initial_angular_velocity = np.zeros(3)  # Default angular velocity

        if inertia_tensor is None:
            if verbose: print("Inertia tensor not provided. Using default inertia tensor of identity matrix of zeros.")
            inertia_tensor = np.zeros((3, 3))  # Default inertia tensor

        if mission_manager is None:
            if verbose: print("Mission manager not provided. Using default mission manager of None.")
            mission_manager = None  # Default mission manager

        if name is None:
            name = f"Spacecraft_{len(self.simulation_data.spacecrafts)+1}"        

        initial_state = np.concatenate((initial_position, initial_velocity, initial_attitude, initial_angular_velocity))

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

        if target_name is not None:
            self.add_spacecraft_target(spacecraft_name=name, target_name=target_name)
            if self.verbose: print(f"Target '{target_name}' assigned to spacecraft '{name}'.")

        if self.verbose:
            print(f"Spacecraft '{name}' added to the simulation.")

    def add_spacecraft_target(self, spacecraft_name, target_name):
        for spacecraft in self.simulation_data.spacecrafts:
            if spacecraft.spacecraft_data.name == spacecraft_name:
                spacecraft.spacecraft_data.target_name = target_name
                if self.verbose:
                    print(f"Target '{target_name}' assigned to spacecraft '{spacecraft_name}'.")
                return


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