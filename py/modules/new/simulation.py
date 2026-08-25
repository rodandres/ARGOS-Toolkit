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
    def __init__(self, max_sim_time: float,
                 environment: EnvironmentBase,                 
                 verbose: bool = False):        

        simulation_data = SimulationData(
            max_sim_time=max_sim_time,                    
            spacecrafts=[],            
            )

        self.simulation_data = simulation_data        

        self.environment = environment

        self.simulation_data.simulation_history = SimulationHistory()

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
            if verbose: print("Inertia tensor not provided. Using default inertia tensor of identity matrix.")
            inertia_tensor = np.eye(3)  # Default inertia tensor

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

        if self.verbose:
            print(f"Spacecraft '{name}' added to the simulation.")

    def add_spacecraft_target(self, spacecraft_name, target_name):
        for spacecraft in self.simulation_data.spacecrafts:
            if spacecraft.name == spacecraft_name:
                spacecraft.change_target(target_name)                
                if self.verbose:
                    print(f"Target '{target_name}' assigned to spacecraft '{spacecraft_name}'.")
                return

    def _set_dt_master(self): # POSSIBLE BUG: dts must be with a minimum common multiple
        dts = []
        for spacecraft in self.simulation_data.spacecrafts:
            dt_nav, dt_guid, dt_control, dt_propagation = spacecraft.get_dts()
            print(f"Spacecraft '{spacecraft.name}' dt_nav: {dt_nav}, dt_guid: {dt_guid}, dt_control: {dt_control}")
            if not np.isnan(dt_nav):
                dts.append(dt_nav)
            if not np.isnan(dt_guid):
                dts.append(dt_guid)
            if not np.isnan(dt_control):
                dts.append(dt_control)

            dts.append(dt_propagation)  # Add the spacecraft's propagation time step

        self.simulation_data.dt_master = min(dts)

    def _init_simulation(self):
        if self.verbose:
            print("Initializing simulation...")
        
        self._set_dt_master()

        for spacecraft in self.simulation_data.spacecrafts:
            self.simulation_data.simulation_history.record_spacecraft(spacecraft)

    
        if self.verbose:
            print("Simulation initialized.")


    def simulate(self):
        self._init_simulation()

        if self.verbose:
            print("Starting simulation...")


        t = 0.0
        current_ts = []
        self.simulation_data.t = np.nan
        self.simulation_data.tick = np.nan

        while t < self.simulation_data.max_sim_time:

            for spacecraft in self.simulation_data.spacecrafts:
                if spacecraft.spacecraft_data.t >= self.simulation_data.max_sim_time:                    
                    continue  # Skip this spacecraft if its time exceeds the max simulation time

                # Update Mission Manager
                has_mission_changed = spacecraft.update_mission_manager(self.simulation_data)

                # Check for updates in dts due to possible changes in the mission phase
                if has_mission_changed:
                    self._set_dt_master()  # Update the master time step if any spacecraft's mission phase changed

                # Update sensors
                spacecraft.update_sensors(self.simulation_data)

                # Update Navigation
                spacecraft.update_navigation(self.simulation_data)

                # Update Guidance
                spacecraft.update_guidance(self.simulation_data)

                # Update Control
                spacecraft.update_control(self.simulation_data)

                # Compute Actuation
                spacecraft.compute_actuation(self.simulation_data)  

                # Propagate Translational and Rotational Dynamics
                spacecraft.propagate_translational(self.simulation_data, self.environment)
                spacecraft.propagate_rotational(self.simulation_data, self.environment)

                t_spacecraft = spacecraft.spacecraft_data.t
                spacecraft.spacecraft_data.t += spacecraft.spacecraft_data.current_master_dt
                spacecraft.spacecraft_data.tick += 1
                current_ts.append(t_spacecraft)

                self.simulation_data.simulation_history.record_spacecraft(spacecraft)


            try:
                t = min(current_ts)
            except ValueError:
                break  # Exit the loop if there are no spacecrafts to simulate

            current_ts = []  # Reset for the next iteration
            
        
        if self.verbose: print("Simulation completed.")

        self.simulation_data.simulation_history.finalize()  # Finalize the history after the simulation is complete

        return self.simulation_data.simulation_history




        # while self.simulation_data.t < self.simulation_data.max_sim_time:
            
        #     self.simulation_data.t = self.simulation_data.tick * self.simulation_data.dt_master

        #     for spacecraft in self.simulation_data.spacecrafts:

        #         # Update Mission Manager
        #         has_mission_changed = spacecraft.update_mission_manager(self.simulation_data)

        #         # Check for updates in dts due to possible changes in the mission phase
        #         if has_mission_changed:
        #             self._set_dt_master()  # Update the master time step if any spacecraft's mission phase changed

        #         # Update sensors
        #         spacecraft.update_sensors(self.simulation_data)

        #         # Update Navigation
        #         spacecraft.update_navigation(self.simulation_data)

        #         # Update Guidance
        #         spacecraft.update_guidance(self.simulation_data)

        #         # Update Control
        #         spacecraft.update_control(self.simulation_data)

        #         # Compute Actuation
        #         spacecraft.compute_actuation(self.simulation_data)  

        #         # Propagate Translational and Rotational Dynamics
        #         spacecraft.propagate_translational(self.simulation_data, self.environment)
        #         spacecraft.propagate_rotational(self.simulation_data, self.environment)


        #     #if self.simulation_data.tick % int(self.simulation_data.dt_propagation / self.simulation_data.dt_master) == 0:
        #     #    if self.translational_propagator_engine is not None:
        #     #        self.translational_propagator_engine.propagate(self.simulation_data, self.environment)  # Assuming environment is not needed for translational propagation
        #     #    if self.rotational_propagator_engine is not None:                    
        #     #        self.rotational_propagator_engine.propagate(self.simulation_data, self.environment)  # Assuming environment is not needed for rotational propagation

            # Save the current state to history