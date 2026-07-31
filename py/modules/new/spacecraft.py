import warnings
import numpy as np

from py.general.dataclasses import SpacecraftData
from py.modules.new.mission_manager import MissionManager
from py.modules.sensors.sensor_base import SensorBase
from py.modules.actuators.actuators_base import ActuatorBase

from typing import TYPE_CHECKING
if TYPE_CHECKING:    
    from py.general.dataclasses import MissionPhase

class Spacecraft():

    def __init__(self, name: str, mass: float, initial_state: np.ndarray, inertia_tensor: np.ndarray, 
                 actuators: list | None = None, sensors: list[SensorBase] | None = None,
                 mission_manager: MissionManager | None = None,
                 verbose: bool = False):                        

        # --- ATRIBUTES DECLARATION ---
        self.name = name
        self.mass = mass
        self.inertia_tensor = inertia_tensor
        self.inertia_tensor_inv = np.linalg.inv(inertia_tensor)

        self.spacecraft_data = SpacecraftData()

        self.sensors = sensors
        self.actuators = actuators        

        self.mission_manager = mission_manager

        self.spacecraft_data.target_name = None  # Name of the target spacecraft, if any

        self.current_phase = None
        self.current_guidance_law = None
        self.current_navigation_law = None
        self.current_control_law = None
        self.current_allocator = None

        self.current_navigation_dt = np.nan
        self.current_guidance_dt = np.nan
        self.current_control_dt = np.nan
        
        self.has_sensors = False
        self.has_actuators = False
        self.has_mission_manager = False
        self.has_navigation_law = False
        self.has_guidance_law = False
        self.has_control_law = False

        self._init_state_variables(initial_state)
        self._check_initialization()

        self.verbose = verbose

        if self.verbose:
            print(f"="*10 + " Spacecraft created successfully. " + "="*10)
            self.show_spacecraft_basic_info()
            self.show_spacecraft_gnc_info()
            self.show_spacecraft_state_info()
            print("="*10 + " End of spacecraft information. " + "="*10)

    def _init_state_variables(self, initial_state: np.ndarray):
        self.spacecraft_data.true_state.position = initial_state[0:3]
        self.spacecraft_data.true_state.velocity = initial_state[3:6]
        self.spacecraft_data.true_state.attitude = initial_state[6:10]
        self.spacecraft_data.true_state.angular_velocity = initial_state[10:13]

    def _check_initialization(self):
        
        if isinstance(self.sensors, list) and len(self.sensors) > 0 and all(isinstance(sensor, SensorBase) for sensor in self.sensors):
            self.has_sensors = True

        if isinstance(self.actuators, list) and len(self.actuators) > 0 and all (isinstance(actuator, ActuatorBase) for actuator in self.actuators):
            self.has_actuators = True

        if isinstance(self.mission_manager, MissionManager):
            self.has_mission_manager = True

        if self.has_mission_manager:
            self._check_and_update_phase_info(self.mission_manager.current_phase)
    
    def _check_and_update_phase_info(self, phase: MissionPhase):
        self.current_phase = phase

        if phase.guidance is None:
            self.has_guidance_law = False
            self.current_guidance_law = None
            self.current_guidance_dt = np.nan
        else:
            self.has_guidance_law = True
            self.current_guidance_law = phase.guidance
            self.current_guidance_dt = phase.dt_guid

        if phase.navigation is None:
            self.has_navigation_law = False
            self.current_navigation_law = None
            self.current_navigation_dt = np.nan
        else:
            self.has_navigation_law = True
            self.current_navigation_law = phase.navigation
            self.current_navigation_dt = phase.dt_nav

        if phase.controller is None:
            self.has_control_law = False
            self.current_control_law = None
            self.current_allocator = None
            self.current_control_dt = np.nan
        else:
            self.has_control_law = True
            self.current_control_law = phase.controller
            self.current_allocator = phase.allocator
            self.current_control_dt = phase.dt_control

            self.current_allocator.set_actuators(self.actuators)

    def show_spacecraft_basic_info(self):
        print(f"Spacecraft Name: {self.name}")
        print(f"Mass: {self.mass} kg")
        print(f"Inertia Tensor: \n{self.inertia_tensor}")

    def show_spacecraft_gnc_info(self):
        print("NOT IMPLEMENTED: GNC information display is not yet implemented.")

    def show_spacecraft_state_info(self):
        print("NOT IMPLEMENTED: Spacecraft state information display is not yet implemented.")

    def show_spacecraft_reference_info(self):
       print("NOT IMPLEMENTED: Spacecraft reference information display is not yet implemented.")

    def show_spacecraft_all_info(self):
        self.show_spacecraft_basic_info()
        self.show_spacecraft_gnc_info()
        self.show_spacecraft_state_info()
        self.show_spacecraft_reference_info()    
        
    def get_gnc_dts(self):        
        return self.current_navigation_dt, self.current_guidance_dt, self.current_control_dt


    def _should_compute(self, simulation_data, dt_component):
        tick = simulation_data.tick
        dt_master = simulation_data.dt_master

        return tick % int(dt_component / dt_master) == 0

    def update_mission_manager(self, simulation_data):
        if self.has_mission_manager is False:
            return

        phase_changed = self.mission_manager.update(simulation_data)

        if phase_changed:
            self._check_and_update_phase_info(self.mission_manager.current_phase)
    
    def update_sensors(self, simulation_data):
        if self.has_sensors is False:
            return
    
        for sensor in self.sensors:
            sensor.update(self.spacecraft_data, simulation_data)
    
    def update_navigation(self, simulation_data):
        if self.has_navigation_law is False:
            return

        if self._should_compute(simulation_data, self.current_navigation_dt):
            estimation = self.current_navigation_law.estimate(self.sensors)

            self.spacecraft_data.estimated_data = estimation
            
    def update_guidance(self, simulation_data):
        if self.has_guidance_law is False:
            return

        if self._should_compute(simulation_data, self.current_guidance_dt):            
            self.spacecraft_data.reference_data = self.current_guidance_law.compute_reference(
                self.spacecraft_data.estimated_data, simulation_data
            )

    def update_control(self, simulation_data):
        if self.has_control_law is False:
            return

        if self._should_compute(simulation_data, self.current_control_dt):
            self.spacecraft_data.control_output_data = self.current_control_law.compute_control(
                self.spacecraft_data.estimated_data,
                self.spacecraft_data.reference_data
            )

            self.current_allocator.allocate(self.spacecraft_data.control_output_data)

    def compute_actuation(self, simulation_data):
        if self.has_actuators is False:
            return

        # Placeholder for actuation computation logic
        force = np.zeros(3)
        torque = np.zeros(3)

        
        self.spacecraft_data.current_force_exerted = force
        self.spacecraft_data.current_torque_exerted = torque        

        for actuator in self.actuators:
            actuator.update(simulation_data.t)

            actuator_output = actuator.get_output()

            force += actuator_output.force
            torque += actuator_output.torque

        self.spacecraft_data.current_force_exerted = force
        self.spacecraft_data.current_torque_exerted = torque    

    def change_target(self, new_target_name: str):
        self.spacecraft_data.target_name = new_target_name
        print("ENtro acá")
        if self.verbose:
            print(f"Spacecraft '{self.name}' target changed to '{new_target_name}'.")