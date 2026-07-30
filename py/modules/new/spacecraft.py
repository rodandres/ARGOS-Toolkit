import warnings
import numpy as np

from py.general.dataclasses import SpacecraftData
from py.modules.new.mission_manager import MissionManager
from py.modules.sensors.sensor_base import SensorBase

class Spacecraft():

    def __init__(self, name: str, mass: float, initial_state: np.ndarray, inertia_tensor: np.ndarray, 
                 actuators: list | None = None, sensors: list[SensorBase] | None = None,
                 mission_manager: MissionManager | None = None,
                 verbose: bool = False):

        spacecraft_data = SpacecraftData(
            name=name,

            mass=mass,
            inertia_tensor=inertia_tensor,            
        )

        self.spacecraft_data = spacecraft_data

        # Initialize the spacecraft state variables based on the provided initial state
        self.spacecraft_data.true_pos = initial_state[0:3]
        self.spacecraft_data.true_vel = initial_state[3:6]
        self.spacecraft_data.true_q = initial_state[6:10]
        self.spacecraft_data.true_omega = initial_state[10:13]
        self.verbose = verbose

        # Check and set the GNC components
        self.has_gnc = self.check_gnc_components(mission_manager, actuators, sensors)        

        if self.verbose:
            print(f"="*10 + " Spacecraft created successfully. " + "="*10)
            self.show_spacecraft_basic_info()
            self.show_spacecraft_gnc_info()
            self.show_spacecraft_state_info()
            print("="*10 + " End of spacecraft information. " + "="*10)            

    def check_gnc_components(self, mission_manager, actuators, sensors):
        has_mission_manager = mission_manager is not None
        has_actuators = actuators is not None and len(actuators) > 0
        has_sensors = sensors is not None and len(sensors) > 0
        
        self.actuators = actuators
        self.sensors = sensors

        self.has_sensors = has_sensors

        if has_mission_manager and not has_actuators:
            raise ValueError("Mission manager is present, but no actuators are defined.")
        if has_mission_manager and not has_sensors:
            raise ValueError("Mission manager is present, but no sensors are defined.")

        if has_actuators and not has_mission_manager:
            warnings.warn("Actuators are defined, but no mission manager is present. The spacecraft may not be able to perform any actions.")

        if has_sensors and not has_mission_manager:
            warnings.warn("Sensors are defined, but no mission manager is present. The spacecraft may not be able to process sensor data.")

        if not has_mission_manager:
            return False  # No GNC components present


        self.mission_manager = mission_manager

        
        phase = self.mission_manager.get_current_phase()

        self.current_phase = phase
        self.current_guidance = phase.guidance
        self.current_navigation = phase.navigation
        self.current_controller = phase.controller
        self.current_allocator = phase.allocator
        self.current_allocator.set_actuators(self.actuators)

        self.current_controller_dt = phase.dt_control
        self.current_guidance_dt = phase.dt_guid
        self.current_navigation_dt = phase.dt_nav

        return True  # GNC components are present

    def show_spacecraft_basic_info(self):
        print(f"Spacecraft Name: {self.spacecraft_data.name}")
        print(f"Mass: {self.spacecraft_data.mass} kg")
        print(f"Inertia Tensor: \n{self.spacecraft_data.inertia_tensor}")

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
        if self.has_gnc:
            return self.current_navigation_dt, self.current_guidance_dt, self.current_controller_dt
        else:
            return np.nan, np.nan, np.nan
    
    def __compute_navigation(self, simulation_data):

        if self.__should_compute(simulation_data, self.current_navigation_dt):
            estimation = self.current_navigation.estimate(self.sensors)

            self.spacecraft_data.navigation_estimated_data = estimation
            
    def __compute_guidance(self, simulation_data):
        if self.__should_compute(simulation_data, self.current_guidance_dt):
            # Placeholder for guidance computation logic
            self.spacecraft_data.guidance_reference_data = self.current_guidance.compute_reference(
                self.spacecraft_data.navigation_estimated_data, simulation_data
            )

    def __compute_control(self, simulation_data):
        if self.__should_compute(simulation_data, self.current_controller_dt):
            self.spacecraft_data.control_output_data = self.current_controller.compute_control(
                self.spacecraft_data.navigation_estimated_data,
                self.spacecraft_data.guidance_reference_data
            )

            self.current_allocator.allocate(self.spacecraft_data.control_output_data)

    def __compute_actuation(self, simulation_data):
        # Placeholder for actuation computation logic
        force = np.zeros(3)
        torque = np.zeros(3)

        if self.actuators is None:
            self.spacecraft_data.current_force_exerted = force
            self.spacecraft_data.current_torque_exerted = torque
            return

        for actuator in self.actuators:
            actuator.update(simulation_data.t)

            actuator_output = actuator.get_output()
            #print(f"Actuator output: Force = {actuator_output.force}, Torque = {actuator_output.torque}")

            force += actuator_output.force
            torque += actuator_output.torque

        self.spacecraft_data.current_force_exerted = force
        self.spacecraft_data.current_torque_exerted = torque

    def update_gnc_components(self):
        # Update the current phase and its associated components        
        self.current_phase = self.mission_manager.get_current_phase()
        self.current_guidance = self.current_phase.guidance
        self.current_navigation = self.current_phase.navigation
        self.current_controller = self.current_phase.controller

        self.current_allocator = self.current_phase.allocator
        self.current_allocator.set_actuators(self.actuators)


        self.current_controller_dt = self.current_phase.dt_control
        self.current_guidance_dt = self.current_phase.dt_guid
        self.current_navigation_dt = self.current_phase.dt_nav

    def __should_compute(self, simulation_data, dt_component):
        tick = simulation_data.tick
        dt_master = simulation_data.dt_master
        return tick % int(dt_component / dt_master) == 0

    def compute_tick_step(self, simulation_data):

        # Update sensors
        if self.has_sensors:
            for sensor in self.sensors:
                sensor.update(self.spacecraft_data, simulation_data)

        if self.has_gnc:
            # Update mission manager
            phase_has_changed = self.mission_manager.update(simulation_data)

            if phase_has_changed:
                self.update_gnc_components()

            self.__compute_navigation(simulation_data)
            self.__compute_guidance(simulation_data)
            self.__compute_control(simulation_data)

        self.__compute_actuation(simulation_data)
