from dataclasses import dataclass, field

import numpy as np

@dataclass
class SpacecraftData:
    name: str

    mass: float
    inertia_tensor: np.ndarray

    # Spacraft state variables in inertial frame (same as the inertial frame of the simulation)
    true_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    true_vel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    true_accel: np.ndarray = field(default_factory=lambda: np.zeros(3))

    true_q: np.ndarray = field(default_factory=lambda: np.array([1, 0, 0, 0]))
    true_omega: np.ndarray = field(default_factory=lambda: np.zeros(3))
    true_alpha: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # Spacecraft state variables in sensor frame (same as the sensor frame of the spacecraft)
    measured_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    measured_vel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    measured_accel: np.ndarray = field(default_factory=lambda: np.zeros(3))

    measured_q: np.ndarray = field(default_factory=lambda: np.array([1, 0, 0, 0]))
    measured_omega: np.ndarray = field(default_factory=lambda: np.zeros(3))
    measured_alpha: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # Spacraft state variables in body frame (same as the body frame of the spacecraft)
    body_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    body_vel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    body_accel: np.ndarray = field(default_factory=lambda: np.zeros(3))

    body_q: np.ndarray = field(default_factory=lambda: np.array([1, 0, 0, 0]))
    body_omega: np.ndarray = field(default_factory=lambda: np.zeros(3))
    body_alpha: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # Reference state variables in inertial frame (same as the inertial frame of the simulation)
    true_ref_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    true_ref_vel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    true_ref_accel: np.ndarray = field(default_factory=lambda: np.zeros(3))

    true_ref_q: np.ndarray = field(default_factory=lambda: np.array([1, 0, 0, 0]))
    true_ref_omega: np.ndarray = field(default_factory=lambda: np.zeros(3))
    true_ref_alpha: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # Reference state variables in sensor frame (same as the sensor frame of the spacecraft)
    measured_ref_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    measured_ref_vel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    measured_ref_accel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    measured_ref_q: np.ndarray = field(default_factory=lambda: np.array([1, 0, 0, 0]))
    measured_ref_omega: np.ndarray = field(default_factory=lambda: np.zeros(3))
    measured_ref_alpha: np.ndarray = field(default_factory=lambda: np.zeros(3))


class Spacecraft():

    def __init__(self, name: str, mass: float, initial_state: np.ndarray, inertia_tensor: np.ndarray, 
                 actuators: list | None = None, sensors: list | None = None,
                 mission_manager: object | None = None,
                 verbose: bool = False):

        spacecraft_data = SpacecraftData(
            name=name,

            mass=mass,
            inertia_tensor=inertia_tensor,

            verbose=verbose
        )

        self.spacecraft_data = spacecraft_data

        # Initialize the spacecraft state variables based on the provided initial state
        self.spacecraft_data.true_pos = initial_state[0:3]
        self.spacecraft_data.true_vel = initial_state[3:6]
        self.spacecraft_data.true_q = initial_state[6:10]
        self.spacecraft_data.true_omega = initial_state[10:13]

        # Check and set the GNC components
        self.has_gnc = self.check_gnc_components(mission_manager, actuators, sensors)


        if self.spacecraft_data.verbose:
            print(f"="*10 + " Spacecraft created successfully. " + "="*10)
            self.show_spacecraft_basic_info()
            self.show_spacecraft_gnc_info()
            self.show_spacecraft_state_info()
            print("="*10 + " End of spacecraft information. " + "="*10)            

    def check_gnc_components(self, mission_manager, actuators, sensors):
        has_mission_manager = mission_manager is not None
        has_actuators = actuators is not None and len(actuators) > 0
        has_sensors = sensors is not None and len(sensors) > 0

        if has_mission_manager and not has_actuators:
            raise ValueError("Mission manager is present, but no actuators are defined.")
        if has_mission_manager and not has_sensors:
            raise ValueError("Mission manager is present, but no sensors are defined.")

        if has_actuators and not has_mission_manager:
            raise Warning("Actuators are defined, but no mission manager is present. The spacecraft may not be able to perform any actions.")

        if has_sensors and not has_mission_manager:
            raise Warning("Sensors are defined, but no mission manager is present. The spacecraft may not be able to process sensor data.")

        if not has_mission_manager:
            return False  # No GNC components present

        self.actuators = actuators
        self.sensors = sensors

        self.mission_manager = mission_manager

        self.update_gnc_components()  # Update the current phase and its associated components

        return True  # GNC components are present


    def show_spacecraft_basic_info(self):
        print(f"Spacecraft Name: {self.spacecraft_data.name}")
        print(f"Mass: {self.spacecraft_data.mass} kg")
        print(f"Inertia Tensor: \n{self.spacecraft_data.inertia_tensor}")

    def show_spacecraft_gnc_info(self):
        print(f"Actuators: {self.spacecraft_data.actuators}")
        print(f"Sensors: {self.spacecraft_data.sensors}")
        print(f"Controllers: {self.spacecraft_data.controllers}")
        print(f"Guidance: {self.spacecraft_data.guidance}")

        print(f"Navigation Time Step: {self.spacecraft_data.dt_nav} s")
        print(f"Guidance Time Step: {self.spacecraft_data.dt_guid} s")
        print(f"Control Time Step: {self.spacecraft_data.dt_control} s")

    def show_spacecraft_state_info(self):
        print("True State (Inertial Frame):")
        print(f"Position: {self.spacecraft_data.true_pos}")
        print(f"Velocity: {self.spacecraft_data.true_vel}")
        print(f"Acceleration: {self.spacecraft_data.true_accel}")
        print(f"Quaternion: {self.spacecraft_data.true_q}")
        print(f"Angular Velocity: {self.spacecraft_data.true_omega}")
        print(f"Angular Acceleration: {self.spacecraft_data.true_alpha}")

        print("\nMeasured State (Sensor Frame):")
        print(f"Position: {self.spacecraft_data.measured_pos}")
        print(f"Velocity: {self.spacecraft_data.measured_vel}")
        print(f"Acceleration: {self.spacecraft_data.measured_accel}")
        print(f"Quaternion: {self.spacecraft_data.measured_q}")
        print(f"Angular Velocity: {self.spacecraft_data.measured_omega}")
        print(f"Angular Acceleration: {self.spacecraft_data.measured_alpha}")

        print("\nBody State (Body Frame):")
        print(f"Position: {self.spacecraft_data.body_pos}")
        print(f"Velocity: {self.spacecraft_data.body_vel}")
        print(f"Acceleration: {self.spacecraft_data.body_accel}")
        print(f"Quaternion: {self.spacecraft_data.body_q}")
        print(f"Angular Velocity: {self.spacecraft_data.body_omega}")
        print(f"Angular Acceleration: {self.spacecraft_data.body_alpha}")

    def show_spacecraft_reference_info(self):
        print("True Reference State (Inertial Frame):")
        print(f"Position: {self.spacecraft_data.true_ref_pos}")
        print(f"Velocity: {self.spacecraft_data.true_ref_vel}")
        print(f"Acceleration: {self.spacecraft_data.true_ref_accel}")
        print(f"Quaternion: {self.spacecraft_data.true_ref_q}")
        print(f"Angular Velocity: {self.spacecraft_data.true_ref_omega}")
        print(f"Angular Acceleration: {self.spacecraft_data.true_ref_alpha}")

        print("\nMeasured Reference State (Sensor Frame):")
        print(f"Position: {self.spacecraft_data.measured_ref_pos}")
        print(f"Velocity: {self.spacecraft_data.measured_ref_vel}")
        print(f"Acceleration: {self.spacecraft_data.measured_ref_accel}")
        print(f"Quaternion: {self.spacecraft_data.measured_ref_q}")
        print(f"Angular Velocity: {self.spacecraft_data.measured_ref_omega}")
        print(f"Angular Acceleration: {self.spacecraft_data.measured_ref_alpha}")

    def show_spacecraft_all_info(self):
        self.show_spacecraft_basic_info()
        self.show_spacecraft_gnc_info()
        self.show_spacecraft_state_info()
        self.show_spacecraft_reference_info()    
        
    def get_gnc_dts(self):
        return self.spacecraft_data.dt_nav, self.spacecraft_data.dt_guid, self.spacecraft_data.dt_control
    
    def __compute_navigation(self):
        pass

    def __compute_guidance(self):
        # Placeholder for guidance computation logic
        pass

    def __compute_control(self):
        # Placeholder for control computation logic
        pass

    def __compute_actuation(self):
        # Placeholder for actuation computation logic
        pass

    def update_gnc_components(self):
        # Update the current phase and its associated components
        self.current_phase = self.mission_manager.get_current_phase()
        self.current_controller = self.current_phase.controller
        self.current_guidance = self.current_phase.guidance
        self.current_navigation = self.current_phase.navigation
        self.current_controller_dt = self.current_phase.dt_control
        self.current_guidance_dt = self.current_phase.dt_guid
        self.current_navigation_dt = self.current_phase.dt_nav        

    def compute_tick_step(self, simulation_data):

        tick = simulation_data.tick
        dt_master = simulation_data.dt_master

        # Update sensors
        for sensor in self.spacecraft_data.sensors:
            sensor.update(self.spacecraft_data, simulation_data)


        if self.has_gnc:
            # Update mission manager
            self.mission_manager.update(simulation_data)

            self.update_gnc_components()

            # Check if it's time to compute navigation
            if tick % int(self.current_navigation_dt / dt_master) == 0:
                self.__compute_navigation()

            # Check if it's time to compute guidance
            if tick % int(self.current_guidance_dt / dt_master) == 0:
                self.__compute_guidance()

            # Check if it's time to compute control
            if tick % int(self.current_controller_dt / dt_master) == 0:
                self.__compute_control()

