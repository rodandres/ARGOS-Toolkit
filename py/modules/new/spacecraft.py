from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

@dataclass
class SpacecraftData:
    name: str

    mass: float
    inertia_tensor: np.ndarray

    actuators: list # NOTE: Add the type of the list
    sensors: list # NOTE: Add the type of the list
    controllers: list # NOTE: Add the type of the list
    guidance: list # NOTE: Add the type of the list

    dt_nav: float 
    dt_guid: float
    dt_control: float
    verbose: bool

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

    def __init__(self, name: str, mass: float, inertia_tensor: np.ndarray, 
                 actuators: list, sensors: list, controllers: list, guidance: list,
                 dt_nav: float, dt_guid: float, dt_control: float, initial_state: np.ndarray, verbose: bool = False):

        spacecraft_data = SpacecraftData(
            name=name,

            mass=mass,
            inertia_tensor=inertia_tensor,

            actuators=actuators,
            sensors=sensors,
            controllers=controllers,
            guidance=guidance,

            dt_nav=dt_nav,
            dt_guid=dt_guid,
            dt_control=dt_control,

            verbose=verbose
        )

        self.spacecraft_data = spacecraft_data

        # Initialize the spacecraft state variables based on the provided initial state
        self.spacecraft_data.true_pos = initial_state[0:3]
        self.spacecraft_data.true_vel = initial_state[3:6]
        self.spacecraft_data.true_q = initial_state[6:10]
        self.spacecraft_data.true_omega = initial_state[10:13]

        if self.spacecraft_data.verbose:
            print(f"="*10 + " Spacecraft created successfully. " + "="*10)
            self.show_spacecraft_basic_info()
            self.show_spacecraft_gnc_info()
            self.show_spacecraft_state_info()
            print("="*10 + " End of spacecraft information. " + "="*10)            

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
        for sensor in self.spacecraft_data.sensors:

            true_state = np.concatenate((
                self.spacecraft_data.true_pos,
                self.spacecraft_data.true_vel,
                self.spacecraft_data.true_accel,
                self.spacecraft_data.true_q,
                self.spacecraft_data.true_omega,
                self.spacecraft_data.true_alpha
                ))
            true_ref_state = np.concatenate((
                self.spacecraft_data.true_ref_pos,
                self.spacecraft_data.true_ref_vel,
                self.spacecraft_data.true_ref_accel,
                self.spacecraft_data.true_ref_q,
                self.spacecraft_data.true_ref_omega,
                self.spacecraft_data.true_ref_alpha
                ))

            #sensor_output = sensor.get_measurement(true_state, true_ref_state) NOTE DEBE CMABIARSE POR LOS ARGUMENTOS

            if sensor.get_type() == "IMU":
                pass
            if sensor.get_type() == "GPS":
                pass
            
            
            # if sensor.get_type() == "Gyroscope":
            #     pass
            # if sensor.get_type() == "Accelerometer":
            #     pass
            # if sensor.get_type() == "Star Tracker":
            #     pass
            # if sensor.get_type() == "Magnetometer":
            #     pass
            # if sensor.get_type() == "Sun Sensor":
            #     pass
            # if sensor.get_type() == "LIDAR":
            #     pass
            # if sensor.get_type() == "Camera":
            #     pass


                


        # Placeholder for navigation computation logic
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

    def compute_tick_step(self, tick: int, dt_master: float):        

        # Check if it's time to compute navigation
        if tick % int(self.spacecraft_data.dt_nav / dt_master) == 0:
            self.__compute_navigation()

        # Check if it's time to compute guidance
        if tick % int(self.spacecraft_data.dt_guid / dt_master) == 0:
            self.__compute_guidance()

        # Check if it's time to compute control
        if tick % int(self.spacecraft_data.dt_control / dt_master) == 0:
            self.__compute_control()

