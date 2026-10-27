import warnings
import numpy as np

from argos.general.dataclasses import SpacecraftData
from argos.core.mission_manager import MissionManager
from argos.sensors.sensor_base import SensorBase
from argos.actuators.actuators_base import ActuatorBase
from argos.faults.fault_manager import FaultManager
from argos.general.data_save import TransitionEventInfo
from argos.frames.frames import NONE_frame, BODY_frame
from argos.frames.frames_base import Frame
from argos.frames.transform_manager import TransformsManager
from argos.math import quaternion_from_euler, quaternion_to_DCM

from typing import TYPE_CHECKING
if TYPE_CHECKING:    
    from argos.general.dataclasses import MissionPhase
    from argos.core.simulation import Simulation
    from argos.maneuvers.impulsive_maneuver import ImpulsiveManeuver

class Spacecraft():
    """
    Represent a spacecraft within an ARGOS simulation.

    Parameters
    ----------
    name : str
        Spacecraft identifier.
    mass : float
        Spacecraft mass [kg].
    initial_state : ndarray, shape (13,)
        Initial spacecraft state vector. The state is ordered as
        position [m], velocity [m/s], attitude quaternion, and angular
        velocity [rad/s].
    inertia_tensor : ndarray, shape (3, 3)
        Spacecraft inertia tensor [kg·m²].
    actuators : list of ActuatorBase, optional
        Actuators attached to the spacecraft.
    sensors : list of SensorBase, optional
        Sensors attached to the spacecraft.
    mission_manager : MissionManager, optional
        Mission manager defining the spacecraft mission phases and
        transitions.
    parent : Simulation, optional
        Simulation instance containing the spacecraft.
    verbose : bool, optional
        If True, print spacecraft information during initialization.
    """
    
    def __init__(
        self,
        name: str,
        mass: float,
        initial_state: np.ndarray,
        inertia_tensor: np.ndarray,
        actuators: list[ActuatorBase] | None = None,
        sensors: list[SensorBase] | None = None,
        mission_manager: MissionManager | None = None,
        parent: Simulation | None = None,
        verbose: bool = False,
    ):                       

        # --- ATRIBUTES DECLARATION ---        
        self.name = name
        self.mass = mass
        self.inertia_tensor = inertia_tensor
        self.inertia_tensor_inv = np.linalg.inv(inertia_tensor)

        self.spacecraft_data = SpacecraftData()

        self.sensors = sensors
        self.actuators = actuators        

        self.mission_manager = mission_manager
        self.parent = parent

        self.spacecraft_data.target_name = None  # Name of the target spacecraft, if any

        self.rotational_model = None
        self.translational_model = None        

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
        self.has_translational_model = False
        self.has_rotational_model = False

        self.transforms_manager = TransformsManager()

        self._check_initialization()
        self._init_state_variables(initial_state)
        self._set_sensors_names()
        self._set_actuators_names()
        self._set_sensors_frames()
        self._update_propagator_transforms()

        self.fault_manager = FaultManager(self)

        self.verbose = verbose

        if self.verbose:
            print(f"="*10 + " Spacecraft created successfully. " + "="*10)
            self.show_spacecraft_basic_info()
            self.show_spacecraft_gnc_info()
            self.show_spacecraft_state_info()
            print("="*10 + " End of spacecraft information. " + "="*10)

    def _check_initialization(self):
        
        if isinstance(self.sensors, list) and len(self.sensors) > 0 and all(isinstance(sensor, SensorBase) for sensor in self.sensors):
            self.has_sensors = True

        if isinstance(self.actuators, list) and len(self.actuators) > 0 and all (isinstance(actuator, ActuatorBase) for actuator in self.actuators):
            self.has_actuators = True

        if isinstance(self.mission_manager, MissionManager):
            self.has_mission_manager = True

        if self.has_mission_manager:
            self._check_and_update_phase_info(self.mission_manager.current_phase)
 
    def _check_and_update_phase_info(self, phase: MissionPhase, update_info: dict | None = None):
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

        if phase.translational_model is None:
            self.has_translational_model = False
            self.translational_model = None
        else:
            self.has_translational_model = True
            self.translational_model = phase.translational_model

        if phase.rotational_model is None:
            self.has_rotational_model = False
            self.rotational_model = None
        else:
            self.has_rotational_model = True
            self.rotational_model = phase.rotational_model

        self._update_true_state_frames()
        self._update_propagator_transforms()        

        self.spacecraft_data.current_propagation_dt = phase.dt_propagation        
        
        self.spacecraft_data.current_master_dt = self.get_min_dt()

        if update_info is None:
            update_info = {}
            update_info["from_phase"] = "Initial State"
            update_info["to_phase"] = self.current_phase.name
            update_info["via"] = "Initial Setup"

        event_info = TransitionEventInfo(
                                        time=self.spacecraft_data.t,
                                        tick=self.spacecraft_data.tick,
                                        from_phase=update_info["from_phase"],
                                        to_phase=update_info["to_phase"],
                                        transition_name=update_info["via"],
                                        dt_navigation=self.current_navigation_dt,
                                        dt_guidance=self.current_guidance_dt,
                                        dt_control=self.current_control_dt,
                                        dt_propagation=self.spacecraft_data.current_propagation_dt,
                                        dt_master=self.spacecraft_data.current_master_dt
                                    )
        
        self.parent.record_transition(self.name, event_info)

    def _update_true_state_frames(self):
        """
        Update the reference frames associated with the true spacecraft state.

        The translational frame is obtained from the active translational
        propagator.

        If both translational and rotational propagation are active, both
        states use the same reference frame.

        If only rotational propagation is active, its frame is obtained
        from the rotational propagator.

        If neither propagator is active, both frames are set to None and
        a warning is issued.
        """

        state = self.spacecraft_data.true_state

        # ================================================================
        # Neither propagator is active
        # ================================================================
        if not self.has_translational_model and not self.has_rotational_model:

            state.translational_frame = None
            state.rotational_frame = None

            warnings.warn(
                "Neither translational nor rotational propagation is active. "
                "The reference frames of the true state cannot be determined.",
                UserWarning,
                stacklevel=2,
            )

            return
        
        # Translational frame        
        if self.has_translational_model:

            frame = self.translational_model.propagation_frame

            if frame is None:
                raise ValueError(
                    "The active translational propagator does not define "
                    "its propagation frame."
                )

            state.translational_frame = frame

        else:
            state.translational_frame = NONE_frame
        
        # Rotational frame        
        if not self.has_rotational_model:

            state.rotational_frame = NONE_frame

        elif self.has_translational_model:

            # Both propagators must use the same reference frame.
            state.rotational_frame = state.translational_frame
            self.rotational_model.propagation_frame = state.translational_frame

        else:

            # Rotational propagation without translational propagation.
            frame = self.rotational_model.propagation_frame

            if frame is NONE_frame:
                raise ValueError(
                    "The active rotational propagator does not define "
                    "its propagation frame."
                )

            state.rotational_frame = frame


        # TO BE DELETED LATER:
        self.spacecraft_data.perturbation_data.frame = state.rotational_frame        

    def _init_state_variables(self, initial_state: np.ndarray):
        self.spacecraft_data.true_state.position = initial_state[0:3]
        self.spacecraft_data.true_state.velocity = initial_state[3:6]
        self.spacecraft_data.true_state.attitude = initial_state[6:10]
        self.spacecraft_data.true_state.angular_velocity = initial_state[10:]        

    def _set_sensors_names(self):
        if not self.has_sensors:
            return
        
        sensors_count = 0
        
        for sensor in self.sensors:
            if sensor.name is None:
                sensor.name = f"Sensor_{sensors_count}"
                sensors_count += 1

    def _set_actuators_names(self):
        if not self.has_actuators:
            return

        actuators_count = 0
        for actuator in self.actuators:
            if actuator.name is None:
                actuator.name = f"Actuator_{actuators_count}"
                actuators_count += 1

    def _set_sensors_frames(self):
        if not self.has_sensors:
            return

        for sensor in self.sensors:            
            if sensor.frame is None:
                sensor.frame = Frame(
                    name=sensor.name + "_frame",
                    inertial=False,
                    origin= "Sensor position in body frame",                    
                )                
                rotation_matrix = quaternion_to_DCM(sensor.rotation_quat)

                self.transforms_manager.register_static(
                    source_frame=sensor.frame,
                    target_frame=BODY_frame,
                    matrix=rotation_matrix,
                    translation=sensor.position
                )

    def _update_propagator_transforms(self):        
        def get_quaternion_propagator_from_body():
            return quaternion_to_DCM(self.spacecraft_data.true_state.attitude)        

        # Register current propagator to body - translational            
        translation_frame = None
        if self.has_translational_model and self.current_phase.translational_model.propagation_frame is not NONE_frame:            
            translation_frame = self.spacecraft_data.true_state.translational_frame
            self.transforms_manager.register_dynamic(
                source_frame=translation_frame,
                target_frame=BODY_frame,
                provider=get_quaternion_propagator_from_body
            )            

        # Register current propagator to body - rotational
        if self.has_rotational_model and self.current_phase.rotational_model.propagation_frame is not NONE_frame:
            rotational_frame = self.spacecraft_data.true_state.rotational_frame            

            if translation_frame != rotational_frame:
                self.transforms_manager.register_dynamic(
                    source_frame=rotational_frame,
                    target_frame=BODY_frame,
                    provider=get_quaternion_propagator_from_body
                )

    def update_sensor_name(self, current_name: str, new_name: str):
        '''
        Update the name of a sensor in the spacecraft's sensor list.
        
        Parameters:
        actual_name (str): The current name of the sensor to be updated.
        new_name (str): The new name to assign to the sensor.                    
        '''
        name_changed = False
        for sensor in self.sensors:
            if sensor.name == current_name:
                sensor.name = new_name
                name_changed = True
                print(f"Sensor name changed from '{current_name}' to '{new_name}'.")
                break

        if not name_changed:
            print(f"No sensor found with the name '{current_name}'. Name change not applied.")

    def update_actuator_name(self, current_name: str, new_name: str):
        name_changed = False
        for actuator in self.actuators:
            if actuator.name == current_name:
                actuator.name = new_name
                name_changed = True
                print(f"Actuator name changed from '{current_name}' to '{new_name}'.")
                break

        if not name_changed:
            print(f"No actuator found with the name '{current_name}'. Name change not applied.")            
            
    def get_dts(self):
        return self.current_navigation_dt, self.current_guidance_dt, self.current_control_dt, self.spacecraft_data.current_propagation_dt

    def get_min_dt(self):
        
        dts = [dt for dt in [self.current_navigation_dt, self.current_guidance_dt, self.current_control_dt, self.spacecraft_data.current_propagation_dt] if dt is not None and not np.isnan(dt)]

        if self.has_sensors:    
            for sensor in self.sensors:            
                if sensor.sample_rate_sec is not None:
                    print(f"  Sensor {sensor.name} sample rate: {sensor.sample_rate_sec}")
                    dts.append(sensor.sample_rate_sec)

        print(f"Current dt values for spacecraft '{self.name}': {dts}")
        print(f"Minimum dt for spacecraft '{self.name}': {min(dts) if dts else None}")

        return min(dts) if dts else None

    def apply_impulsive_maneuver(self, maneuver: ImpulsiveManeuver) -> None:
        """Apply an impulsive velocity change to the spacecraft."""

        if maneuver.executed:
            raise ValueError("This maneuver has already been executed.")

        true_state = self.spacecraft_data.true_state        


        if self.verbose:
            print(f"Impulsive maneuver '{maneuver.name}' applied at time {self.spacecraft_data.t:.2f}s.")
            print(f"Delta-v: {maneuver.delta_v}")
            print(f"Previous velocity: {true_state.velocity}")
        true_state.velocity += maneuver.delta_v
        if self.verbose:
            print(f"New velocity: {true_state.velocity}")            

        maneuver.executed = True

    def _should_compute(self, simulation_data, dt_component):
        tick = self.spacecraft_data.tick
        dt_master = self.spacecraft_data.current_master_dt

        return tick % int(dt_component / dt_master) == 0

    def init_sensors_parameters(self, simulation_data):
        if self.has_sensors is False:
            return
        
        for sensor in self.sensors:
            sensor.spacecraft_transform_manager = self.transforms_manager
            sensor.parent_spacecraft = self
            sensor.spacecraft_data = self.spacecraft_data
            sensor.simulation_data = simulation_data

    def update_mission_manager(self, simulation_data):
        if self.has_mission_manager is False:
            return

        phase_changed, update_info = self.mission_manager.update(simulation_data)
        
        if phase_changed:
            self._check_and_update_phase_info(self.mission_manager.current_phase, update_info)            

        return
    
    def update_sensors(self):
        if self.has_sensors is False:
            return
    
        for sensor in self.sensors:            
            sensor.update()
    
    def update_navigation(self, simulation_data):
        if self.has_navigation_law is False:
            return

        if self._should_compute(simulation_data, self.current_navigation_dt):            
            estimation = self.current_navigation_law.estimate(self.sensors)

            self.spacecraft_data.navigation_data = estimation
            
    def update_guidance(self, simulation_data):
        if self.has_guidance_law is False:
            return

        if self._should_compute(simulation_data, self.current_guidance_dt):   
            self.spacecraft_data.guidance_data = self.current_guidance_law.compute_reference(
                self.spacecraft_data.navigation_data,
                simulation_data
            )

    def update_control(self, simulation_data):
        if self.has_control_law is False:
            return

        if self._should_compute(simulation_data, self.current_control_dt):            
            self.spacecraft_data.control_data = self.current_control_law.compute_control(
                self.spacecraft_data.navigation_data,
                self.spacecraft_data.guidance_data
            )            

            self.current_allocator.allocate(self.spacecraft_data.control_data)

    def compute_actuation(self, simulation_data):
        """ Compute the total force and torque exerted by the spacecraft actuators in the BODY frame."""
        if self.has_actuators is False:
            return

        # Placeholder for actuation computation logic
        force = np.zeros(3)
        torque = np.zeros(3)

        
        self.spacecraft_data.exerted_data.force = force
        self.spacecraft_data.exerted_data.torque = torque        

        for actuator in self.actuators:
            actuator.update(self.spacecraft_data.t)

            actuator_output = actuator.get_output()

            force += actuator_output.force
            torque += actuator_output.torque

        self.spacecraft_data.exerted_data.force = force
        self.spacecraft_data.exerted_data.torque = torque

    def propagate_translational(self, simulation_data, environment):
        has_propagated = False
        if self.has_translational_model is False:
            return has_propagated

        if self._should_compute(simulation_data, self.spacecraft_data.current_propagation_dt):
            self.translational_model.propagate(self, simulation_data, environment)            
            has_propagated = True
            return has_propagated

    def propagate_rotational(self, simulation_data, environment):
        has_propagated = False
        if self.has_rotational_model is False:
            return has_propagated
        
        if self._should_compute(simulation_data, self.spacecraft_data.current_propagation_dt):
            self.rotational_model.propagate(self, simulation_data, environment)
            self.transforms_manager.invalidate_dynamic()
            has_propagated = True
            return has_propagated

    def change_target(self, new_target_name: str):
        self.spacecraft_data.target_name = new_target_name
        
        if self.verbose:
            print(f"Spacecraft '{self.name}' target changed to '{new_target_name}'.")

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

    def show_sensors_info(self):
        if self.has_sensors is False:
            print("No sensors available for this spacecraft.")
            return

        print(f"Spacecraft '{self.name}' Sensors Information:")
        for sensor in self.sensors:
            sensor.print_info()

    def show_faults_info(self):
        self.fault_manager.show_faults_info()

    def show_spacecraft_all_info(self):
        self.show_spacecraft_basic_info()
        self.show_spacecraft_gnc_info()
        self.show_spacecraft_state_info()
        self.show_spacecraft_reference_info()
        self.show_sensors_info()
        self.show_faults_info()
