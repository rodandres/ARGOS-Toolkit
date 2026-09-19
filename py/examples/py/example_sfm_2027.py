import numpy as np

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from py.modules.math import quaternion_from_euler
from py.modules.enviroments.environments import ClassicalEnvironment
from py.modules.core.simulation import Simulation
from py.modules.actuators.RCS import RCSThruster
from py.modules.sensors.generic_sensor import AbsoluteSensor
from py.modules.sensors.inertial_sensors import Accelerometer, Gyroscope
from py.modules.guidance.basic_laws import GuidancePlaceholder
from py.modules.navigation.basic_laws import NavigationPlaceholder
from py.modules.controllers.basic_laws import ControllerPlaceholder, ControlAllocatorPlaceholder
from py.modules.propagators.native_propagator import NativeTranslationalPropagator, NativeRotationalPropagator
from py.modules.core.mission_manager import MissionPhase, MissionManager

sim_max_time = 60*15
sim = Simulation(max_sim_time=sim_max_time,
                 environment= ClassicalEnvironment(),
                 verbose = True
)

# ========================================================
#   ACTUATORS (2 SETS - Translational and Rotational)
# ========================================================
actuators = []
THRUSTER_FORCE = 100  # N
THRUSTER_TORQUE = 300  # N*m

axes = {
    "X": np.array([1, 0, 0]),
    "Y": np.array([0, 1, 0]),
    "Z": np.array([0, 0, 1]),
}

for axis_name, axis in axes.items():

    for sign_name, sign in [("pos", 1), ("neg", -1)]:

        direction = sign * axis

        # Translational thruster
        name = f"translational_thruster_{axis_name}_{sign_name}"

        actuators.append(
            RCSThruster(
                name=name,
                nominal_thrust=THRUSTER_FORCE,
                direction=direction,
                override_torque_value=0,
                modulation_window=0.1,
                minimum_on_time=0.01,
                activation_threshold=0.0,
            )
        )

        # Rotational thruster
        name = f"rotational_thruster_{axis_name}_{sign_name}"

        actuators.append(
            RCSThruster(
                name=name,
                nominal_thrust=0,
                direction=direction,
                override_torque_value=THRUSTER_TORQUE,
                modulation_window=0.1,
                minimum_on_time=0.01,
                activation_threshold=0.0,
            )
        )

# ========================================================
#                       SENSORS
# ========================================================

sensors_sample_rate = 100  # Hz

accel_1 = Accelerometer(
    name="accelerometer_1",
    sample_rate_freq=sensors_sample_rate,
    verbose=False,
)

gyro_1 = Gyroscope(
    name="gyroscope_1",
    sample_rate_freq=sensors_sample_rate,
    verbose=False,
)

abs_sensor_1 = AbsoluteSensor(
    name="absolute_sensor_1",
    sample_rate_freq=sensors_sample_rate,
    verbose=False,
)

# ========================================================
#               COMMON LAWS & PROPAGATORS
# ========================================================


actuator_allocator = ControlAllocatorPlaceholder()
translational_propagator = NativeTranslationalPropagator(dynamics="CR3BP")
rotational_propagator = NativeRotationalPropagator()

# ========================================================
#           PHASE 1 - Docking Axis Acquistion
# ======================================================== 

axis_acquistion_guidance_law = GuidancePlaceholder()
axis_acquistion_guidance_law_dt = 0.1
axis_acqustion_navigation_law = NavigationPlaceholder()
axis_acqustion_navigation_law_dt = 0.1
axis_acquistion_controller = ControllerPlaceholder()
axis_acquistion_controller_dt = 0.1

axis_acquistion_propagation_dt = 0.1

docking_axis_acquisition_phase = MissionPhase(
    name="Docking Axis Acquisition",
    guidance=axis_acquistion_guidance_law,
    navigation=axis_acqustion_navigation_law,
    controller=axis_acquistion_controller,
    allocator=actuator_allocator,
    translational_model=translational_propagator.copy(),
    rotational_model=rotational_propagator.copy(),

    dt_guid=axis_acquistion_guidance_law_dt,
    dt_nav=axis_acqustion_navigation_law_dt,
    dt_control=axis_acquistion_controller_dt,

    dt_propagation=axis_acquistion_propagation_dt,    
)

# ========================================================
#           PHASE 2 - Relative Hold
# ========================================================

relative_hold_guidance_law = GuidancePlaceholder()
relative_hold_guidance_law_dt = 0.1
relative_hold_navigation_law = NavigationPlaceholder()
relative_hold_navigation_law_dt = 0.1
relative_hold_controller = ControllerPlaceholder()
relative_hold_controller_dt = 0.1

relative_hold_propagation_dt = 0.1

relative_hold_phase = MissionPhase(
    name="Relative Hold",
    guidance=relative_hold_guidance_law,
    navigation=relative_hold_navigation_law,
    controller=relative_hold_controller,
    allocator=actuator_allocator,
    translational_model=translational_propagator.copy(),
    rotational_model=rotational_propagator.copy(),
    dt_guid=relative_hold_guidance_law_dt,
    dt_nav=relative_hold_navigation_law_dt,
    dt_control=relative_hold_controller_dt,
    dt_propagation=relative_hold_propagation_dt,
)

# ========================================================
# PHASE 3 - Docking (final) Approach
# ========================================================

final_approach_guidance_law = GuidancePlaceholder()
final_approach_guidance_law_dt = 0.1
final_approach_navigation_law = NavigationPlaceholder()
final_approach_navigation_law_dt = 0.1
final_approach_controller = ControllerPlaceholder()
final_approach_controller_dt = 0.1

final_approach_propagation_dt = 0.1

final_approach_phase = MissionPhase(
    name="Final Approach",
    guidance=final_approach_guidance_law,
    navigation=final_approach_navigation_law,
    controller=final_approach_controller,
    allocator=actuator_allocator,
    translational_model=translational_propagator.copy(),
    rotational_model=rotational_propagator.copy(),
    dt_guid=final_approach_guidance_law_dt,
    dt_nav=final_approach_navigation_law_dt,
    dt_control=final_approach_controller_dt,
    dt_propagation=final_approach_propagation_dt,
)

# ========================================================
# PHASE 4 - Docking Complete
# ========================================================

docking_complete = MissionPhase(
    name="Docking Complete",
    translational_model=translational_propagator.copy(),
    rotational_model=rotational_propagator.copy(),
    dt_propagation=0.1,
)

# ========================================================
#                    MISSION MANAGER
# ========================================================

mission_manager = MissionManager(
    initial_phase=docking_axis_acquisition_phase
)
mission_manager.add_phases([relative_hold_phase, final_approach_phase, docking_complete])

# ========================================================
#                   PHASES TRANSITIONS
# ========================================================

def condition_to_relative_hold(spacecraft_data, simulation_data):
    # Place holder
    return False

def condition_to_final_approach(spacecraft_data, simulation_data):
    # Place holder
    return False

def condition_to_docking_complete(spacecraft_data, simulation_data):
    # Place holder
    return False

mission_manager.add_transition(
    from_phase=docking_axis_acquisition_phase,
    target_phase=relative_hold_phase,
    condition=condition_to_relative_hold,
    transition_name="Docking Axis Acquired"
)

mission_manager.add_transition(
    from_phase=relative_hold_phase,
    target_phase=final_approach_phase,
    condition=condition_to_final_approach,
    transition_name="Relative Hold Complete"
)

mission_manager.add_transition(
    from_phase=final_approach_phase,
    target_phase=docking_complete,
    condition=condition_to_docking_complete,
    transition_name="Docking Complete"
)

# ========================================================
#           CHASER SPACECRAFT CONFIGURATION
# ========================================================
chaser_initial_orientation = [0, 0, 0]  # Initial orientation in Euler angles (degrees)
chaser_initial_orientation_quat = quaternion_from_euler(
    np.deg2rad(chaser_initial_orientation[0]),
    np.deg2rad(chaser_initial_orientation[1]),
    np.deg2rad(chaser_initial_orientation[2])
)
chaser_initial_angular_velocity = np.array([0.0, 0.0, 0.0])  # Initial angular velocity in rad/s
chaser_initial_position = np.array([0, 0, 0])  # Initial position in meters
chaser_initial_velocity = np.array([0, 0, 0])  # Initial velocity in meters per second

inertial_tensor = np.diag([1, 1, 1])  # Inertia tensor in kg*m^2

sim.add_spacecraft(
    name="Chaser",
    initial_position=chaser_initial_position,
    initial_velocity=chaser_initial_velocity,
    initial_attitude=chaser_initial_orientation_quat,
    initial_angular_velocity=chaser_initial_angular_velocity,
    inertia_tensor=inertial_tensor,
    sensors=[accel_1, gyro_1, abs_sensor_1],
    actuators=actuators,
    mission_manager=mission_manager,
    verbose=False
)

# ========================================================
#            TARGET SPACECRAFT CONFIGURATION
# ========================================================

sim.add_spacecraft(
    name="Target",
    initial_position=np.array([0, 0, 0]),  # Initial position in meters
    initial_velocity=np.array([0, 0, 0]),  # Initial velocity in meters per second        
    mission_manager=MissionManager(
        initial_phase=MissionPhase(
            name="Target Phase",
            translational_model=translational_propagator.copy(),
        )
    ),    
    verbose=False
)