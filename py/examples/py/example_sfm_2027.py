import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from py.modules.math import quaternion_from_euler, quaternion_to_DCM, DCM_to_quaternion
from py.modules.enviroments.environments import ClassicalEnvironment
from py.modules.core.simulation import Simulation
from py.modules.actuators.RCS import RCSThruster
from py.modules.sensors.generic_sensor import AbsoluteSensor
from py.modules.sensors.inertial_sensors import Accelerometer, Gyroscope
from py.modules.guidance.basic_laws import CustomGuidanceLaw
from py.modules.navigation.basic_laws import IdealNavigation
from py.modules.controllers.basic_laws import MultiRCSAllocator
from py.modules.controllers.classic_controllers import PDController
from py.modules.propagators.native_propagator import NativeTranslationalPropagator, NativeRotationalPropagator
from py.modules.core.mission_manager import MissionPhase, MissionManager
from py.modules.general.dataclasses import StateVariables, GuidanceOutput
from py.modules.general_tools import get_state_at
from py.modules.general.general_data import CM_mass, SM_mass, Ix_total, Iy_total, Iz_total

# ========================================================
#                      SIMULATION SETUP
# ========================================================

PD_KP_ATTITUDE = 1000
PD_KD_ATTITUDE = 20000
PD_KP_TRANSLATIONAL = 10
PD_KD_TRANSLATIONAL = 740

TIME_FACTOR_SEC = 27.321661 / (2.0 * np.pi) * 24 * 3600     # time unit unidad -> sideral lunar month (days)

SIM_MAX_TIME = 0.75952417*2 * TIME_FACTOR_SEC #60*13
print("SIM_MAX_TIME: ", SIM_MAX_TIME)

TARGET_DOCKING_AXIS_WRT_TARGET_FRAME = np.array([0.0, 0.0, 1.0])  # Docking axis of the target in the target frame
TARGET_COMPLEMENTARY_AXIS_WRT_TARGET_FRAME = np.array([1.0, 0.0, 0.0])  # Complementary axis of the target in the target frame
CHASER_DOCKING_AXIS_WRT_CHASER_FRAME = np.array([1.0, 0.0, 0.0])  # Docking axis of the chaser in the chaser frame
CHASER_COMPLEMENTARY_AXIS_WRT_CHASER_FRAME = np.array([0.0, 1.0, 0.0])  # Complementary axis of the chaser in the chaser frame
DISTANCE_ALONG_DOCKING_AXIS = 10.0  # Distance along the docking axis in meters from the target's docking port
INTERCEPTION_POINT_WRT_TARGET_FRAME = DISTANCE_ALONG_DOCKING_AXIS * TARGET_DOCKING_AXIS_WRT_TARGET_FRAME  # Interception point in the target frame

LONG_INTEGRATION_DT = 127

# ========================================================
#                   SIMULATION CREATION
# ========================================================

sim = Simulation(max_sim_time=SIM_MAX_TIME,
                 environment= ClassicalEnvironment(),
                 verbose = True
)

# ========================================================
#   ACTUATORS (2 SETS - Translational and Rotational)
# ========================================================
actuators = []
scale_factor = 1

THRUSTER_CLASSES = {
    "LOW": {
        "thrust": 50*scale_factor,       # N  (~23 kgf)
        "torque": 1805.0,      # N*m, reference torque with 4 m lever arm
    },
    "MEDIUM": {
        "thrust": 225*scale_factor,      # N  (~46 kgf)
        "torque": 3600.0,      # N*m, reference torque with 4 m lever arm
    },
    "HIGH": {
        "thrust": 716.1*scale_factor,      # N  (~73 kgf)
        "torque": 5729.0,      # N*m, reference torque with 4 m lever arm
    },
}

axes = {
    "X": np.array([1, 0, 0]),
    "Y": np.array([0, 1, 0]),
    "Z": np.array([0, 0, 1]),
}

for thruster_class, parameters in THRUSTER_CLASSES.items():

    thrust = parameters["thrust"]
    torque = parameters["torque"]

    for axis_name, axis in axes.items():

        for sign_name, sign in [("pos", 1), ("neg", -1)]:

            direction = sign * axis

            # ------------------------------------------------------------------
            # Translational thruster
            # ------------------------------------------------------------------

            name = (
                f"translational_thruster_"
                f"{thruster_class}_{axis_name}_{sign_name}"
            )

            actuators.append(
                RCSThruster(
                    name=name,
                    nominal_thrust=thrust,
                    direction=direction,                    
                    modulation_window=0.1,
                    minimum_on_time=0.01,
                    activation_threshold=0.0,
                )
            )

            # ------------------------------------------------------------------
            # Rotational thruster
            # ------------------------------------------------------------------

            name = (
                f"rotational_thruster_"
                f"{thruster_class}_{axis_name}_{sign_name}"
            )

            actuators.append(
                RCSThruster(
                    name=name,
                    nominal_thrust=0.0,
                    direction=direction,
                    override_torque_value=torque,
                    modulation_window=0.1,
                    minimum_on_time=0.01,
                    activation_threshold=0.0,
                )
            )

# ========================================================
#                       SENSORS
# ========================================================

sensors_sample_rate = 100  # Hz

abs_sensor_1 = AbsoluteSensor(
    name="absolute_sensor_1",
    sample_rate_freq=sensors_sample_rate,
    verbose=False,
)

sensors = [abs_sensor_1]

# ========================================================
#               COMMON LAWS & PROPAGATORS
# =======================================================

control_law = PDController(attitude_proportional_gain=PD_KP_ATTITUDE,
                            attitude_derivative_gain=PD_KD_ATTITUDE,
                            translational_proportional_gain=PD_KP_TRANSLATIONAL,
                            translational_derivative_gain=PD_KD_TRANSLATIONAL
                        )
actuator_allocator = MultiRCSAllocator()
translational_propagator = NativeTranslationalPropagator(dynamics="CR3BP")
rotational_propagator = NativeRotationalPropagator()

# ========================================================
#           PHASE 1 - Docking Axis Acquistion
# ======================================================== 

def acquisition_of_docking_axis_guidance_function(navigation_estimated_data, simulation_data):
    spacecraft_state = navigation_estimated_data.spacecraft_state
    target_state = navigation_estimated_data.target_state    

    # Current target position and orientation in the inertial frame
    target_current_position_wrt_inertal_frame = target_state.position  # Current target position in meters wrt the inertial frame    
    target_current_quaternion_wrt_inertal_frame = target_state.attitude  # Convert target attitude to quaternion

    rotation_matrix_target_to_inertial = quaternion_to_DCM(target_current_quaternion_wrt_inertal_frame)  # Rotation matrix from target frame to inertial frame

    # Current chaser position and orientation in the inertial frame
    chaser_current_position_wrt_inertial_frame = spacecraft_state.position  # Chaser position in meters wrt the inertial frame
    chaser_current_quaternion_wrt_inertial_frame = spacecraft_state.attitude  # Chaser attitude in quaternion wrt the inertial frame

    rotation_matrix_chaser_to_inertial = quaternion_to_DCM(chaser_current_quaternion_wrt_inertial_frame)  # Rotation matrix from chaser frame to inertial frame

    # Calculate the interception point in the inertial frame and the relative distance to it
    interception_point_wrt_inertial_frame = target_current_position_wrt_inertal_frame + rotation_matrix_target_to_inertial @ INTERCEPTION_POINT_WRT_TARGET_FRAME  # Position of the docking distance in the inertial frame
    chaser_relative_distance_to_interception_point = interception_point_wrt_inertial_frame - chaser_current_position_wrt_inertial_frame  # Relative distance between the chaser and the interception point

    # Calculate the desired orientation of the chaser based on the target's docking axis and complementary axis
    target_docking_axis_wrt_inertial_frame = rotation_matrix_target_to_inertial @ TARGET_DOCKING_AXIS_WRT_TARGET_FRAME  # Target docking axis in the inertial frame
    target_complementary_axis_wrt_inertial_frame = rotation_matrix_target_to_inertial @ TARGET_COMPLEMENTARY_AXIS_WRT_TARGET_FRAME  # Target complementary axis in the inertial frame
    chaser_docking_axis_wrt_inertial_frame = rotation_matrix_chaser_to_inertial @ CHASER_DOCKING_AXIS_WRT_CHASER_FRAME  # Chaser docking axis in the inertial frame
    chaser_complementary_axis_wrt_inertial_frame = rotation_matrix_chaser_to_inertial @ CHASER_COMPLEMENTARY_AXIS_WRT_CHASER_FRAME  # Chaser complementary axis in the inertial frame

    # It is needed that the chaser's docking axis is aligned with the target's docking axis, but in the opposite direction (to face each other)
    desired_chaser_docking_axis_wrt_inertial_frame = -target_docking_axis_wrt_inertial_frame

    # Additionally, the chaser's complementary axis should be aligned with the target's complementary axis
    desired_chaser_complementary_axis_wrt_inertial_frame = target_complementary_axis_wrt_inertial_frame

    # The desired chaser's normal axis can be computed as the cross product of the desired docking axis and the desired complementary axis
    desired_chaser_normal_axis_wrt_inertial_frame = np.cross(
        desired_chaser_docking_axis_wrt_inertial_frame,
        desired_chaser_complementary_axis_wrt_inertial_frame
    )

    # Construct the desired chaser's orientation matrix (DCM) using the desired axes
    desired_chaser_x_wrt_inertial = desired_chaser_docking_axis_wrt_inertial_frame
    desired_chaser_z_wrt_inertial = desired_chaser_complementary_axis_wrt_inertial_frame

    desired_chaser_y_wrt_inertial = np.cross(
        desired_chaser_z_wrt_inertial,
        desired_chaser_x_wrt_inertial
    )

    desired_chaser_dcm = np.column_stack([
        desired_chaser_x_wrt_inertial,
        desired_chaser_y_wrt_inertial,
        desired_chaser_z_wrt_inertial
    ])

    # Convert the desired chaser's DCM to quaternion representation
    desired_chaser_quaternion_wrt_inertial = DCM_to_quaternion(
        desired_chaser_dcm
    )

    return GuidanceOutput(
        state= StateVariables(
            position=interception_point_wrt_inertial_frame,
            velocity=target_state.velocity.copy(),
            attitude=desired_chaser_quaternion_wrt_inertial,
            angular_velocity=np.zeros(3),
        )
    )

axis_acquistion_guidance_law = CustomGuidanceLaw(custom_reference_function=acquisition_of_docking_axis_guidance_function)
axis_acquistion_guidance_law_dt = 0.1
axis_acqustion_navigation_law = IdealNavigation()
axis_acqustion_navigation_law_dt = 0.1
axis_acquistion_controller = control_law.copy()
axis_acquistion_controller_dt = 0.1

axis_acquistion_propagation_dt = 0.01

docking_axis_acquisition_phase = MissionPhase(
    name="Docking Axis Acquisition",

    guidance=axis_acquistion_guidance_law,
    navigation=axis_acqustion_navigation_law,
    controller=axis_acquistion_controller,
    allocator=actuator_allocator.copy(),

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

relative_hold_guidance_law = CustomGuidanceLaw(custom_reference_function=acquisition_of_docking_axis_guidance_function)
relative_hold_guidance_law_dt = 0.1
relative_hold_navigation_law = IdealNavigation()
relative_hold_navigation_law_dt = 0.1
relative_hold_controller = control_law.copy()
relative_hold_controller_dt = 0.1

relative_hold_propagation_dt = 0.01

relative_hold_phase = MissionPhase(
    name="Relative Hold",

    guidance=relative_hold_guidance_law,
    navigation=relative_hold_navigation_law,
    controller=relative_hold_controller,
    allocator=actuator_allocator.copy(),

    translational_model=translational_propagator.copy(),    

    dt_guid=relative_hold_guidance_law_dt,
    dt_nav=relative_hold_navigation_law_dt,
    dt_control=relative_hold_controller_dt,
    dt_propagation=relative_hold_propagation_dt,
)

# ========================================================
# PHASE 3 - Docking (final) Approach
# ========================================================

def final_approach_guidance_function(navigation_estimated_data, simulation_data):
    spacecraft_state = navigation_estimated_data.spacecraft_state
    target_state = navigation_estimated_data.target_state    

    # Current target position and orientation in the inertial frame
    target_current_position_wrt_inertal_frame = target_state.position  # Current target position in meters wrt the inertial frame            

    # Current chaser position and orientation in the inertial frame
    chaser_current_position_wrt_inertial_frame = spacecraft_state.position  # Chaser position in meters wrt the inertial frame
    
    # Calculate the interception point in the inertial frame and the relative distance to it
    interception_point_wrt_inertial_frame = target_current_position_wrt_inertal_frame
    
    return GuidanceOutput(
        state= StateVariables(
            position=interception_point_wrt_inertial_frame,
            velocity=target_state.velocity.copy(),
        )
    )



final_approach_guidance_law = CustomGuidanceLaw(custom_reference_function=final_approach_guidance_function)
final_approach_guidance_law_dt = 0.1
final_approach_navigation_law = IdealNavigation()
final_approach_navigation_law_dt = 0.1
final_approach_controller = control_law.copy()
final_approach_controller_dt = 0.1

final_approach_propagation_dt = 0.01

final_approach_phase = MissionPhase(
    name="Final Approach",

    guidance=final_approach_guidance_law,
    navigation=final_approach_navigation_law,
    controller=final_approach_controller,
    allocator=actuator_allocator.copy(),

    translational_model=translational_propagator.copy(),    

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
    navigation=IdealNavigation(),

    dt_nav= LONG_INTEGRATION_DT,
    dt_propagation= LONG_INTEGRATION_DT,
)

# ========================================================
#                    MISSION MANAGER
# ========================================================

mission_manager = MissionManager(
    initial_phase=docking_axis_acquisition_phase,
    verbose=True
)
mission_manager.add_phases([relative_hold_phase, final_approach_phase, docking_complete])

# ========================================================
#                   PHASES TRANSITIONS
# ========================================================

time_of_first_trheshold = None
time_of_relative_hold_transition = None

time_of_final_approach_transition = None

def condition_to_relative_hold(simulation_data):
    global time_of_relative_hold_transition, time_of_first_trheshold
    
    chaser_pos = None
    target_pos = None

    t = 0.0
    
    for spacecraft in simulation_data.spacecrafts:
        if spacecraft.name == "Target":
            t = spacecraft.spacecraft_data.t
            if t > 2:
                target_pos = get_state_at(simulation_data.simulation_history, "Target", t).position
        elif spacecraft.name == "Chaser":
            if t > 2:
                chaser_pos = get_state_at(simulation_data.simulation_history, "Chaser", t).position
        else:
            raise ValueError(f"Unknown spacecraft name: {spacecraft.name}")

    if chaser_pos is not None and target_pos is not None:
        relative_distance_magnitude = np.linalg.norm(chaser_pos - target_pos)            
        
        if relative_distance_magnitude <= 10.0 and time_of_first_trheshold is None:  # Transition when the relative distance is less than or equal to 8 meters
            time_of_first_trheshold = t

        if time_of_first_trheshold is not None:
            if t - time_of_first_trheshold >= 70:
                time_of_relative_hold_transition = t        
                return True


def condition_to_final_approach(simulation_data):
    if time_of_relative_hold_transition is not None:
            elapsed_time = simulation_data.spacecrafts[0].spacecraft_data.t - time_of_relative_hold_transition
            if elapsed_time >= 60*3:  # Transition after 3 minutes in relative hold
                global time_of_final_approach_transition
                time_of_final_approach_transition = simulation_data.spacecrafts[0].spacecraft_data.t
                return True

def condition_to_docking_complete(simulation_data):
    t = simulation_data.spacecrafts[0].spacecraft_data.t

    if t >= 60*12:  # Transition after 12 minutes in final approach
        pos_target = simulation_data.spacecrafts[0].spacecraft_data.true_state.position
        vel_target = simulation_data.spacecrafts[0].spacecraft_data.true_state.velocity
        acceleration_target = simulation_data.spacecrafts[0].spacecraft_data.true_state.acceleration

        # Overwrite the target's state to match the chaser's state at the moment of docking (eliminate drift due to uncertainities)
        simulation_data.spacecrafts[0].spacecraft_data.true_state.position = pos_target
        simulation_data.spacecrafts[0].spacecraft_data.true_state.velocity = vel_target
        simulation_data.spacecrafts[0].spacecraft_data.true_state.acceleration = acceleration_target

        # Change Sensors Freqs

        for spacecraft in simulation_data.spacecrafts:    
            for sensor in spacecraft.sensors:
                sensor.sample_rate_sec = LONG_INTEGRATION_DT
            

        return True


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
#           TARGET SPACECRAFT CONFIGURATION
# ========================================================
POSITION_FOR_NRHO_L2 = np.array([3.79543557e+08, -9.91715693e+03, 3.33735189e+06])
VELOCITY_FOR_NRHO_L2 = np.array([-1.65568984e-01, 1.68125627e+03, 2.58673649e+00])

target_initial_attitude = np.array([45, 45, 0])  # Target initial orientation in Euler angles (degrees)
target_initial_attitude_quat = quaternion_from_euler(
    np.deg2rad(target_initial_attitude[0]),
    np.deg2rad(target_initial_attitude[1]),
    np.deg2rad(target_initial_attitude[2])
)

target_initial_phase=MissionPhase(
    name="Target Motion",
    translational_model=NativeTranslationalPropagator(dynamics="CR3BP"),
    navigation=IdealNavigation(),
    dt_nav=1,
    dt_propagation=0.01
)

target_mission_manager = MissionManager(
    initial_phase=target_initial_phase,
    verbose=True
)

docking_complete = MissionPhase(
    name="Docking Complete",
    translational_model=translational_propagator.copy(),
    navigation=IdealNavigation(),

    dt_nav= LONG_INTEGRATION_DT,
    dt_propagation=LONG_INTEGRATION_DT,
)
target_mission_manager.add_phases([docking_complete])

target_mission_manager.add_transition(
    from_phase=target_initial_phase,
    target_phase=docking_complete,
    condition=condition_to_docking_complete,
    transition_name="Docking Complete"
)

sim.add_spacecraft(
    name="Target",    
    initial_position= POSITION_FOR_NRHO_L2,  # Target position in meters
    initial_velocity= VELOCITY_FOR_NRHO_L2,  # Target stationary    
    initial_attitude= target_initial_attitude_quat,
    sensors=[AbsoluteSensor(sensors_sample_rate)],
    mission_manager=target_mission_manager,
    verbose=False
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

chaser_initial_position = (POSITION_FOR_NRHO_L2 -50)   # Initial position in meters
chaser_initial_velocity = VELOCITY_FOR_NRHO_L2  # Initial velocity in meters per second

inertial_tensor = np.diag([Ix_total, Iy_total, Iz_total])  # Inertia tensor in kg*m^2

sim.add_spacecraft(
    name="Chaser",
    target_name="Target",
    mass=CM_mass + SM_mass,
    initial_position=chaser_initial_position,
    initial_velocity=chaser_initial_velocity,
    initial_attitude=chaser_initial_orientation_quat,
    initial_angular_velocity=chaser_initial_angular_velocity,
    inertia_tensor=inertial_tensor,
    sensors= sensors,
    actuators=actuators,
    mission_manager=mission_manager,    
    verbose=False
)

# ========================================================
#                   SIMULATION EXECUTION
# ========================================================

metada = sim.simulate()

# =======================================================
#                 SIMULATION RESULTS
# =======================================================

print("="*50)
print("Chaser Mass: ", CM_mass + SM_mass, "kg")
print("Chaser Inertia Tensor: ", inertial_tensor, "kg*m^2")
print("="*50)
