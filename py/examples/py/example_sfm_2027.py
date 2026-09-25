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
from py.general.dataclasses import StateVariables, GuidanceOutput
from py.modules.general_tools import get_state_at
from py.general.general_data import CM_mass, SM_mass, Ix_total, Iy_total, Iz_total

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
    target_state = navigation_estimated_data.reference_state    

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
    target_state = navigation_estimated_data.reference_state    

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

#result = sim.simulate()

# =======================================================
#                 SIMULATION RESULTS
# =======================================================

print("="*50)
print("Chaser Mass: ", CM_mass + SM_mass, "kg")
print("Chaser Inertia Tensor: ", inertial_tensor, "kg*m^2")
print("="*50)


import os
import numpy as np
import pandas as pd


import os
import numpy as np

import os
import numpy as np


def save_simulation_history_csv(
    simulation_history,
    directory="simulation_history",
):
    """
    Exporta una SimulationHistory a archivos CSV.

    Las estructuras anidadas se recorren recursivamente.
    Los datos numéricos se guardan como números.
    Los datos no numéricos (None, strings, etc.) se guardan como texto.
    """

    os.makedirs(directory, exist_ok=True)

    # ==========================================================================
    # Generic recursive exporter
    # ==========================================================================

    def save_data(data, path, prefix):

        # ----------------------------------------------------------------------
        # Dictionary
        # ----------------------------------------------------------------------

        if isinstance(data, dict):

            for key, value in data.items():

                save_data(
                    value,
                    path,
                    f"{prefix}_{key}",
                )

            return

        # ----------------------------------------------------------------------
        # Dataclass
        # ----------------------------------------------------------------------

        if hasattr(data, "__dataclass_fields__"):

            for field_name in data.__dataclass_fields__:

                save_data(
                    getattr(data, field_name),
                    path,
                    f"{prefix}_{field_name}",
                )

            return

        # ----------------------------------------------------------------------
        # Convert to NumPy
        # ----------------------------------------------------------------------

        data = np.asarray(data)

        if data.size == 0:
            return

        # ----------------------------------------------------------------------
        # Scalar
        # ----------------------------------------------------------------------

        if data.ndim == 0:

            with open(
                os.path.join(path, f"{prefix}.csv"),
                "w",
            ) as file:

                file.write(f"{data.item()}\n")

            return

        # ----------------------------------------------------------------------
        # Determine whether data is numeric
        # ----------------------------------------------------------------------

        is_numeric = np.issubdtype(
            data.dtype,
            np.number,
        )

        # ----------------------------------------------------------------------
        # Numeric data
        # ----------------------------------------------------------------------

        if is_numeric:

            if data.ndim == 1:

                np.savetxt(
                    os.path.join(path, f"{prefix}.csv"),
                    data,
                    delimiter=",",
                    header=prefix,
                    comments="",
                )

            else:

                flattened = data.reshape(
                    data.shape[0],
                    -1,
                )

                np.savetxt(
                    os.path.join(path, f"{prefix}.csv"),
                    flattened,
                    delimiter=",",
                )

            return

        # ----------------------------------------------------------------------
        # Non-numeric data
        # ----------------------------------------------------------------------

        flattened = data.reshape(
            data.shape[0],
            -1,
        )

        with open(
            os.path.join(path, f"{prefix}.csv"),
            "w",
        ) as file:

            for row in flattened:

                file.write(
                    ",".join(
                        str(value)
                        for value in row
                    )
                    + "\n"
                )

    # ==========================================================================
    # Simulation time
    # ==========================================================================

    save_data(
        simulation_history.time,
        directory,
        "simulation_time",
    )

    # ==========================================================================
    # Spacecraft histories
    # ==========================================================================

    for spacecraft_name, history in simulation_history.spacecrafts_history.items():

        spacecraft_directory = os.path.join(
            directory,
            spacecraft_name,
        )

        os.makedirs(
            spacecraft_directory,
            exist_ok=True,
        )

        save_data(
            history.t,
            spacecraft_directory,
            "t",
        )

        save_data(
            history.tick,
            spacecraft_directory,
            "tick",
        )

        save_data(
            history.target_name,
            spacecraft_directory,
            "target_name",
        )

        save_data(
            history.true_state,
            spacecraft_directory,
            "true_state",
        )

        save_data(
            history.estimated_data,
            spacecraft_directory,
            "estimated_data",
        )

        save_data(
            history.reference_data,
            spacecraft_directory,
            "reference_data",
        )

        save_data(
            history.control_output_data,
            spacecraft_directory,
            "control_output_data",
        )

        save_data(
            history.current_force_exerted,
            spacecraft_directory,
            "force",
        )

        save_data(
            history.current_torque_exerted,
            spacecraft_directory,
            "torque",
        )

save_simulation_history_csv(
    result,
    directory="simulation_history",
)

# ------------------------------------------------------------------------------
# Spacecraft names
# ------------------------------------------------------------------------------

chaser_name = "Chaser"
target_name = "Target"


# ------------------------------------------------------------------------------
# Retrieve histories
# ------------------------------------------------------------------------------

chaser_history = result.spacecrafts_history[chaser_name]
target_history = result.spacecrafts_history[target_name]


# ------------------------------------------------------------------------------
# Retrieve time vectors
# ------------------------------------------------------------------------------

chaser_t = np.asarray(
    chaser_history.t,
    dtype=float,
)

target_t = np.asarray(
    target_history.t,
    dtype=float,
)


# ------------------------------------------------------------------------------
# Retrieve position vectors
# ------------------------------------------------------------------------------

chaser_position = np.asarray(
    chaser_history.true_state["position"],
    dtype=float,
)

target_position = np.asarray(
    target_history.true_state["position"],
    dtype=float,
)


# ------------------------------------------------------------------------------
# Validate data
# ------------------------------------------------------------------------------

if len(chaser_t) != len(chaser_position):
    raise ValueError(
        f"{chaser_name}: time and position have different lengths: "
        f"{len(chaser_t)} vs {len(chaser_position)}"
    )

if len(target_t) != len(target_position):
    raise ValueError(
        f"{target_name}: time and position have different lengths: "
        f"{len(target_t)} vs {len(target_position)}"
    )

if chaser_position.shape[1] != 3:
    raise ValueError(
        f"{chaser_name}: position must have shape (N, 3)."
    )

if target_position.shape[1] != 3:
    raise ValueError(
        f"{target_name}: position must have shape (N, 3)."
    )


# ------------------------------------------------------------------------------
# Sort by time
# ------------------------------------------------------------------------------

chaser_indices = np.argsort(chaser_t)
target_indices = np.argsort(target_t)

chaser_t = chaser_t[chaser_indices]
chaser_position = chaser_position[chaser_indices]

target_t = target_t[target_indices]
target_position = target_position[target_indices]


# ==============================================================================
# TIME LIMITS
# ==============================================================================

# Complete orbit
ORBIT_MAX_TIME = SIM_MAX_TIME

# Relative motion
RELATIVE_MAX_TIME = 60 * 12


# ==============================================================================
# COMMON TIME FOR COMPLETE ORBIT
# ==============================================================================

orbit_t_min = max(
    chaser_t[0],
    target_t[0],
)

orbit_t_max = min(
    chaser_t[-1],
    target_t[-1],
    ORBIT_MAX_TIME,
)

if orbit_t_max <= orbit_t_min:
    raise ValueError(
        "No common time interval exists for the complete orbit."
    )


chaser_orbit_t = chaser_t[
    (chaser_t >= orbit_t_min) &
    (chaser_t <= orbit_t_max)
]

target_orbit_t = target_t[
    (target_t >= orbit_t_min) &
    (target_t <= orbit_t_max)
]

orbit_t = np.unique(
    np.concatenate(
        [
            chaser_orbit_t,
            target_orbit_t,
        ]
    )
)


# ==============================================================================
# INTERPOLATE COMPLETE ORBIT
# ==============================================================================

chaser_orbit_position = np.column_stack(
    [
        np.interp(
            orbit_t,
            chaser_t,
            chaser_position[:, i],
        )
        for i in range(3)
    ]
)

target_orbit_position = np.column_stack(
    [
        np.interp(
            orbit_t,
            target_t,
            target_position[:, i],
        )
        for i in range(3)
    ]
)


# ==============================================================================
# COMMON TIME FOR RELATIVE MOTION
# ==============================================================================

relative_t_min = max(
    chaser_t[0],
    target_t[0],
)

relative_t_max = min(
    chaser_t[-1],
    target_t[-1],
    RELATIVE_MAX_TIME,
)

if relative_t_max <= relative_t_min:
    raise ValueError(
        "No common time interval exists for the relative motion."
    )


chaser_relative_t = chaser_t[
    (chaser_t >= relative_t_min) &
    (chaser_t <= relative_t_max)
]

target_relative_t = target_t[
    (target_t >= relative_t_min) &
    (target_t <= relative_t_max)
]

relative_t = np.unique(
    np.concatenate(
        [
            chaser_relative_t,
            target_relative_t,
        ]
    )
)


# ==============================================================================
# INTERPOLATE RELATIVE MOTION
# ==============================================================================

chaser_relative_position = np.column_stack(
    [
        np.interp(
            relative_t,
            chaser_t,
            chaser_position[:, i],
        )
        for i in range(3)
    ]
)

target_relative_position = np.column_stack(
    [
        np.interp(
            relative_t,
            target_t,
            target_position[:, i],
        )
        for i in range(3)
    ]
)


# ==============================================================================
# RELATIVE POSITION
#
# r_C/T = r_C - r_T
# ==============================================================================

relative_position = (
    chaser_relative_position
    - target_relative_position
)

# ==============================================================================
# FIGURE
# ==============================================================================

fig = plt.figure(
    figsize=(18, 6),
)


# ==============================================================================
# SUBPLOT 1 — ABSOLUTE ORBIT
# ==============================================================================

ax1 = fig.add_subplot(1, 3, 1, projection="3d")

ax1.plot(
    chaser_orbit_position[:, 0],
    chaser_orbit_position[:, 1],
    chaser_orbit_position[:, 2],
    label=chaser_name,
)

ax1.plot(
    target_orbit_position[:, 0],
    target_orbit_position[:, 1],
    target_orbit_position[:, 2],
    label=target_name,
)

MU = 1.215e-2
LENGTH_FACTOR = 384400.0e3

ax1.scatter(
    (1-MU)*LENGTH_FACTOR,  # X position of the Moon in meters
    0,
    0,
    marker="o",
    color="white",
    edgecolors="black",
    label="Moon",
)

ax1.set_title(f"Complete Orbit ($t \\leq {SIM_MAX_TIME:.0f}$ s)")
ax1.set_xlabel("X [m]")
ax1.set_ylabel("Y [m]")
ax1.set_zlabel("Z [m]")
ax1.legend()


# ==============================================================================
# SUBPLOT 2 — RELATIVE POSITION VS TIME
#
# Time: 0 --> 720 s
# ==============================================================================

ax2 = fig.add_subplot(
    1,
    3,
    2,
)


labels = (
    r"$\Delta x$",
    r"$\Delta y$",
    r"$\Delta z$",
)


for i in range(3):

    ax2.plot(
        relative_t,
        relative_position[:, i],
        linewidth=1.3,
        label=labels[i],
    )


ax2.axhline(
    0.0,
    linestyle="--",
    linewidth=0.8,
)


ax2.set_title(
    "Relative Position"
)

ax2.set_xlabel(
    "Time [s]"
)

ax2.set_ylabel(
    "Relative Position [m]"
)

ax2.set_xlim(
    relative_t_min,
    relative_t_max,
)

ax2.legend(
    fontsize=8,
)


# ==============================================================================
# SUBPLOT 3 — RELATIVE TRAJECTORY
#
# Target fixed at origin.
# Chaser trajectory:
#
#       r_C/T = r_C - r_T
# ==============================================================================

ax3 = fig.add_subplot(
    1,
    3,
    3,
    projection="3d",
)


ax3.plot(
    relative_position[:, 0],
    relative_position[:, 1],
    relative_position[:, 2],
    linewidth=1.3,
    label=chaser_name,
)


# Target fixed at origin

ax3.scatter(
    0.0,
    0.0,
    0.0,
    marker="o",
    s=50,
    label=target_name,
)


# Initial relative position

ax3.scatter(
    relative_position[0, 0],
    relative_position[0, 1],
    relative_position[0, 2],
    marker="o",
    s=35,
    label="Chaser Relative Initial Position",
)


# Final relative position

ax3.scatter(
    relative_position[-1, 0],
    relative_position[-1, 1],
    relative_position[-1, 2],
    marker="x",
    label="Chaser Relative Final Position",
    s=50,
)


ax3.set_title(
    "Relative Trajectory"
)

ax3.set_xlabel(
    r"$\Delta x$ [m]"
)

ax3.set_ylabel(
    r"$\Delta y$ [m]"
)

ax3.set_zlabel(
    r"$\Delta z$ [m]"
)

ax3.legend(
    fontsize=8,
)


# ------------------------------------------------------------------------------
# Change camera view
#
# Rotate the view to the opposite side.
# ------------------------------------------------------------------------------

# Rotate 45 degrees positively around Z
ax3.view_init(elev=25, azim=135)


# ==============================================================================
# EQUAL ASPECT RATIO FOR 3D PLOTS
# ==============================================================================

def set_equal_3d_axes(ax, data):

    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]

    x_range = np.ptp(x)
    y_range = np.ptp(y)
    z_range = np.ptp(z)

    max_range = max(
        x_range,
        y_range,
        z_range,
    )

    if max_range == 0:
        max_range = 1.0

    x_mid = (x.max() + x.min()) / 2
    y_mid = (y.max() + y.min()) / 2
    z_mid = (z.max() + z.min()) / 2

    ax.set_xlim(
        x_mid - max_range / 2,
        x_mid + max_range / 2,
    )

    ax.set_ylim(
        y_mid - max_range / 2,
        y_mid + max_range / 2,
    )

    ax.set_zlim(
        z_mid - max_range / 2,
        z_mid + max_range / 2,
    )


set_equal_3d_axes(
    ax1,
    np.vstack(
        [
            chaser_orbit_position,
            target_orbit_position,
        ]
    ),
)


set_equal_3d_axes(
    ax3,
    np.vstack(
        [
            relative_position,
            np.zeros((1, 3)),
        ]
    ),
)


# ==============================================================================
# FIGURE TITLE
# ==============================================================================

fig.suptitle(
    f"{chaser_name} / {target_name}",
    fontsize=14,
)


fig.tight_layout(
    rect=[0, 0, 1, 0.94],
)


# ==============================================================================
# SHOW
# ==============================================================================

plt.show()

#relative_position = plot_relative_position(["Chaser", "Target"], result, max_time=60*13)
#plot_trajectory(["Chaser", "Target"], result, show=True,)
#plot_control_result("Chaser", result)