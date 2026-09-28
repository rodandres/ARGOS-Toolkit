import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from py.modules.math import quaternion_from_euler, quaternion_to_DCM
from py.modules.enviroments.environments import ClassicalEnvironment
from py.modules.core.simulation import Simulation
from py.modules.actuators.RCS import RCSThruster
from py.modules.sensors.generic_sensor import AbsoluteSensor
from py.modules.sensors.inertial_sensors import Accelerometer, Gyroscope
from py.modules.guidance.basic_laws import CustomGuidanceLaw
from py.modules.navigation.basic_laws import CustomNavigation, IdealNavigation
from py.modules.controllers.basic_laws import CustomController, ControlAllocatorPlaceholder, CustomControlAllocator
from py.modules.propagators.native_propagator import NativeTranslationalPropagator, NativeRotationalPropagator
from py.modules.core.mission_manager import MissionPhase, MissionManager
from py.modules.general.general_data import CM_mass, SM_mass, Ix_total, Iy_total, Iz_total
from py.modules.visualization.trajectories import plot_trajectory, plot_trajectory_3d

sim_max_time = 60*12

sim = Simulation(max_sim_time=sim_max_time,
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
        "thrust": 225*scale_factor,       # N  (~23 kgf)
        "torque": 1805.0,      # N*m, reference torque with 4 m lever arm
    },
    "MEDIUM": {
        "thrust": 450*scale_factor,      # N  (~46 kgf)
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

def allocate_actuators(control_output, actuators):
    def _allocate_vector(command, actuator_type, actuators):
        command = np.asarray(command, dtype=float)

        # Select actuators of the required type
        actuators = [
            actuator
            for actuator in actuators
            if actuator.name.startswith(actuator_type)
        ]

        # Allocate X, Y and Z independently
        for axis_idx in range(3):

            requested = command[axis_idx]

            if np.isclose(requested, 0.0):
                continue

            sign = np.sign(requested)
            magnitude = abs(requested)

            # --------------------------------------------------
            # Find actuators capable of producing this direction
            # --------------------------------------------------

            candidates = []

            for actuator in actuators:

                # Check whether actuator produces the requested
                # direction along this axis
                if actuator.direction[axis_idx] * sign <= 0:
                    continue

                # Determine actuator capacity
                if actuator_type == "translational_thruster":
                    capacity = actuator.nominal_thrust
                else:
                    capacity = actuator.override_torque_value

                candidates.append((capacity, actuator))

            if not candidates:
                continue

            # --------------------------------------------------
            # Select the smallest actuator capable of satisfying
            # the complete demand
            # --------------------------------------------------

            candidates.sort(key=lambda x: x[0])

            selected_actuator = None

            for capacity, actuator in candidates:

                if capacity >= magnitude:
                    selected_actuator = actuator
                    break

            # --------------------------------------------------
            # If no single actuator is large enough, use the
            # largest available actuator and let it saturate
            # --------------------------------------------------

            if selected_actuator is None:
                _, selected_actuator = candidates[-1]

            # -----------------------------------   ---------------
            # Send the COMPLETE requested command to the actuator
            # --------------------------------------------------

            selected_actuator.set_command(
                magnitude
            )

    force_to_allocate = np.asarray(control_output.force,dtype=float)
    torque_to_allocate = np.asarray(control_output.torque,dtype=float)

    # Reset all actuators
    for actuator in actuators:
        actuator.set_command(0.0)

    # Allocate translation
    _allocate_vector(
        command=force_to_allocate,
        actuator_type="translational_thruster",
        actuators=actuators
    )

    # Allocate rotation
    _allocate_vector(
        command=torque_to_allocate,
        actuator_type="rotational_thruster",
        actuators=actuators
    )


actuator_allocator = CustomControlAllocator(allocation_function=allocate_actuators)
translational_propagator = NativeTranslationalPropagator(dynamics="CR3BP")
rotational_propagator = NativeRotationalPropagator()


from py.modules.general.dataclasses import GuidanceReference, StateVariables, ControlOutput

def docking_axis_guidance(navigation_estimated_data, simulation_data):
    spacecraft_state = navigation_estimated_data.spacecraft_state
    target_state = navigation_estimated_data.reference_state

    # Información definida por el problema
    target_docking_axis_wrt_target_frame = np.array([0.0, 0.0, 1.0])  # Docking axis of the target in the target frame
    target_complementary_axis_wrt_target_frame = np.array([1.0, 0.0, 0.0])  # Complementary axis of the target in the target frame
    chaser_docking_axis_wrt_chaser_frame = np.array([1.0, 0.0, 0.0])  # Docking axis of the chaser in the chaser frame
    chaser_complementary_axis_wrt_chaser_frame = np.array([0.0, 1.0, 0.0])  # Complementary axis of the chaser in the chaser frame
    distance_along_docking_axis = 10.0  # Distance along the docking axis in meters from the target's docking port
    interception_point_wrt_target_frame = distance_along_docking_axis * target_docking_axis_wrt_target_frame  # Interception point in the target frame

    # Información actual del objetivo y del perseguidor
    target_current_position_wrt_inertal_frame = target_state.position  # Current target position in meters wrt the inertial frame    
    target_current_quaternion_wrt_inertal_frame = target_state.attitude  # Convert target attitude to quaternion

    rotation_matrix_target_to_inertial = quaternion_to_DCM(target_current_quaternion_wrt_inertal_frame)  # Rotation matrix from target frame to inertial frame

    chaser_current_position_wrt_inertial_frame = spacecraft_state.position  # Chaser position in meters wrt the inertial frame
    chaser_current_quaternion_wrt_inertial_frame = spacecraft_state.attitude  # Chaser attitude in quaternion wrt the inertial frame

    rotation_matrix_chaser_to_inertial = quaternion_to_DCM(chaser_current_quaternion_wrt_inertial_frame)  # Rotation matrix from chaser frame to inertial frame

    # Calcular la posición del punto de intercepción en el marco inercial
    interception_point_wrt_inertial_frame = target_current_position_wrt_inertal_frame + rotation_matrix_target_to_inertial @ interception_point_wrt_target_frame  # Position of the docking distance in the inertial frame
    chaser_relative_distance_to_interception_point = interception_point_wrt_inertial_frame - chaser_current_position_wrt_inertial_frame  # Relative distance between the chaser and the interception point

    # Calcular la orientación deseada del perseguidor para alinear su eje de acoplamiento con el eje de acoplamiento del objetivo
    target_docking_axis_wrt_inertial_frame = rotation_matrix_target_to_inertial @ target_docking_axis_wrt_target_frame  # Target docking axis in the inertial frame
    target_complementary_axis_wrt_inertial_frame = rotation_matrix_target_to_inertial @ target_complementary_axis_wrt_target_frame  # Target complementary axis in the inertial frame
    chaser_docking_axis_wrt_inertial_frame = rotation_matrix_chaser_to_inertial @ chaser_docking_axis_wrt_chaser_frame  # Chaser docking axis in the inertial frame
    chaser_complementary_axis_wrt_inertial_frame = rotation_matrix_chaser_to_inertial @ chaser_complementary_axis_wrt_chaser_frame  # Chaser complementary axis in the inertial frame

    # Se necesita que los dos ejes del docking esten enfrentados, es decir dC = -dT
    desired_chaser_docking_axis_wrt_inertial_frame = -target_docking_axis_wrt_inertial_frame

    # Pero adicional, se necesita una orientación especifica, basicamente cerrar el roll sobre ese del docking una vez alineados.
    desired_chaser_complementary_axis_wrt_inertial_frame = target_complementary_axis_wrt_inertial_frame

    desired_chaser_normal_axis_wrt_inertial_frame = np.cross(
        desired_chaser_docking_axis_wrt_inertial_frame,
        desired_chaser_complementary_axis_wrt_inertial_frame
    )

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

    desired_chaser_quaternion_wrt_inertial = DCM_to_quaternion(
        desired_chaser_dcm
    )

    return GuidanceReference(
        state= StateVariables(
            position=interception_point_wrt_inertial_frame,
            velocity=target_state.velocity.copy(),
            attitude=desired_chaser_quaternion_wrt_inertial,
            angular_velocity=np.zeros(3),
        )
    )

def final_approach_guidance(navigation_estimated_data, simulation_data):
    spacecraft_state = navigation_estimated_data.spacecraft_state
    target_state = navigation_estimated_data.reference_state

    # Información definida por el problema
    target_docking_axis_wrt_target_frame = np.array([0.0, 0.0, 1.0])  # Docking axis of the target in the target frame
    target_complementary_axis_wrt_target_frame = np.array([1.0, 0.0, 0.0])  # Complementary axis of the target in the target frame
    chaser_docking_axis_wrt_chaser_frame = np.array([1.0, 0.0, 0.0])  # Docking axis of the chaser in the chaser frame
    chaser_complementary_axis_wrt_chaser_frame = np.array([0.0, 1.0, 0.0])  # Complementary axis of the chaser in the chaser frame
    distance_along_docking_axis = 0.0  # Distance along the docking axis in meters from the target's docking port
    interception_point_wrt_target_frame = distance_along_docking_axis * target_docking_axis_wrt_target_frame  # Interception point in the target frame

    # Información actual del objetivo y del perseguidor
    target_current_position_wrt_inertal_frame = target_state.position  # Current target position in meters wrt the inertial frame    
    target_current_quaternion_wrt_inertal_frame = target_state.attitude  # Convert target attitude to quaternion

    rotation_matrix_target_to_inertial = quaternion_to_DCM(target_current_quaternion_wrt_inertal_frame)  # Rotation matrix from target frame to inertial frame

    chaser_current_position_wrt_inertial_frame = spacecraft_state.position  # Chaser position in meters wrt the inertial frame
    chaser_current_quaternion_wrt_inertial_frame = spacecraft_state.attitude  # Chaser attitude in quaternion wrt the inertial frame

    rotation_matrix_chaser_to_inertial = quaternion_to_DCM(chaser_current_quaternion_wrt_inertial_frame)  # Rotation matrix from chaser frame to inertial frame

    # Calcular la posición del punto de intercepción en el marco inercial
    interception_point_wrt_inertial_frame = target_current_position_wrt_inertal_frame + rotation_matrix_target_to_inertial @ interception_point_wrt_target_frame  # Position of the docking distance in the inertial frame
    chaser_relative_distance_to_interception_point = interception_point_wrt_inertial_frame - chaser_current_position_wrt_inertial_frame  # Relative distance between the chaser and the interception point

    # Calcular la orientación deseada del perseguidor para alinear su eje de acoplamiento con el eje de acoplamiento del objetivo
    target_docking_axis_wrt_inertial_frame = rotation_matrix_target_to_inertial @ target_docking_axis_wrt_target_frame  # Target docking axis in the inertial frame
    target_complementary_axis_wrt_inertial_frame = rotation_matrix_target_to_inertial @ target_complementary_axis_wrt_target_frame  # Target complementary axis in the inertial frame
    chaser_docking_axis_wrt_inertial_frame = rotation_matrix_chaser_to_inertial @ chaser_docking_axis_wrt_chaser_frame  # Chaser docking axis in the inertial frame
    chaser_complementary_axis_wrt_inertial_frame = rotation_matrix_chaser_to_inertial @ chaser_complementary_axis_wrt_chaser_frame  # Chaser complementary axis in the inertial frame

    # Se necesita que los dos ejes del docking esten enfrentados, es decir dC = -dT
    desired_chaser_docking_axis_wrt_inertial_frame = -target_docking_axis_wrt_inertial_frame

    # Pero adicional, se necesita una orientación especifica, basicamente cerrar el roll sobre ese del docking una vez alineados.
    desired_chaser_complementary_axis_wrt_inertial_frame = target_complementary_axis_wrt_inertial_frame

    desired_chaser_normal_axis_wrt_inertial_frame = np.cross(
        desired_chaser_docking_axis_wrt_inertial_frame,
        desired_chaser_complementary_axis_wrt_inertial_frame
    )

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

    desired_chaser_quaternion_wrt_inertial = DCM_to_quaternion(
        desired_chaser_dcm
    )

    return GuidanceReference(
        state= StateVariables(
            position=interception_point_wrt_inertial_frame,
            velocity=target_state.velocity.copy(),
            attitude=desired_chaser_quaternion_wrt_inertial,
            angular_velocity=np.zeros(3),
        )
    )


from py.modules.math import quaternion_error as quat_error
from py.modules.math import DCM_to_quaternion

def PD_controller(estimated_state, reference):

    kp_attitude = 1000
    kd_attitude = 20000

    minimum_torque = None
    maximum_torque = None

    kp_position = 10
    kd_position = 740


    # ==========================================================
    # State retrieval
    # ==========================================================

    actual_quaternion = estimated_state.spacecraft_state.attitude.astype(float).copy()
    actual_angular_velocity = estimated_state.spacecraft_state.angular_velocity.astype(float).copy()    

    reference_quaternion = reference.state.attitude.astype(float).copy()    

    # ==========================================================
    # Quaternion attitude error
    # ==========================================================

    actual_quaternion /= np.linalg.norm(actual_quaternion)

    quaternion_error = quat_error(
        reference_quaternion,
        actual_quaternion,
    )

    quaternion_vector = quaternion_error[:3]
    quaternion_scalar = quaternion_error[3]

    # Always follow the shortest rotation.
    if quaternion_scalar < 0.0:

        quaternion_vector *= -1.0

    # ==========================================================
    # PD control law
    # ==========================================================

    commanded_torque = (
        -kp_attitude * quaternion_vector
        -kd_attitude * actual_angular_velocity
    )

    # ==========================================================
    # Torque saturation
    # ==========================================================

    if (
        minimum_torque is not None
        and maximum_torque is not None
    ):
        commanded_torque = np.clip(
            commanded_torque,
            minimum_torque,
            maximum_torque,
        )

    actual_position = (
        estimated_state.spacecraft_state.position
        .astype(float)
        .copy()
    )

    actual_velocity = (
        estimated_state.spacecraft_state.velocity
        .astype(float)
        .copy()
    )

    reference_position = (
        reference.state.position
        .astype(float)
        .copy()
    )

    reference_velocity = (
        reference.state.velocity
        .astype(float)
        .copy()
    )

    # ==========================================================
    # Position and velocity errors
    # ==========================================================

    position_error = (
        reference_position
        - actual_position
    )

    velocity_error = (
        reference_velocity
        - actual_velocity
    )

    # ==========================================================
    # PD control law
    # ==========================================================

    commanded_force = (
        kp_position * position_error
        + kd_position * velocity_error
    )

    return ControlOutput(
        force=commanded_force,
        torque=commanded_torque,
    )


docking_axis_transition = MissionPhase(
    name="Transition to docking axis",
    guidance= CustomGuidanceLaw(custom_reference_function=docking_axis_guidance),
    navigation= IdealNavigation(),
    controller= CustomController(control_function=PD_controller),
    allocator=actuator_allocator,
    translational_model=translational_propagator.copy(),
    rotational_model=rotational_propagator.copy(),

    dt_nav = 0.1,  # Navigation update rate (10 Hz)
    dt_guid = 0.1,  # Guidance update rate (1 Hz)
    dt_control = 0.1, # Control update rate (10 Hz)

    dt_propagation= 0.01 # Propagation time step (100 Hz)
)

relative_hold_phase = MissionPhase(
    name="relative_hold",        
    
    translational_model=translational_propagator.copy(),
    guidance = CustomGuidanceLaw(custom_reference_function=docking_axis_guidance),
    navigation= IdealNavigation(),
    controller= CustomController(control_function=PD_controller),
    allocator=actuator_allocator,    

    dt_nav = 0.1,  # Navigation update rate (10 Hz)
    dt_guid = 0.1,  # Guidance update rate (1 Hz)
    dt_control = 0.1, # Control update rate (10 Hz)

    dt_propagation= 0.01 # Propagation time step (100 Hz)
)

final_approach_phase = MissionPhase(
    name="final_approach",    
    navigation= IdealNavigation(),
    
    translational_model=translational_propagator.copy(),
    guidance =CustomGuidanceLaw(custom_reference_function=final_approach_guidance),
    controller= CustomController(control_function=PD_controller),
    allocator=actuator_allocator.copy(),    

    dt_nav = 0.1,  # Navigation update rate (10 Hz)
    dt_guid = 0.1,  # Guidance update rate (1 Hz)
    dt_control = 0.1, # Control update rate (10 Hz)

    dt_propagation= 0.01 # Propagation time step (100 Hz)
)

from py.modules.general_tools import get_state_at

time_of_first_trheshold = None
time_of_relative_hold_transition = None

time_of_final_approach_transition = None

def docking_axis_to_relative_hold_transition(simulation_data):
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
        print(f"Relative distance magnitude: {relative_distance_magnitude:.2f} m, relative_state_vector is {chaser_pos - target_pos}, at t={simulation_data.spacecrafts[0].spacecraft_data.t:.2f} s")
        
        
        if relative_distance_magnitude <= 10.0 and time_of_first_trheshold is None:  # Transition when the relative distance is less than or equal to 8 meters
            time_of_first_trheshold = t

        if time_of_first_trheshold is not None:
            if t - time_of_first_trheshold >= 70:
                time_of_relative_hold_transition = t        
                return True

def relative_hold_to_final_approach_transition(simulation_data):
    if time_of_relative_hold_transition is not None:
        elapsed_time = simulation_data.spacecrafts[0].spacecraft_data.t - time_of_relative_hold_transition
        if elapsed_time >= 60*3:  # Transition after 3 minutes in relative hold
            global time_of_final_approach_transition
            time_of_final_approach_transition = simulation_data.spacecrafts[0].spacecraft_data.t
            return True

mission_manager = MissionManager(
    initial_phase=docking_axis_transition,
    verbose=True
)
mission_manager.add_phases(
    [relative_hold_phase,
    final_approach_phase]
)

mission_manager.add_transition(
    from_phase=docking_axis_transition,
    target_phase=relative_hold_phase,
    condition=docking_axis_to_relative_hold_transition
)

mission_manager.add_transition(
    from_phase=relative_hold_phase,
    target_phase=final_approach_phase,
    condition=relative_hold_to_final_approach_transition
)

# ========================================================
#           CHASER SPACECRAFT CONFIGURATION
# ========================================================

# FROM NRHO Pos final: [ 3.79543557e+08 -9.91715693e+03  3.33735189e+06]
# FROM NRHO Vel final: [-1.65568984e-01  1.68125627e+03  2.58673649e+00]

pos_from_NRHO = np.array([3.79543557e+08, -9.91715693e+03, 3.33735189e+06])
vel_from_NRHO = np.array([-1.65568984e-01, 1.68125627e+03, 2.58673649e+00])

chaser_initial_orientation = [0, 0, 0]  # Initial orientation in Euler angles (degrees)
chaser_initial_orientation_quat = quaternion_from_euler(
    np.deg2rad(chaser_initial_orientation[0]),
    np.deg2rad(chaser_initial_orientation[1]),
    np.deg2rad(chaser_initial_orientation[2])
)
chaser_initial_angular_velocity = np.array([0.0, 0.0, 0.0])  # Initial angular velocity in rad/s



chaser_initial_position = (pos_from_NRHO -50)   # Initial position in meters
chaser_initial_velocity = vel_from_NRHO  # Initial velocity in meters per second

inertial_tensor = np.diag([Ix_total, Iy_total, Iz_total])  # Inertia tensor in kg*m^2


target_initial_attitude = np.array([45, 45, 0])  # Target initial orientation in Euler angles (degrees)
target_initial_attitude_quat = quaternion_from_euler(
    np.deg2rad(target_initial_attitude[0]),
    np.deg2rad(target_initial_attitude[1]),
    np.deg2rad(target_initial_attitude[2])
)

sim.add_spacecraft(
    name="Target",    
    initial_position= pos_from_NRHO,  # Target position in meters
    initial_velocity= vel_from_NRHO,  # Target stationary    
    initial_attitude= target_initial_attitude_quat,
    sensors=[AbsoluteSensor(sensors_sample_rate)],    
    mission_manager=MissionManager(initial_phase= MissionPhase(
        name="target_phase",
        translational_model=NativeTranslationalPropagator(dynamics="CR3BP"),
        navigation=IdealNavigation(),
        dt_nav=1,
        dt_propagation=0.01
    )),    
    verbose=False
)

sim.add_spacecraft(
    name="Chaser",
    mass=CM_mass + SM_mass,
    initial_position=chaser_initial_position,
    initial_velocity=chaser_initial_velocity,
    initial_attitude=chaser_initial_orientation_quat,
    initial_angular_velocity=chaser_initial_angular_velocity,
    inertia_tensor=inertial_tensor,
    sensors=[accel_1, gyro_1, abs_sensor_1],
    actuators=actuators,
    mission_manager=mission_manager,
    verbose=False,
    target_name="Target"
)



result = sim.simulate()

from py.modules.visualization.state_variables import *
from py.modules.visualization.gnc import plot_control_result
    
from py.modules.visualization.style_dark import *


def plot_relative_position(
    spacecraft_names: str | list[str],
    result,
    save: bool = False,
    show: bool = True,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    filename: str = "relative_position.png",
    dpi: int = 300,
):
    """
    Plot relative position between two spacecraft.

    The relative position is defined as:

        r_relative = r_chaser - r_target

    Parameters
    ----------
    spacecraft_names : str | list[str]
        Must contain exactly two spacecraft names.
        The first spacecraft is treated as the chaser and
        the second spacecraft as the target.

    result : SimulationHistory
        General simulation result containing spacecraft histories.

    save : bool, optional
        Save the generated figure to disk. Default is False.

    show : bool, optional
        Display the generated figure. Default is True.

    output_dir : Path | str, optional
        Directory where the figure will be saved.

    filename : str, optional
        Output filename.

    dpi : int, optional
        Figure resolution in dots per inch.

    Returns
    -------
    matplotlib.figure.Figure
        Generated figure.
    """

    # ------------------------------------------------------------------
    # Normalize spacecraft input
    # ------------------------------------------------------------------

    if isinstance(spacecraft_names, str):
        spacecraft_names = [spacecraft_names]

    if len(spacecraft_names) != 2:
        raise ValueError(
            "Exactly two spacecraft names must be provided: "
            "[chaser, target]."
        )

    chaser_name, target_name = spacecraft_names

    # ------------------------------------------------------------------
    # Validate spacecrafts
    # ------------------------------------------------------------------

    for name in spacecraft_names:

        if name not in result.spacecrafts_history:
            raise KeyError(
                f"Spacecraft '{name}' not found in simulation history."
            )

    # ------------------------------------------------------------------
    # Retrieve histories
    # ------------------------------------------------------------------

    chaser_history = result.spacecrafts_history[chaser_name]
    target_history = result.spacecrafts_history[target_name]

    # ------------------------------------------------------------------
    # Time
    # ------------------------------------------------------------------

    chaser_t = np.asarray(chaser_history.t)
    target_t = np.asarray(target_history.t)

    if len(chaser_t) != len(target_t):

        raise ValueError(
            "Chaser and target histories have different lengths: "
            f"{chaser_name}={len(chaser_t)}, "
            f"{target_name}={len(target_t)}."
        )

    if not np.allclose(chaser_t, target_t):

        raise ValueError(
            "Chaser and target histories do not have matching time vectors."
        )

    t = chaser_t

    # ------------------------------------------------------------------
    # Position
    # ------------------------------------------------------------------

    chaser_position = np.asarray(
        chaser_history.true_state["position"]
    )

    target_position = np.asarray(
        target_history.true_state["position"]
    )

    # ------------------------------------------------------------------
    # Relative position
    #
    # r_C/T = r_C - r_T
    # ------------------------------------------------------------------

    relative_position = (
        chaser_position
        - target_position
    )

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(8, 10),
        sharex=True,
        squeeze=False,
    )

    axes = axes[:, 0]

    style_figure(fig)

    labels = ("X", "Y", "Z")

    line_colors = (
        LINE_COLOR,
        SECONDARY_COLOR,
        ACCENT_COLOR,
    )

    # ------------------------------------------------------------------
    # Plot relative position
    # ------------------------------------------------------------------

    for i, label in enumerate(labels):

        ax = axes[i]

        ax.plot(
            t,
            relative_position[:, i],
            color=line_colors[i],
            linewidth=1.4,
            label=rf"$\Delta r_{label.lower()}$",
        )

        if time_of_relative_hold_transition is not None:
            ax.axvline(time_of_relative_hold_transition, color="gray", linestyle="--", linewidth=1.0, label="Relative Hold Transition")
        if time_of_final_approach_transition is not None:
            ax.axvline(time_of_final_approach_transition, color="black", linestyle="--", linewidth=1.0, label="Final Approach Transition")

        style_axes(
            ax,
            title=f"Relative Position {label}",
            ylabel=rf"$\Delta r_{label.lower()}$ [m]",
        )

        ax.legend(
            fontsize=7,
            labelcolor=TEXT_COLOR,
            edgecolor="none",
            facecolor=AXES_BG,
        )

    axes[-1].set_xlabel("Time [s]")

    # ------------------------------------------------------------------
    # Figure title
    # ------------------------------------------------------------------

    fig.suptitle(
        f"Relative Position: {chaser_name} - {target_name}",
        fontsize=13,
        y=0.99,
    )

    fig.tight_layout(
        rect=[0, 0, 1, 0.96]
    )

    

    # ------------------------------------------------------------------
    # Finalize
    # ------------------------------------------------------------------

    return finalize_figure(
        fig,
        show=show,
        save=save,
        output_dir=output_dir,
        filename=filename,
        dpi=dpi,
    )

chaser_position = result.spacecrafts_history["Chaser"].true_state["position"][-1]
target_position = result.spacecrafts_history["Target"].true_state["position"][-1]

relative_position = chaser_position - target_position

target_position_relative = np.array([0.0, 0.0, 10.0])

error = relative_position - target_position_relative

# Escala de referencia para X/Y
xy_reference = 50.0

error_percentage = np.array([
    abs(error[0]) / xy_reference * 100,
    abs(error[1]) / xy_reference * 100,
    abs(error[2]) / abs(target_position_relative[2]) * 100
])

print(f"Relative position : {relative_position} m")
print(f"Target position   : {target_position_relative} m")
print(f"Error             : {error} m")
print(f"Error percentage  : {error_percentage} %")
print(f"Error magnitude   : {np.linalg.norm(error):.3f} m")

relative_position = plot_relative_position(["Chaser", "Target"], result)

plot_trajectory(["Chaser", "Target"], result, show=True,)
plot_position(["Chaser", "Target"], result)
plot_control_result("Chaser", result)