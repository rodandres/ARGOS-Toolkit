import numpy as np
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
from py.general.general_data import CM_mass, SM_mass, Ix_total, Iy_total, Iz_total
from py.modules.visualization.trajectories import plot_trajectory, plot_trajectory_3d

sim_max_time = 60*10
sim = Simulation(max_sim_time=sim_max_time,
                 environment= ClassicalEnvironment(),
                 verbose = True 
)

# ========================================================
#   ACTUATORS (2 SETS - Translational and Rotational)
# ========================================================
actuators = []
scale_factor = 0.1

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
translational_propagator = NativeTranslationalPropagator()
rotational_propagator = NativeRotationalPropagator()


from py.general.dataclasses import GuidanceReference, StateVariables, ControlOutput

def docking_axis_guidance(navigation_estimated_data, simulation_data):
    spacecraft_state = navigation_estimated_data.spacecraft_state

    # Información definida por el problema
    target_docking_axis_wrt_target_frame = np.array([1.0, 0.0, 0.0])  # Docking axis of the target in the target frame
    target_complementary_axis_wrt_target_frame = np.array([0.0, 1.0, 0.0])  # Complementary axis of the target in the target frame
    chaser_docking_axis_wrt_chaser_frame = np.array([1.0, 0.0, 0.0])  # Docking axis of the chaser in the chaser frame
    chaser_complementary_axis_wrt_chaser_frame = np.array([0.0, 1.0, 0.0])  # Complementary axis of the chaser in the chaser frame
    distance_along_docking_axis = 10.0  # Distance along the docking axis in meters from the target's docking port
    interception_point_wrt_target_frame = distance_along_docking_axis * target_docking_axis_wrt_target_frame  # Interception point in the target frame

    # Información actual del objetivo y del perseguidor
    target_current_position_wrt_inertal_frame = np.array([0.0, 0.0, 0.0])  # Current target position in meters wrt the inertial frame
    target_current_attitude_wrt_inertal_frame = np.array([0.0, 0.0, 0.0])  # Target attitude in degrees wrt the inertial frame
    target_current_attitude_wrt_inertal_frame = np.radians(target_current_attitude_wrt_inertal_frame)  # Convert target attitude to radians
    target_current_quaternion_wrt_inertal_frame = quaternion_from_euler(*target_current_attitude_wrt_inertal_frame)  # Convert target attitude to quaternion

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


unique_phase = MissionPhase(
    name="unique_phase",
    guidance= CustomGuidanceLaw(custom_reference_function=docking_axis_guidance),
    navigation= IdealNavigation(),
    controller= CustomController(control_function=PD_controller),
    allocator=actuator_allocator,
    translational_model=translational_propagator,
    rotational_model=rotational_propagator,

    dt_nav = 0.1,  # Navigation update rate (10 Hz)
    dt_guid = 1,  # Guidance update rate (1 Hz)
    dt_control = 0.1, # Control update rate (10 Hz)

    dt_propagation= 0.01 # Propagation time step (100 Hz)
)

mission_manager = MissionManager(
    initial_phase=unique_phase,
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
chaser_initial_position = np.array([50, -20, 35])  # Initial position in meters
chaser_initial_velocity = np.array([0, 0, 0])  # Initial velocity in meters per second

inertial_tensor = np.diag([Ix_total, Iy_total, Iz_total])  # Inertia tensor in kg*m^2

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
    verbose=False
)

result = sim.simulate()

from py.modules.visualization.state_variables import plot_attitude_quaternions, plot_angular_velocity, plot_position
from py.modules.visualization.gnc import plot_control_result
    

plot_trajectory("Chaser", result, show=True,)

plot_position("Chaser", result)

plot_attitude_quaternions("Chaser", result)
plot_angular_velocity("Chaser", result)
plot_control_result("Chaser", result)