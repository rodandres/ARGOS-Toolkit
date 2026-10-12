import numpy as np

from argos.enviroments.environments import ClassicalEnvironment
from argos.core.simulation import Simulation

from argos.actuators.RCS import RCSThruster
from argos.sensors.inertial_sensors import Accelerometer

from argos.propagators.native_propagator import NativeRotationalPropagator
from argos.frames.frames import ENU_inertial_frame

from argos.general.dataclasses import MissionPhase
from argos.core.mission_manager import MissionManager

from argos.math import quaternion_from_euler


# ============================================================
# 1. Simulation setup
# ============================================================

env = ClassicalEnvironment()

sim = Simulation(
    max_sim_time=10.0,
    environment=env,
    verbose=True,
)


# ============================================================
# 2. Spacecraft initial conditions
# ============================================================

inertia_tensor = np.diag([10.0, 10.0, 10.0])

initial_orientation = np.array([0.0, 0.0, 0.0])

initial_attitude = quaternion_from_euler(
    np.deg2rad(initial_orientation[0]),
    np.deg2rad(initial_orientation[1]),
    np.deg2rad(initial_orientation[2]),
)

initial_angular_velocity = np.array([
    0.0,
    0.0,
    0.1,
])


# ============================================================
# 3. Actuators
# ============================================================
#
# The actuators are included in the spacecraft configuration,
# but no control law is configured to command them.
#
# Therefore, the experiment assumes that no external torque
# is applied by these actuators.
#
# The zero nominal thrust also avoids introducing thrust forces.
# ============================================================

thrusters = [
    RCSThruster(
        nominal_thrust=0.0,
        direction=np.array([1.0, 0.0, 0.0]),
        override_torque_value=0.0,
    ),
    RCSThruster(
        nominal_thrust=0.0,
        direction=np.array([-1.0, 0.0, 0.0]),
        override_torque_value=0.0,
    ),
    RCSThruster(
        nominal_thrust=0.0,
        direction=np.array([0.0, 1.0, 0.0]),
        override_torque_value=0.0,
    ),
    RCSThruster(
        nominal_thrust=0.0,
        direction=np.array([0.0, -1.0, 0.0]),
        override_torque_value=0.0,
    ),
    RCSThruster(
        nominal_thrust=0.0,
        direction=np.array([0.0, 0.0, 1.0]),
        override_torque_value=0.0,
    ),
    RCSThruster(
        nominal_thrust=0.0,
        direction=np.array([0.0, 0.0, -1.0]),
        override_torque_value=0.0,
    ),
]


# ============================================================
# 4. Accelerometers
# ============================================================
#
# All error models are disabled to make the expected results
# deterministic.
# ============================================================

imu_cg = Accelerometer(
    position=np.array([0.0, 0.0, 0.0]),
    rotation=np.array([0.0, 0.0, 0.0]),
    bias=0.0,
    noise_std=0.0,
    random_walk_std=0.0,
    saturation_limit=np.inf,
    quantization_bits=None,
    sample_rate_freq=100.0,
    name="IMU_CG",
)

imu_x = Accelerometer(
    position=np.array([1.0, 0.0, 0.0]),
    rotation=np.array([0.0, 0.0, 0.0]),
    bias=0.0,
    noise_std=0.0,
    random_walk_std=0.0,
    saturation_limit=np.inf,
    quantization_bits=None,
    sample_rate_freq=100.0,
    name="IMU_X",
)

imu_offset = Accelerometer(
    position=np.array([0.5, 0.2, 0.1]),
    rotation=np.array([0.0, 0.0, 90.0]),
    bias=0.0,
    noise_std=0.0,
    random_walk_std=0.0,
    saturation_limit=np.inf,
    quantization_bits=None,
    sample_rate_freq=100.0,
    name="IMU_OFFSET",
)

sensors = [
    imu_cg,
    imu_x,
    imu_offset,
]


# ============================================================
# 5. Rotational propagator
# ============================================================

rot_propagator = NativeRotationalPropagator(
    integration_method="CPP_RK45",
    propagation_frame=ENU_inertial_frame,
)


#Navigation law to store sensors data:

from argos.navigation.basic_laws import CustomNavigation
from argos.general.dataclasses import NavigationOutput, StateVariables
def estimation_func(sensors: list):
    for sensor in sensors:
        if sensor.name == "IMU_CG":            
            imu_cg_data = sensor.get_measurement()
        elif sensor.name == "IMU_X":
            imu_x_data = sensor.get_measurement()
        elif sensor.name == "IMU_OFFSET":
            imu_offset_data = sensor.get_measurement()

    # STORE measurements as other variables names

    print(f"IMU_CG: {imu_cg_data}")
    print(f"IMU_X: {imu_x_data}")
    print(f"IMU_OFFSET: {imu_offset_data}")
    state_variables = StateVariables(
        position=imu_cg_data,
        velocity=imu_x_data,
        acceleration=imu_offset_data,
    )

    nav_output = NavigationOutput(
        spacecraft_state=state_variables,
    )

    return nav_output


nav_law = CustomNavigation(estimation_func)


# ============================================================
# 6. Mission phase and mission manager
# ============================================================
#
# This phase has no guidance, navigation, or control law.
# If MissionPhase requires these fields in your ARGOS version,
# provide the corresponding no-op implementations.
# ============================================================

phase = MissionPhase(
    name="IMUValidation",
    rotational_model=rot_propagator,
    navigation = nav_law,
    dt_nav= 0.01,
    dt_propagation=0.01,
)

mission_manager = MissionManager(
    initial_phase=phase,
)


# ============================================================
# 7. Spacecraft initialization
# ============================================================

sim.add_spacecraft(
    initial_attitude=initial_attitude,
    initial_angular_velocity=initial_angular_velocity,
    inertia_tensor=inertia_tensor,
    actuators=thrusters,
    sensors=sensors,
    mission_manager=mission_manager,
    mass=100.0,
    verbose=True,
)


# ============================================================
# 8. Run simulation
# ============================================================

metadata = sim.simulate()


# ============================================================
# 9. Inspect sensor measurements
# ============================================================
#
# The exact measurement-history accessor depends on how
# SensorBase stores measurements in your ARGOS version.
#
# During the simulation, inspect:
#
#     imu_cg.get_measurement()
#     imu_x.get_measurement()
#     imu_offset.get_measurement()
#
# to confirm the output format and sampling history.
# ============================================================