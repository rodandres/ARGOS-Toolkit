import numpy as np

from argos.enviroments.environments import ClassicalEnvironment
from argos.core.simulation import Simulation

from argos.actuators.RCS import RCSThruster
from argos.sensors.inertial_sensors import Gyroscope

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

# Initial angular velocity expressed in the propagation frame.
# The assumed propagation frame is inertial.
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
# but no control law commands them.
#
# Their nominal thrust and torque are zero, so the experiment
# assumes that no actuator torque is applied.
#
# With a spherical inertia tensor and no external torque,
# the expected angular velocity remains constant.
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
# 4. Gyroscopes
# ============================================================
#
# All error models are disabled to obtain deterministic
# measurements.
#
# GYRO_CG:
#   Located at the spacecraft center of mass.
#   Aligned with the body frame.
#
# GYRO_ROTATED:
#   Same location, but rotated 90 degrees around Z.
#
# GYRO_ROTATED_X:
#   Same location, but rotated 90 degrees around X.
#
# The position does not affect the ideal angular velocity
# measurement of a rigid-body gyroscope.
# ============================================================

gyro_cg = Gyroscope(
    position=np.array([0.0, 0.0, 0.0]),
    rotation=np.array([0.0, 0.0, 0.0]),
    bias=0.0,
    noise_std=0.0,
    random_walk_std=0.0,
    saturation_limit=np.inf,
    quantization_bits=None,
    sample_rate_freq=100.0,
    name="GYRO_CG",
)

gyro_rotated = Gyroscope(
    position=np.array([0.0, 0.0, 0.0]),
    rotation=np.array([0.0, 0.0, 90.0]),
    bias=0.0,
    noise_std=0.0,
    random_walk_std=0.0,
    saturation_limit=np.inf,
    quantization_bits=None,
    sample_rate_freq=100.0,
    name="GYRO_ROTATED",
)

gyro_rotated_x = Gyroscope(
    position=np.array([0.0, 0.0, 0.0]),
    rotation=np.array([90.0, 0.0, 0.0]),
    bias=0.0,
    noise_std=0.0,
    random_walk_std=0.0,
    saturation_limit=np.inf,
    quantization_bits=None,
    sample_rate_freq=100.0,
    name="GYRO_ROTATED_X",
)

sensors = [
    gyro_cg,
    gyro_rotated,
    gyro_rotated_x,
]


# ============================================================
# 5. Rotational propagator
# ============================================================

rot_propagator = NativeRotationalPropagator(
    integration_method="CPP_RK45",
    propagation_frame=ENU_inertial_frame,
)


# ============================================================
# 6. Navigation law to inspect gyro measurements
# ============================================================

from argos.navigation.basic_laws import CustomNavigation
from argos.general.dataclasses import NavigationOutput, StateVariables


def estimation_func(sensors: list):

    measurements = {}

    for sensor in sensors:
        measurements[sensor.name] = sensor.get_measurement()

    print(f"GYRO_CG:        {measurements['GYRO_CG']}")
    print(f"GYRO_ROTATED:   {measurements['GYRO_ROTATED']}")
    print(f"GYRO_ROTATED_X: {measurements['GYRO_ROTATED_X']}")

    # Store measurements for inspection.
    #
    # These assignments are only a convenient way to expose
    # the measurements through the navigation output. They do
    # not represent a physical position, velocity, or acceleration
    # estimate.

    state_variables = StateVariables(
        position=measurements["GYRO_CG"],
        velocity=measurements["GYRO_ROTATED"],
        acceleration=measurements["GYRO_ROTATED_X"],
    )

    nav_output = NavigationOutput(
        spacecraft_state=state_variables,
    )

    return nav_output


nav_law = CustomNavigation(estimation_func)


# ============================================================
# 7. Mission phase and mission manager
# ============================================================

phase = MissionPhase(
    name="GyroscopeValidation",
    rotational_model=rot_propagator,
    navigation=nav_law,
    dt_nav=0.01,
    dt_propagation=0.01,
)

mission_manager = MissionManager(
    initial_phase=phase,
)


# ============================================================
# 8. Spacecraft initialization
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
# 9. Run simulation
# ============================================================

metadata = sim.simulate()