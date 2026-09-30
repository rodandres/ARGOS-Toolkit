import numpy as np
from types import SimpleNamespace

from argos.sensors.inertial_sensors import (
    Accelerometer,
    Gyroscope,
)


# ============================================================================
# Helpers
# ============================================================================


def make_spacecraft_data(
    acceleration=(0.0, 0.0, 0.0),
    angular_velocity=(0.0, 0.0, 0.0),
    angular_acceleration=(0.0, 0.0, 0.0),
):
    return SimpleNamespace(
        true_accel=np.asarray(acceleration, dtype=float),
        true_omega=np.asarray(angular_velocity, dtype=float),
        true_alpha=np.asarray(angular_acceleration, dtype=float),
    )


def make_simulation_data(
    gravity_accel=(0.0, 0.0, 0.0),
    dcm=None,
):
    if dcm is None:
        dcm = np.eye(3)

    return SimpleNamespace(
        gravity_accel=np.asarray(gravity_accel, dtype=float),
        DCM_inertial_to_body=np.asarray(dcm, dtype=float),
    )


# ============================================================================
# Accelerometer
# ============================================================================


def test_accelerometer_specific_acceleration():
    """
    Specific acceleration is the true inertial acceleration minus gravity.
    """
    sensor = Accelerometer(
        noise_std=0.0,
        bias=0.0,
        random_walk_std=0.0,
        saturation_limit=np.inf,
        quantization_bits=None,
    )

    spacecraft_data = make_spacecraft_data(
        acceleration=[5.0, 2.0, -1.0],
    )

    simulation_data = make_simulation_data(
        gravity_accel=[1.0, 0.5, -0.5],
    )

    result = sensor.compute_specific_accel(
        spacecraft_data,
        simulation_data,
    )

    assert np.allclose(
        result,
        [4.0, 1.5, -0.5],
    )


def test_accelerometer_gravity_and_spacecraft_rotation():
    """
    Gravity subtraction must happen in the inertial frame before the
    specific force is transformed into the body frame.
    """
    sensor = Accelerometer(
        position=np.zeros(3),
        rotation=np.zeros(3),
        noise_std=0.0,
        bias=0.0,
        random_walk_std=0.0,
        saturation_limit=np.inf,
        quantization_bits=None,
    )

    spacecraft_data = make_spacecraft_data(
        acceleration=[2.0, 0.0, 0.0],
    )

    # Body frame is rotated -90 deg with respect to the inertial frame.
    dcm = np.array([
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ])

    simulation_data = make_simulation_data(
        gravity_accel=[1.0, 0.0, 0.0],
        dcm=dcm,
    )

    result = sensor.get_ideal_measurement(
        spacecraft_data,
        simulation_data,
    )

    # Specific force in inertial frame:
    #
    # [2, 0, 0] - [1, 0, 0] = [1, 0, 0]
    #
    # After inertial -> body transformation:
    #
    # [0, -1, 0]
    expected = np.array([0.0, -1.0, 0.0])

    assert np.allclose(result, expected)


def test_accelerometer_identity_transform():
    """
    With identity attitude and identity sensor orientation, the vector
    must remain unchanged.
    """
    sensor = Accelerometer(
        position=np.zeros(3),
        rotation=np.zeros(3),
    )

    spacecraft_data = make_spacecraft_data()

    simulation_data = make_simulation_data()

    accel = np.array([1.0, 2.0, 3.0])

    result = sensor.transform_to_sensor_frame(
        accel,
        spacecraft_data,
        simulation_data,
    )

    assert np.allclose(result, accel)


def test_accelerometer_spacecraft_rotation():
    """
    The accelerometer must correctly transform inertial acceleration
    into the spacecraft body frame.
    """
    sensor = Accelerometer(
        position=np.zeros(3),
        rotation=np.zeros(3),
    )

    spacecraft_data = make_spacecraft_data()

    dcm = np.array([
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ])

    simulation_data = make_simulation_data(
        dcm=dcm,
    )

    accel = np.array([1.0, 0.0, 0.0])

    result = sensor.transform_to_sensor_frame(
        accel,
        spacecraft_data,
        simulation_data,
    )

    expected = np.array([0.0, -1.0, 0.0])

    assert np.allclose(result, expected)


def test_accelerometer_sensor_rotation():
    """
    The accelerometer must account for its own orientation relative
    to the spacecraft body frame.
    """
    sensor = Accelerometer(
        position=np.zeros(3),
        rotation=[0.0, 0.0, 90.0],
    )

    spacecraft_data = make_spacecraft_data()

    simulation_data = make_simulation_data()

    accel = np.array([1.0, 0.0, 0.0])

    result = sensor.transform_to_sensor_frame(
        accel,
        spacecraft_data,
        simulation_data,
    )

    expected = np.array([0.0, -1.0, 0.0])

    assert np.allclose(result, expected)


def test_accelerometer_zero_rotation_has_no_lever_arm_effect():
    """
    A sensor displaced from the center of mass has no rotational
    acceleration contribution when omega and alpha are zero.
    """
    sensor = Accelerometer(
        position=[1.0, 2.0, 3.0],
    )

    spacecraft_data = make_spacecraft_data(
        angular_velocity=[0.0, 0.0, 0.0],
        angular_acceleration=[0.0, 0.0, 0.0],
    )

    simulation_data = make_simulation_data()

    accel = np.array([1.0, 2.0, 3.0])

    result = sensor.transform_to_sensor_frame(
        accel,
        spacecraft_data,
        simulation_data,
    )

    assert np.allclose(result, accel)


def test_accelerometer_tangential_lever_arm_acceleration():
    """
    Verify the alpha x r tangential acceleration term.
    """
    sensor = Accelerometer(
        position=[1.0, 0.0, 0.0],
    )

    spacecraft_data = make_spacecraft_data(
        angular_velocity=[0.0, 0.0, 0.0],
        angular_acceleration=[0.0, 0.0, 2.0],
    )

    simulation_data = make_simulation_data()

    accel = np.zeros(3)

    result = sensor.transform_to_sensor_frame(
        accel,
        spacecraft_data,
        simulation_data,
    )

    expected = np.array([0.0, 2.0, 0.0])

    assert np.allclose(result, expected)


def test_accelerometer_centripetal_lever_arm_acceleration():
    """
    Verify the omega x (omega x r) centripetal acceleration term.
    """
    sensor = Accelerometer(
        position=[1.0, 0.0, 0.0],
    )

    spacecraft_data = make_spacecraft_data(
        angular_velocity=[0.0, 0.0, 2.0],
        angular_acceleration=[0.0, 0.0, 0.0],
    )

    simulation_data = make_simulation_data()

    accel = np.zeros(3)

    result = sensor.transform_to_sensor_frame(
        accel,
        spacecraft_data,
        simulation_data,
    )

    expected = np.array([-4.0, 0.0, 0.0])

    assert np.allclose(result, expected)


def test_accelerometer_combined_rotational_acceleration():
    """
    Verify the complete rotational acceleration contribution:

        alpha x r + omega x (omega x r)
    """
    sensor = Accelerometer(
        position=[1.0, 0.0, 0.0],
    )

    spacecraft_data = make_spacecraft_data(
        angular_velocity=[0.0, 0.0, 2.0],
        angular_acceleration=[0.0, 0.0, 2.0],
    )

    simulation_data = make_simulation_data()

    accel = np.zeros(3)

    result = sensor.transform_to_sensor_frame(
        accel,
        spacecraft_data,
        simulation_data,
    )

    # Tangential:
    # alpha x r = [0, 2, 0]
    #
    # Centripetal:
    # omega x (omega x r) = [-4, 0, 0]
    #
    # Total:
    # [-4, 2, 0]
    expected = np.array([-4.0, 2.0, 0.0])

    assert np.allclose(result, expected)


def test_accelerometer_complete_physical_case():
    """
    Test the complete accelerometer pipeline:

        true acceleration
        - gravity
        -> specific acceleration
        -> inertial to body transformation
        -> rotational lever-arm acceleration
        -> body to sensor transformation
    """
    sensor = Accelerometer(
        position=[1.0, 0.0, 0.0],
        rotation=[0.0, 0.0, 90.0],
        noise_std=0.0,
        bias=0.0,
        random_walk_std=0.0,
        saturation_limit=np.inf,
        quantization_bits=None,
    )

    spacecraft_data = make_spacecraft_data(
        acceleration=[3.0, 2.0, 0.0],
        angular_velocity=[0.0, 0.0, 2.0],
        angular_acceleration=[0.0, 0.0, 2.0],
    )

    dcm_inertial_to_body = np.array([
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ])

    simulation_data = make_simulation_data(
        gravity_accel=[1.0, 1.0, 0.0],
        dcm=dcm_inertial_to_body,
    )

    result = sensor.get_ideal_measurement(
        spacecraft_data,
        simulation_data,
    )

    # Specific acceleration in inertial frame:
    #
    # [3, 2, 0] - [1, 1, 0] = [2, 1, 0]
    #
    # In body frame:
    #
    # C_BI @ [2, 1, 0] = [1, -2, 0]
    #
    # Rotational contribution:
    #
    # alpha x r = [0, 2, 0]
    # omega x (omega x r) = [-4, 0, 0]
    #
    # Total body-frame acceleration:
    #
    # [1, -2, 0] + [-4, 2, 0] = [-3, 0, 0]
    #
    # Sensor is rotated +90 deg around Z, so:
    #
    # body -> sensor:
    # [-3, 0, 0] -> [0, 3, 0]
    expected = np.array([0.0, 3.0, 0.0])

    assert np.allclose(result, expected)


# ============================================================================
# Gyroscope
# ============================================================================


def test_gyroscope_identity_transform():
    """
    With identity spacecraft attitude and sensor orientation,
    angular velocity must remain unchanged.
    """
    sensor = Gyroscope(
        rotation=np.zeros(3),
    )

    simulation_data = make_simulation_data()

    gyro = np.array([1.0, 2.0, 3.0])

    result = sensor.transform_to_sensor_frame(
        gyro,
        simulation_data,
    )

    assert np.allclose(result, gyro)


def test_gyroscope_transforms_inertial_to_body():
    """
    Verify the inertial-to-body transformation.
    """
    sensor = Gyroscope(
        rotation=np.zeros(3),
    )

    dcm = np.array([
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ])

    simulation_data = make_simulation_data(
        dcm=dcm,
    )

    gyro = np.array([1.0, 0.0, 0.0])

    result = sensor.transform_to_sensor_frame(
        gyro,
        simulation_data,
    )

    expected = np.array([0.0, -1.0, 0.0])

    assert np.allclose(result, expected)


def test_gyroscope_sensor_rotation():
    """
    The gyroscope must account for its own orientation relative
    to the spacecraft body frame.
    """
    sensor = Gyroscope(
        rotation=[0.0, 0.0, 90.0],
    )

    simulation_data = make_simulation_data()

    gyro = np.array([1.0, 0.0, 0.0])

    result = sensor.transform_to_sensor_frame(
        gyro,
        simulation_data,
    )

    expected = np.array([0.0, -1.0, 0.0])

    assert np.allclose(result, expected)


def test_gyroscope_spacecraft_and_sensor_rotation():
    """
    Verify the complete inertial -> body -> sensor transformation.
    """
    sensor = Gyroscope(
        rotation=[0.0, 0.0, 90.0],
    )

    dcm = np.array([
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ])

    simulation_data = make_simulation_data(
        dcm=dcm,
    )

    gyro = np.array([1.0, 0.0, 0.0])

    result = sensor.transform_to_sensor_frame(
        gyro,
        simulation_data,
    )

    # Inertial -> body:
    #
    # [1, 0, 0] -> [0, -1, 0]
    #
    # Body -> sensor:
    #
    # [0, -1, 0] -> [-1, 0, 0]
    expected = np.array([-1.0, 0.0, 0.0])

    assert np.allclose(result, expected)


def test_gyroscope_get_ideal_measurement():
    """
    The ideal gyroscope measurement must correspond to the spacecraft
    angular velocity when both spacecraft and sensor frames are aligned.
    """
    sensor = Gyroscope(
        rotation=np.zeros(3),
    )

    spacecraft_data = make_spacecraft_data(
        angular_velocity=[0.1, 0.2, 0.3],
    )

    simulation_data = make_simulation_data()

    result = sensor.get_ideal_measurement(
        spacecraft_data,
        simulation_data,
    )

    assert np.allclose(
        result,
        [0.1, 0.2, 0.3],
    )


def test_gyroscope_sensor_measurement_is_independent_of_sensor_position():
    """
    A gyroscope measures angular velocity and therefore its ideal
    measurement should not depend on its translational position.
    """
    sensor = Gyroscope(
        position=[10.0, -5.0, 2.0],
        rotation=np.zeros(3),
    )

    spacecraft_data = make_spacecraft_data(
        angular_velocity=[0.1, 0.2, 0.3],
    )

    simulation_data = make_simulation_data()

    result = sensor.get_ideal_measurement(
        spacecraft_data,
        simulation_data,
    )

    assert np.allclose(
        result,
        [0.1, 0.2, 0.3],
    )