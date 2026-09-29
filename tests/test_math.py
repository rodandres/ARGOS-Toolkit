import numpy as np

from argos.math import (
    DCM_to_quaternion,
    euler_from_quaternion,
    quaternion_error,
    quaternion_from_euler,
    quaternion_multiply,
    quaternion_to_DCM,
)


def test_quaternion_from_euler_identity():
    result = quaternion_from_euler(
        0.0,
        0.0,
        0.0,
    )

    expected = np.array([
        0.0,
        0.0,
        0.0,
        1.0,
    ])

    assert np.allclose(result, expected)


def test_quaternion_from_euler_roll_90_degrees():
    result = quaternion_from_euler(
        np.pi / 2,
        0.0,
        0.0,
    )

    expected = np.array([
        np.sqrt(0.5),
        0.0,
        0.0,
        np.sqrt(0.5),
    ])

    assert np.allclose(result, expected)


def test_quaternion_from_euler_pitch_90_degrees():
    result = quaternion_from_euler(
        0.0,
        np.pi / 2,
        0.0,
    )

    expected = np.array([
        0.0,
        np.sqrt(0.5),
        0.0,
        np.sqrt(0.5),
    ])

    assert np.allclose(result, expected)


def test_quaternion_from_euler_yaw_90_degrees():
    result = quaternion_from_euler(
        0.0,
        0.0,
        np.pi / 2,
    )

    expected = np.array([
        0.0,
        0.0,
        np.sqrt(0.5),
        np.sqrt(0.5),
    ])

    assert np.allclose(result, expected)


def test_euler_quaternion_round_trip():
    expected = np.array([
        0.3,
        -0.4,
        1.2,
    ])

    quaternion = quaternion_from_euler(*expected)

    result = euler_from_quaternion(quaternion)

    assert np.allclose(result, expected)


def test_quaternion_to_DCM_identity():
    quaternion = np.array([
        0.0,
        0.0,
        0.0,
        1.0,
    ])

    result = quaternion_to_DCM(quaternion)

    expected = np.eye(3)

    assert np.allclose(result, expected)


def test_quaternion_to_DCM_returns_valid_rotation_matrix():
    quaternion = quaternion_from_euler(
        0.3,
        -0.4,
        1.2,
    )

    DCM = quaternion_to_DCM(quaternion)

    assert DCM.shape == (3, 3)

    assert np.allclose(
        DCM.T @ DCM,
        np.eye(3),
    )

    assert np.isclose(
        np.linalg.det(DCM),
        1.0,
    )


def test_DCM_quaternion_round_trip():
    quaternion = quaternion_from_euler(
        0.3,
        -0.4,
        1.2,
    )

    DCM = quaternion_to_DCM(quaternion)

    reconstructed_quaternion = DCM_to_quaternion(DCM)

    reconstructed_DCM = quaternion_to_DCM(
        reconstructed_quaternion,
    )

    assert np.allclose(
        reconstructed_DCM,
        DCM,
    )


def test_quaternion_multiply_identity():
    quaternion = quaternion_from_euler(
        0.3,
        -0.4,
        1.2,
    )

    identity = np.array([
        0.0,
        0.0,
        0.0,
        1.0,
    ])

    result_1 = quaternion_multiply(
        quaternion,
        identity,
    )

    result_2 = quaternion_multiply(
        identity,
        quaternion,
    )

    assert np.allclose(
        result_1,
        quaternion,
    )

    assert np.allclose(
        result_2,
        quaternion,
    )


def test_quaternion_multiply_composes_rotations():
    quaternion_x = quaternion_from_euler(
        np.pi / 2,
        0.0,
        0.0,
    )

    quaternion_y = quaternion_from_euler(
        0.0,
        np.pi / 2,
        0.0,
    )

    result = quaternion_multiply(
        quaternion_x,
        quaternion_y,
    )

    result_DCM = quaternion_to_DCM(result)

    expected_DCM = (
        quaternion_to_DCM(quaternion_x)
        @ quaternion_to_DCM(quaternion_y)
    )

    assert np.allclose(
        result_DCM,
        expected_DCM,
    )


def test_quaternion_error_when_attitudes_are_equal():
    quaternion = quaternion_from_euler(
        0.3,
        -0.4,
        1.2,
    )

    result = quaternion_error(
        quaternion,
        quaternion,
    )

    expected = np.array([
        0.0,
        0.0,
        0.0,
        1.0,
    ])

    assert np.allclose(result, expected)


def test_quaternion_error_identity_target():
    target = np.array([
        0.0,
        0.0,
        0.0,
        1.0,
    ])

    actual = quaternion_from_euler(
        0.3,
        -0.4,
        1.2,
    )

    result = quaternion_error(
        target,
        actual,
    )

    assert np.allclose(
        result,
        actual,
    )


def test_quaternion_error_represents_relative_rotation():
    target = quaternion_from_euler(
        0.2,
        -0.3,
        0.4,
    )

    actual = quaternion_from_euler(
        -0.5,
        0.6,
        -0.7,
    )

    result = quaternion_error(
        target,
        actual,
    )

    target_DCM = quaternion_to_DCM(target)
    actual_DCM = quaternion_to_DCM(actual)
    result_DCM = quaternion_to_DCM(result)

    expected_DCM = (
        target_DCM.T
        @ actual_DCM
    )

    assert np.allclose(
        result_DCM,
        expected_DCM,
    )