import numpy as np

from scipy.spatial.transform import Rotation


def quaternion_multiply(
    quaternion_1: np.ndarray,
    quaternion_2: np.ndarray,
) -> np.ndarray:
    """
    Compute the multiplication of two quaternions.

    Parameters
    ----------
    quaternion_1 : np.ndarray
        First quaternion in (x, y, z, w) convention.

    quaternion_2 : np.ndarray
        Second quaternion in (x, y, z, w) convention.

    Returns
    -------
    np.ndarray
        Resulting quaternion in (x, y, z, w) convention.
    """

    rotation_1 = Rotation.from_quat(quaternion_1)
    rotation_2 = Rotation.from_quat(quaternion_2)

    result_rotation = rotation_1 * rotation_2

    return result_rotation.as_quat()


def quaternion_from_euler(
    roll: float,
    pitch: float,
    yaw: float,
) -> np.ndarray:
    """
    Convert Euler angles into quaternion representation.

    Parameters
    ----------
    roll : float
        Rotation angle about the X axis [rad].

    pitch : float
        Rotation angle about the Y axis [rad].

    yaw : float
        Rotation angle about the Z axis [rad].

    Returns
    -------
    np.ndarray
        Quaternion in (x, y, z, w) convention.
    """

    return Rotation.from_euler(
        "xyz",
        [roll, pitch, yaw],
    ).as_quat()


def euler_from_quaternion(
    quaternion: np.ndarray,
) -> np.ndarray:
    """
    Convert quaternion into Euler angles.

    Parameters
    ----------
    quaternion : np.ndarray
        Quaternion in (x, y, z, w) convention.

    Returns
    -------
    np.ndarray
        Euler angles [roll, pitch, yaw] in radians.
    """

    return Rotation.from_quat(
        quaternion,
    ).as_euler("xyz")


def quaternion_to_DCM(
    quaternion: np.ndarray,
) -> np.ndarray:
    """
    Convert quaternion into Direction Cosine Matrix associated with the quaternion.

    If the quaternion represents the orientation of frame A
    with respect to frame B, the returned DCM transforms vectors
    from frame A to frame B.

    Parameters
    ----------
    quaternion : np.ndarray
        Quaternion in (x, y, z, w) convention.

    Returns
    -------
    np.ndarray
        Rotation matrix from body frame to inertial frame.
    """

    return Rotation.from_quat(
        quaternion,
    ).as_matrix()


def quaternion_error(
    target_quaternion: np.ndarray,
    actual_quaternion: np.ndarray,
) -> np.ndarray:
    """
    Compute quaternion attitude error.

    The error quaternion represents the rotation required to move from
    the actual attitude to the target attitude.

    Parameters
    ----------
    target_quaternion : np.ndarray
        Desired quaternion in (x, y, z, w) convention.

    actual_quaternion : np.ndarray
        Current quaternion in (x, y, z, w) convention.

    Returns
    -------
    np.ndarray
        Error quaternion in (x, y, z, w) convention.
    """

    target_rotation = Rotation.from_quat(target_quaternion)
    actual_rotation = Rotation.from_quat(actual_quaternion)

    error_rotation = (
        target_rotation.inv()
        * actual_rotation
    )

    return error_rotation.as_quat()