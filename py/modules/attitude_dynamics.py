import numpy as np

from py.general.data_classes_declaration import SimSharedData

def rotational_dynamics_quaternion(
    state: np.ndarray,
    shared_data: SimSharedData,
    inertia_matrix: np.ndarray,
    inverse_inertia_matrix: np.ndarray,
    disturbance_torque: np.ndarray = np.zeros(3),
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute spacecraft rotational dynamics using Euler's equations and
    quaternion kinematics.

    Parameters
    ----------
    state : np.ndarray
        State vector.

            state = [qx, qy, qz, qw, wx, wy, wz]

    shared_data : SimSharedData
        Shared simulation data containing the control torque.

    inertia_matrix : np.ndarray
        Spacecraft inertia matrix.

    inverse_inertia_matrix : np.ndarray
        Inverse of the spacecraft inertia matrix.

    disturbance_torque : np.ndarray, optional
        External disturbance torque expressed in the body frame.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Quaternion derivative and angular velocity derivative.
    """

    # ==========================================================
    # State decomposition
    # ==========================================================

    quaternion = state[:4]
    angular_velocity = state[4:7]

    quaternion /= np.linalg.norm(quaternion)

    # ==========================================================
    # Rotational dynamics (Euler's equations)
    # ==========================================================

    angular_acceleration = (
        inverse_inertia_matrix
        @ (
            shared_data.tau_to_apply
            - np.cross(
                angular_velocity,
                inertia_matrix @ angular_velocity,
            )
            + disturbance_torque
        )
    )

    # ==========================================================
    # Quaternion kinematics
    # ==========================================================

    qx, qy, qz, qw = quaternion
    wx, wy, wz = angular_velocity

    quaternion_derivative = 0.5 * np.array([
        qw * wx + qy * wz - qz * wy,
        qw * wy + qz * wx - qx * wz,
        qw * wz + qx * wy - qy * wx,
       -qx * wx - qy * wy - qz * wz,
    ])

    return quaternion_derivative, angular_acceleration