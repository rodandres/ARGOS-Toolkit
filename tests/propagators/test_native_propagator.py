import numpy as np
import pytest

from argos.propagators.native_propagator import (
    NativeRotationalPropagator,
    NativeTranslationalPropagator,
    SUPPORTED_DYNAMICS,
)


# ============================================================================
# NativeTranslationalPropagator - initialization
# ============================================================================


def test_translational_propagator_default_initialization():
    propagator = NativeTranslationalPropagator()

    assert propagator.integration_method == "NATIVE_RK45"
    assert propagator.dynamics_function is None
    assert propagator.arguments == ()


@pytest.mark.parametrize("dynamics", SUPPORTED_DYNAMICS)
def test_translational_propagator_accepts_supported_dynamics(dynamics):
    propagator = NativeTranslationalPropagator(dynamics)

    assert propagator.dynamics_function is not None or dynamics == "NEWTON"


def test_translational_propagator_rejects_unsupported_dynamics():
    with pytest.raises(ValueError):
        NativeTranslationalPropagator("INVALID")


# ============================================================================
# NativeTranslationalPropagator - dynamic model selection
# ============================================================================


def test_set_dynamic_model_rel2bp():
    propagator = NativeTranslationalPropagator()

    propagator.set_dynamic_model("REL2BP")

    assert propagator.dynamics_function == propagator.rel2bp
    assert len(propagator.arguments) == 1
    assert propagator.arguments[0] > 0.0


def test_set_dynamic_model_cr3bp():
    propagator = NativeTranslationalPropagator()

    propagator.set_dynamic_model("CR3BP")

    assert propagator.dynamics_function == propagator.cr3bp
    assert len(propagator.arguments) == 3

    mu, length_factor, time_factor = propagator.arguments

    assert mu > 0.0
    assert length_factor > 0.0
    assert time_factor > 0.0


def test_set_dynamic_model_newton():
    propagator = NativeTranslationalPropagator("REL2BP")

    propagator.set_dynamic_model("NEWTON")

    assert propagator.dynamics_function is None
    assert propagator.arguments == ()


# ============================================================================
# NativeTranslationalPropagator - REL2BP
# ============================================================================


def test_rel2bp_returns_velocity_and_gravitational_acceleration():
    propagator = NativeTranslationalPropagator("REL2BP")

    mu = propagator.arguments[0]

    radius = 7_000_000.0
    velocity_input = np.array([0.0, 7_500.0, 0.0])

    state = np.concatenate(
        (
            np.array([radius, 0.0, 0.0]),
            velocity_input,
        )
    )

    velocity, acceleration = propagator.rel2bp(
        0.0,
        state,
        mu,
    )

    expected_acceleration = np.array(
        [
            -mu / radius**2,
            0.0,
            0.0,
        ]
    )

    assert velocity.shape == (3,)
    assert acceleration.shape == (3,)

    assert np.allclose(velocity, velocity_input)
    assert np.allclose(acceleration, expected_acceleration)


def test_rel2bp_acceleration_points_toward_central_body():
    propagator = NativeTranslationalPropagator("REL2BP")

    mu = propagator.arguments[0]

    state = np.array(
        [
            0.0,
            8_000_000.0,
            0.0,
            -1000.0,
            0.0,
            0.0,
        ]
    )

    _, acceleration = propagator.rel2bp(
        0.0,
        state,
        mu,
    )

    assert acceleration[0] == pytest.approx(0.0)
    assert acceleration[1] < 0.0
    assert acceleration[2] == pytest.approx(0.0)


# ============================================================================
# NativeTranslationalPropagator - NEWTON dynamics
# ============================================================================


def test_newton_dynamics_without_forces():
    propagator = NativeTranslationalPropagator("NEWTON")

    state = np.array(
        [
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
        ]
    )

    result = propagator.dynamics(
        t=0.0,
        state=state,
        mass=10.0,
        applied_force=np.zeros(3),
        disturbance_force=np.zeros(3),
    )

    expected = np.array(
        [
            4.0,
            5.0,
            6.0,
            0.0,
            0.0,
            0.0,
        ]
    )

    assert np.allclose(result, expected)


def test_newton_dynamics_with_external_forces():
    propagator = NativeTranslationalPropagator("NEWTON")

    state = np.array(
        [
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
        ]
    )

    mass = 10.0
    applied_force = np.array([10.0, 20.0, 30.0])
    disturbance_force = np.array([1.0, 2.0, 3.0])

    result = propagator.dynamics(
        t=0.0,
        state=state,
        mass=mass,
        applied_force=applied_force,
        disturbance_force=disturbance_force,
    )

    expected = np.array(
        [
            4.0,
            5.0,
            6.0,
            1.1,
            2.2,
            3.3,
        ]
    )

    assert np.allclose(result, expected)


def test_dynamics_with_non_positive_mass_ignores_external_forces():
    propagator = NativeTranslationalPropagator("NEWTON")

    state = np.array(
        [
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
        ]
    )

    result = propagator.dynamics(
        t=0.0,
        state=state,
        mass=0.0,
        applied_force=np.array([10.0, 20.0, 30.0]),
        disturbance_force=np.array([1.0, 2.0, 3.0]),
    )

    expected = np.array(
        [
            4.0,
            5.0,
            6.0,
            0.0,
            0.0,
            0.0,
        ]
    )

    assert np.allclose(result, expected)


# ============================================================================
# NativeTranslationalPropagator - CR3BP
# ============================================================================


def test_cr3bp_returns_velocity_and_acceleration():
    propagator = NativeTranslationalPropagator("CR3BP")

    mu, length_factor, time_factor = propagator.arguments

    state = np.array(
        [
            0.5 * length_factor,
            0.1 * length_factor,
            0.0,
            0.0,
            1_000.0,
            0.0,
        ]
    )

    velocity, acceleration = propagator.cr3bp(
        0.0,
        state,
        mu,
        length_factor,
        time_factor,
    )

    assert velocity.shape == (3,)
    assert acceleration.shape == (3,)

    assert np.all(np.isfinite(velocity))
    assert np.all(np.isfinite(acceleration))


# ============================================================================
# NativeRotationalPropagator - initialization
# ============================================================================


def test_rotational_propagator_default_initialization():
    propagator = NativeRotationalPropagator()

    assert propagator.integration_method == "NATIVE_RK45"


def test_rotational_propagator_custom_integration_method():
    propagator = NativeRotationalPropagator("CUSTOM")

    assert propagator.integration_method == "CUSTOM"


# ============================================================================
# NativeRotationalPropagator - quaternion dynamics
# ============================================================================


def test_quaternion_dynamics_with_zero_angular_velocity():
    propagator = NativeRotationalPropagator()

    inertia = np.diag([10.0, 20.0, 30.0])
    inverse_inertia = np.linalg.inv(inertia)

    quaternion = np.array([0.0, 0.0, 0.0, 1.0])
    angular_velocity = np.zeros(3)

    state = np.concatenate((quaternion, angular_velocity))

    result = propagator.quaternion_dynamics(
        t=0.0,
        state=state,
        inertia_matrix=inertia,
        inverse_inertia_matrix=inverse_inertia,
        applied_torque=np.zeros(3),
        disturbance_torque=np.zeros(3),
    )

    assert np.allclose(result, np.zeros(7))


def test_quaternion_dynamics_with_applied_torque():
    propagator = NativeRotationalPropagator()

    inertia = np.diag([10.0, 20.0, 30.0])
    inverse_inertia = np.linalg.inv(inertia)

    state = np.array(
        [
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
        ]
    )

    applied_torque = np.array([10.0, 20.0, 30.0])

    result = propagator.quaternion_dynamics(
        t=0.0,
        state=state,
        inertia_matrix=inertia,
        inverse_inertia_matrix=inverse_inertia,
        applied_torque=applied_torque,
        disturbance_torque=np.zeros(3),
    )

    expected_angular_acceleration = np.array(
        [
            1.0,
            1.0,
            1.0,
        ]
    )

    assert np.allclose(
        result[:4],
        np.zeros(4),
    )

    assert np.allclose(
        result[4:],
        expected_angular_acceleration,
    )


def test_quaternion_dynamics_with_disturbance_torque():
    propagator = NativeRotationalPropagator()

    inertia = np.diag([10.0, 20.0, 30.0])
    inverse_inertia = np.linalg.inv(inertia)

    state = np.array(
        [
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
        ]
    )

    disturbance_torque = np.array([5.0, 10.0, 15.0])

    result = propagator.quaternion_dynamics(
        t=0.0,
        state=state,
        inertia_matrix=inertia,
        inverse_inertia_matrix=inverse_inertia,
        applied_torque=np.zeros(3),
        disturbance_torque=disturbance_torque,
    )

    expected_angular_acceleration = np.array(
        [
            0.5,
            0.5,
            0.5,
        ]
    )

    assert np.allclose(
        result[4:],
        expected_angular_acceleration,
    )


def test_quaternion_dynamics_includes_gyroscopic_term():
    propagator = NativeRotationalPropagator()

    inertia = np.diag([2.0, 3.0, 4.0])
    inverse_inertia = np.linalg.inv(inertia)

    angular_velocity = np.array([1.0, 2.0, 3.0])

    state = np.concatenate(
        (
            np.array([0.0, 0.0, 0.0, 1.0]),
            angular_velocity,
        )
    )

    result = propagator.quaternion_dynamics(
        t=0.0,
        state=state,
        inertia_matrix=inertia,
        inverse_inertia_matrix=inverse_inertia,
        applied_torque=np.zeros(3),
        disturbance_torque=np.zeros(3),
    )

    expected_angular_acceleration = (
        -inverse_inertia
        @ np.cross(
            angular_velocity,
            inertia @ angular_velocity,
        )
    )

    assert np.allclose(
        result[4:],
        expected_angular_acceleration,
    )


def test_quaternion_dynamics_normalizes_quaternion():
    propagator = NativeRotationalPropagator()

    inertia = np.eye(3)
    inverse_inertia = np.eye(3)

    quaternion = np.array([0.0, 0.0, 0.0, 2.0])
    state = np.concatenate(
        (
            quaternion,
            np.zeros(3),
        )
    )

    result = propagator.quaternion_dynamics(
        t=0.0,
        state=state,
        inertia_matrix=inertia,
        inverse_inertia_matrix=inverse_inertia,
        applied_torque=np.zeros(3),
        disturbance_torque=np.zeros(3),
    )

    assert np.allclose(
        result,
        np.zeros(7),
    )