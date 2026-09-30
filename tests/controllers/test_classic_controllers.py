import numpy as np
import pytest

from argos.controllers.classic_controllers import (
    PDController,
    PDAttitudeController,
)
from argos.general.dataclasses import ControlOutput


class MockState:
    def __init__(
        self,
        position=None,
        velocity=None,
        attitude=None,
        angular_velocity=None,
    ):
        self.position = np.array(
            position if position is not None else [0.0, 0.0, 0.0]
        )
        self.velocity = np.array(
            velocity if velocity is not None else [0.0, 0.0, 0.0]
        )
        self.attitude = np.array(
            attitude if attitude is not None else [0.0, 0.0, 0.0, 1.0]
        )
        self.angular_velocity = np.array(
            angular_velocity
            if angular_velocity is not None
            else [0.0, 0.0, 0.0]
        )


class MockNavigationOutput:
    def __init__(self, state):
        self.spacecraft_state = state


class MockGuidanceOutput:
    def __init__(self, state):
        self.state = state


# ============================================================
# PDController
# ============================================================


class TestPDControllerInitialization:

    def test_initialization(self):
        controller = PDController(
            Kp_rotational=2.0,
            Kd_rotational=0.5,
            Kp_translational=3.0,
            Kd_translational=1.0,
        )

        np.testing.assert_array_equal(
            controller.Kp_rotational,
            np.array([2.0, 2.0, 2.0]),
        )
        np.testing.assert_array_equal(
            controller.Kd_rotational,
            np.array([0.5, 0.5, 0.5]),
        )

        np.testing.assert_array_equal(
            controller.Kp_translational,
            np.array([3.0, 3.0, 3.0]),
        )

        np.testing.assert_array_equal(
            controller.Kd_translational,
            np.array([1.0, 1.0, 1.0]),
        )
        assert controller.minimum_torque is None
        assert controller.maximum_torque is None

    @pytest.mark.parametrize(
        "parameter",
        [
            "Kp_rotational",
            "Kd_rotational",
            "Kp_translational",
            "Kd_translational",
        ],
    )
    def test_gain_must_be_numeric(self, parameter):
        kwargs = {
            "Kp_rotational": 1.0,
            "Kd_rotational": 1.0,
            "Kp_translational": 1.0,
            "Kd_translational": 1.0,
        }

        kwargs[parameter] = "invalid"

        with pytest.raises(TypeError):
            PDController(**kwargs)

    def test_minimum_torque_must_be_numeric(self):
        with pytest.raises(TypeError):
            PDController(
                1.0,
                1.0,
                1.0,
                1.0,
                minimum_torque="invalid",
            )

    def test_maximum_torque_must_be_numeric(self):
        with pytest.raises(TypeError):
            PDController(
                1.0,
                1.0,
                1.0,
                1.0,
                maximum_torque="invalid",
            )

    def test_minimum_torque_cannot_exceed_maximum_torque(self):
        with pytest.raises(ValueError):
            PDController(
                1.0,
                1.0,
                1.0,
                1.0,
                minimum_torque=5.0,
                maximum_torque=2.0,
            )

    def test_zero_gains_are_valid(self):
        controller = PDController(
            0.0,
            0.0,
            0.0,
            0.0,
        )

        np.testing.assert_array_equal(
            controller.Kp_rotational,
            np.zeros(3),
        )

        np.testing.assert_array_equal(
            controller.Kd_rotational,
            np.zeros(3),
        )

        np.testing.assert_array_equal(
            controller.Kp_translational,
            np.zeros(3),
        )

        np.testing.assert_array_equal(
            controller.Kd_translational,
            np.zeros(3),
        )


class TestPDControllerControl:

    def test_zero_error_produces_zero_force_and_torque(self):
        controller = PDController(
            Kp_rotational=10.0,
            Kd_rotational=2.0,
            Kp_translational=5.0,
            Kd_translational=3.0,
        )

        state = MockState(
            position=[1.0, 2.0, 3.0],
            velocity=[0.5, -1.0, 2.0],
            attitude=[0.0, 0.0, 0.0, 1.0],
            angular_velocity=[0.0, 0.0, 0.0],
        )

        navigation = MockNavigationOutput(state)
        guidance = MockGuidanceOutput(
            MockState(
                position=[1.0, 2.0, 3.0],
                velocity=[0.5, -1.0, 2.0],
                attitude=[0.0, 0.0, 0.0, 1.0],
            )
        )

        output = controller.compute_control(navigation, guidance)

        np.testing.assert_allclose(output.force, np.zeros(3))
        np.testing.assert_allclose(output.torque, np.zeros(3))

    def test_translational_proportional_term(self):
        controller = PDController(
            Kp_rotational=0.0,
            Kd_rotational=0.0,
            Kp_translational=2.0,
            Kd_translational=0.0,
        )

        navigation = MockNavigationOutput(
            MockState(
                position=[1.0, 2.0, 3.0],
                velocity=[0.0, 0.0, 0.0],
                attitude=[0.0, 0.0, 0.0, 1.0],
            )
        )

        guidance = MockGuidanceOutput(
            MockState(
                position=[4.0, 0.0, 5.0],
                velocity=[0.0, 0.0, 0.0],
                attitude=[0.0, 0.0, 0.0, 1.0],
            )
        )

        output = controller.compute_control(navigation, guidance)

        expected_force = np.array([6.0, -4.0, 4.0])

        np.testing.assert_allclose(output.force, expected_force)
        np.testing.assert_allclose(output.torque, np.zeros(3))

    def test_translational_derivative_term(self):
        controller = PDController(
            Kp_rotational=0.0,
            Kd_rotational=0.0,
            Kp_translational=0.0,
            Kd_translational=2.0,
        )

        navigation = MockNavigationOutput(
            MockState(
                position=[0.0, 0.0, 0.0],
                velocity=[1.0, 2.0, 3.0],
                attitude=[0.0, 0.0, 0.0, 1.0],
            )
        )

        guidance = MockGuidanceOutput(
            MockState(
                position=[0.0, 0.0, 0.0],
                velocity=[4.0, 0.0, 5.0],
                attitude=[0.0, 0.0, 0.0, 1.0],
            )
        )

        output = controller.compute_control(navigation, guidance)

        expected_force = np.array([6.0, -4.0, 4.0])

        np.testing.assert_allclose(output.force, expected_force)

    def test_translational_pd_terms_are_combined(self):
        controller = PDController(
            Kp_rotational=0.0,
            Kd_rotational=0.0,
            Kp_translational=2.0,
            Kd_translational=3.0,
        )

        navigation = MockNavigationOutput(
            MockState(
                position=[1.0, 1.0, 1.0],
                velocity=[1.0, 2.0, 3.0],
                attitude=[0.0, 0.0, 0.0, 1.0],
            )
        )

        guidance = MockGuidanceOutput(
            MockState(
                position=[3.0, 4.0, 5.0],
                velocity=[2.0, 4.0, 6.0],
                attitude=[0.0, 0.0, 0.0, 1.0],
            )
        )

        output = controller.compute_control(navigation, guidance)

        position_error = np.array([2.0, 3.0, 4.0])
        velocity_error = np.array([1.0, 2.0, 3.0])

        expected_force = (
            2.0 * position_error
            + 3.0 * velocity_error
        )

        np.testing.assert_allclose(output.force, expected_force)

    def test_derivative_torque_term(self):
        controller = PDController(
            Kp_rotational=0.0,
            Kd_rotational=2.0,
            Kp_translational=0.0,
            Kd_translational=0.0,
        )

        angular_velocity = np.array([1.0, -2.0, 3.0])

        navigation = MockNavigationOutput(
            MockState(
                attitude=[0.0, 0.0, 0.0, 1.0],
                angular_velocity=angular_velocity,
            )
        )

        guidance = MockGuidanceOutput(
            MockState(
                attitude=[0.0, 0.0, 0.0, 1.0],
            )
        )

        output = controller.compute_control(navigation, guidance)

        expected_torque = -2.0 * angular_velocity

        np.testing.assert_allclose(output.torque, expected_torque)

    def test_torque_saturation(self):
        controller = PDController(
            Kp_rotational=0.0,
            Kd_rotational=10.0,
            Kp_translational=0.0,
            Kd_translational=0.0,
            minimum_torque=-2.0,
            maximum_torque=2.0,
        )

        navigation = MockNavigationOutput(
            MockState(
                attitude=[0.0, 0.0, 0.0, 1.0],
                angular_velocity=[10.0, -10.0, 1.0],
            )
        )

        guidance = MockGuidanceOutput(
            MockState(
                attitude=[0.0, 0.0, 0.0, 1.0],
            )
        )

        output = controller.compute_control(navigation, guidance)

        expected_torque = np.array([-2.0, 2.0, -2.0])

        np.testing.assert_allclose(output.torque, expected_torque)

    def test_only_torque_saturation_does_not_modify_force(self):
        controller = PDController(
            Kp_rotational=0.0,
            Kd_rotational=10.0,
            Kp_translational=2.0,
            Kd_translational=0.0,
            minimum_torque=-1.0,
            maximum_torque=1.0,
        )

        navigation = MockNavigationOutput(
            MockState(
                position=[0.0, 0.0, 0.0],
                attitude=[0.0, 0.0, 0.0, 1.0],
                angular_velocity=[10.0, 0.0, 0.0],
            )
        )

        guidance = MockGuidanceOutput(
            MockState(
                position=[3.0, 4.0, 5.0],
                attitude=[0.0, 0.0, 0.0, 1.0],
            )
        )

        output = controller.compute_control(navigation, guidance)

        np.testing.assert_allclose(
            output.force,
            np.array([6.0, 8.0, 10.0]),
        )

        np.testing.assert_allclose(
            output.torque,
            np.array([-1.0, 0.0, 0.0]),
        )

    def test_actual_quaternion_is_normalized(self):
        controller = PDController(
            Kp_rotational=0.0,
            Kd_rotational=2.0,
            Kp_translational=0.0,
            Kd_translational=0.0,
        )

        navigation = MockNavigationOutput(
            MockState(
                attitude=[0.0, 0.0, 0.0, 10.0],
                angular_velocity=[1.0, 2.0, 3.0],
            )
        )

        guidance = MockGuidanceOutput(
            MockState(
                attitude=[0.0, 0.0, 0.0, 1.0],
            )
        )

        output = controller.compute_control(navigation, guidance)

        np.testing.assert_allclose(
            output.torque,
            np.array([-2.0, -4.0, -6.0]),
        )


# ============================================================
# PDAttitudeController
# ============================================================


class TestPDAttitudeControllerInitialization:

    def test_initialization(self):
        controller = PDAttitudeController(
            Kp=2.0,
            Kd=0.5,
        )

        np.testing.assert_array_equal(
            controller.Kp,
            np.array([2.0, 2.0, 2.0]),
        )
        np.testing.assert_array_equal(
            controller.Kd,
            np.array([0.5, 0.5, 0.5]),
        )
        assert controller.minimum_torque is None
        assert controller.maximum_torque is None

    @pytest.mark.parametrize(
        "parameter",
        [
            "Kp",
            "Kd",
        ],
    )
    def test_gain_must_be_numeric(self, parameter):
        kwargs = {
            "Kp": 1.0,
            "Kd": 1.0,
        }

        kwargs[parameter] = "invalid"

        with pytest.raises(TypeError):
            PDAttitudeController(**kwargs)

    def test_minimum_torque_must_be_numeric(self):
        with pytest.raises(TypeError):
            PDAttitudeController(
                1.0,
                1.0,
                minimum_torque="invalid",
            )

    def test_maximum_torque_must_be_numeric(self):
        with pytest.raises(TypeError):
            PDAttitudeController(
                1.0,
                1.0,
                maximum_torque="invalid",
            )

    def test_minimum_torque_cannot_exceed_maximum_torque(self):
        with pytest.raises(ValueError):
            PDAttitudeController(
                1.0,
                1.0,
                minimum_torque=5.0,
                maximum_torque=2.0,
            )


class TestPDAttitudeControllerControl:

    def test_zero_attitude_error_and_zero_rate_produce_zero_torque(self):
        controller = PDAttitudeController(
            Kp=10.0,
            Kd=2.0,
        )

        navigation = MockNavigationOutput(
            MockState(
                attitude=[0.0, 0.0, 0.0, 1.0],
                angular_velocity=[0.0, 0.0, 0.0],
            )
        )

        guidance = MockGuidanceOutput(
            MockState(
                attitude=[0.0, 0.0, 0.0, 1.0],
            )
        )

        output = controller.compute_control(navigation, guidance)

        np.testing.assert_allclose(output.force, np.zeros(3))
        np.testing.assert_allclose(output.torque, np.zeros(3))

    def test_derivative_term(self):
        controller = PDAttitudeController(
            Kp=0.0,
            Kd=2.0,
        )

        angular_velocity = np.array([1.0, -2.0, 3.0])

        navigation = MockNavigationOutput(
            MockState(
                attitude=[0.0, 0.0, 0.0, 1.0],
                angular_velocity=angular_velocity,
            )
        )

        guidance = MockGuidanceOutput(
            MockState(
                attitude=[0.0, 0.0, 0.0, 1.0],
            )
        )

        output = controller.compute_control(navigation, guidance)

        np.testing.assert_allclose(
            output.torque,
            -2.0 * angular_velocity,
        )

    def test_force_is_always_zero(self):
        controller = PDAttitudeController(
            Kp=1.0,
            Kd=1.0,
        )

        navigation = MockNavigationOutput(
            MockState(
                attitude=[0.0, 0.0, 0.0, 1.0],
                angular_velocity=[1.0, 2.0, 3.0],
            )
        )

        guidance = MockGuidanceOutput(
            MockState(
                attitude=[0.0, 0.0, 0.0, 1.0],
            )
        )

        output = controller.compute_control(navigation, guidance)

        np.testing.assert_allclose(output.force, np.zeros(3))

    def test_torque_saturation(self):
        controller = PDAttitudeController(
            Kp=0.0,
            Kd=10.0,
            minimum_torque=-2.0,
            maximum_torque=2.0,
        )

        navigation = MockNavigationOutput(
            MockState(
                attitude=[0.0, 0.0, 0.0, 1.0],
                angular_velocity=[10.0, -10.0, 1.0],
            )
        )

        guidance = MockGuidanceOutput(
            MockState(
                attitude=[0.0, 0.0, 0.0, 1.0],
            )
        )

        output = controller.compute_control(navigation, guidance)

        np.testing.assert_allclose(
            output.torque,
            np.array([-2.0, 2.0, -2.0]),
        )