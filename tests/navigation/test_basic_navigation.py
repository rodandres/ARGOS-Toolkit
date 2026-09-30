import numpy as np
import pytest

from argos.navigation.basic_laws import (
    IdealNavigation,
    CustomNavigation,
    NavigationPlaceholder,
)
from argos.general.dataclasses import (
    NavigationOutput,
    StateVariables,
)


def make_state(offset=0.0):
    return StateVariables(
        position=np.array([1.0, 2.0, 3.0]) + offset,
        velocity=np.array([4.0, 5.0, 6.0]) + offset,
        acceleration=np.array([7.0, 8.0, 9.0]) + offset,
        attitude=np.array([1.0, 0.0, 0.0, 0.0]),
        angular_velocity=np.array([10.0, 11.0, 12.0]) + offset,
        angular_acceleration=np.array([13.0, 14.0, 15.0]) + offset,
    )


def make_navigation_output():
    return NavigationOutput(
        spacecraft_state=make_state(),
        target_state=make_state(100.0),
    )


class MockAbsoluteSensor:
    type = "AbsoluteSensor"

    def __init__(self, spacecraft_state, target_state):
        self.spacecraft_state = spacecraft_state
        self.target_state = target_state

    def get_measurement(self):
        return (
            (
                self.spacecraft_state.position,
                self.spacecraft_state.velocity,
                self.spacecraft_state.acceleration,
                self.spacecraft_state.attitude,
                self.spacecraft_state.angular_velocity,
                self.spacecraft_state.angular_acceleration,
            ),
            (
                self.target_state.position,
                self.target_state.velocity,
                self.target_state.acceleration,
                self.target_state.attitude,
                self.target_state.angular_velocity,
                self.target_state.angular_acceleration,
            ),
        )


class MockSensor:
    def __init__(self, sensor_type):
        self.type = sensor_type


class TestIdealNavigation:

    def test_initialization_default_sensor_index(self):
        navigation = IdealNavigation()

        assert navigation.sensor_to_be_use == 1

    def test_initialization_custom_sensor_index(self):
        navigation = IdealNavigation(sensor_to_be_use=2)

        assert navigation.sensor_to_be_use == 2

    def test_initialization_accepts_any_sensor_index(self):
        navigation = IdealNavigation(sensor_to_be_use=0)

        assert navigation.sensor_to_be_use == 0

    def test_estimate_raises_when_no_absolute_sensor_exists(self):
        navigation = IdealNavigation()

        sensors = [
            MockSensor("RelativeSensor"),
            MockSensor("TemperatureSensor"),
        ]

        with pytest.raises(
            ValueError,
            match="No AbsoluteSensor found in the provided sensors",
        ):
            navigation.estimate(sensors)

    def test_estimate_raises_when_requested_absolute_sensor_does_not_exist(self):
        navigation = IdealNavigation(sensor_to_be_use=2)

        sensors = [
            MockAbsoluteSensor(
                make_state(),
                make_state(100.0),
            )
        ]

        with pytest.raises(
            ValueError,
            match="No AbsoluteSensor found in the provided sensors",
        ):
            navigation.estimate(sensors)

    def test_estimate_uses_first_absolute_sensor_by_default(self):
        spacecraft_state = make_state()
        target_state = make_state(100.0)

        navigation = IdealNavigation()

        sensors = [
            MockAbsoluteSensor(spacecraft_state, target_state),
        ]

        output = navigation.estimate(sensors)

        assert isinstance(output, NavigationOutput)

        np.testing.assert_array_equal(
            output.spacecraft_state.position,
            spacecraft_state.position,
        )
        np.testing.assert_array_equal(
            output.target_state.position,
            target_state.position,
        )

    def test_estimate_selects_second_absolute_sensor(self):
        first_spacecraft_state = make_state()
        first_target_state = make_state(100.0)

        second_spacecraft_state = make_state(200.0)
        second_target_state = make_state(300.0)

        navigation = IdealNavigation(sensor_to_be_use=2)

        sensors = [
            MockAbsoluteSensor(
                first_spacecraft_state,
                first_target_state,
            ),
            MockAbsoluteSensor(
                second_spacecraft_state,
                second_target_state,
            ),
        ]

        output = navigation.estimate(sensors)

        np.testing.assert_array_equal(
            output.spacecraft_state.position,
            second_spacecraft_state.position,
        )
        np.testing.assert_array_equal(
            output.target_state.position,
            second_target_state.position,
        )

    def test_estimate_ignores_non_absolute_sensors_when_counting(self):
        first_absolute = MockAbsoluteSensor(
            make_state(),
            make_state(100.0),
        )

        second_absolute = MockAbsoluteSensor(
            make_state(200.0),
            make_state(300.0),
        )

        navigation = IdealNavigation(sensor_to_be_use=2)

        sensors = [
            MockSensor("RelativeSensor"),
            first_absolute,
            MockSensor("TemperatureSensor"),
            second_absolute,
        ]

        output = navigation.estimate(sensors)

        np.testing.assert_array_equal(
            output.spacecraft_state.position,
            second_absolute.spacecraft_state.position,
        )
        np.testing.assert_array_equal(
            output.target_state.position,
            second_absolute.target_state.position,
        )

    def test_estimate_copies_all_spacecraft_state_components(self):
        spacecraft_state = make_state()
        target_state = make_state(100.0)

        navigation = IdealNavigation()

        output = navigation.estimate(
            [MockAbsoluteSensor(spacecraft_state, target_state)]
        )

        np.testing.assert_array_equal(
            output.spacecraft_state.position,
            spacecraft_state.position,
        )
        np.testing.assert_array_equal(
            output.spacecraft_state.velocity,
            spacecraft_state.velocity,
        )
        np.testing.assert_array_equal(
            output.spacecraft_state.acceleration,
            spacecraft_state.acceleration,
        )
        np.testing.assert_array_equal(
            output.spacecraft_state.attitude,
            spacecraft_state.attitude,
        )
        np.testing.assert_array_equal(
            output.spacecraft_state.angular_velocity,
            spacecraft_state.angular_velocity,
        )
        np.testing.assert_array_equal(
            output.spacecraft_state.angular_acceleration,
            spacecraft_state.angular_acceleration,
        )

    def test_estimate_copies_all_target_state_components(self):
        spacecraft_state = make_state()
        target_state = make_state(100.0)

        navigation = IdealNavigation()

        output = navigation.estimate(
            [MockAbsoluteSensor(spacecraft_state, target_state)]
        )

        np.testing.assert_array_equal(
            output.target_state.position,
            target_state.position,
        )
        np.testing.assert_array_equal(
            output.target_state.velocity,
            target_state.velocity,
        )
        np.testing.assert_array_equal(
            output.target_state.acceleration,
            target_state.acceleration,
        )
        np.testing.assert_array_equal(
            output.target_state.attitude,
            target_state.attitude,
        )
        np.testing.assert_array_equal(
            output.target_state.angular_velocity,
            target_state.angular_velocity,
        )
        np.testing.assert_array_equal(
            output.target_state.angular_acceleration,
            target_state.angular_acceleration,
        )


class TestCustomNavigation:

    def test_initialization_accepts_callable(self):
        function = lambda sensors: make_navigation_output()

        navigation = CustomNavigation(function)

        assert navigation.custom_estimation_function is function

    def test_initialization_rejects_non_callable(self):
        with pytest.raises(
            ValueError,
            match="custom_estimation_function must be callable",
        ):
            CustomNavigation(None)

    def test_estimate_calls_custom_function_with_sensors(self):
        received_sensors = []

        def custom_function(sensors):
            received_sensors.append(sensors)
            return make_navigation_output()

        navigation = CustomNavigation(custom_function)

        sensors = [MockSensor("AbsoluteSensor")]

        output = navigation.estimate(sensors)

        assert received_sensors == [sensors]
        assert isinstance(output, NavigationOutput)

    def test_estimate_returns_custom_function_output(self):
        expected_output = make_navigation_output()

        def custom_function(sensors):
            return expected_output

        navigation = CustomNavigation(custom_function)

        output = navigation.estimate([])

        assert output is expected_output

    def test_estimate_rejects_invalid_custom_function_output(self):
        def custom_function(sensors):
            return None

        navigation = CustomNavigation(custom_function)

        with pytest.raises(
            TypeError,
            match="custom_estimation_function must return an NavigationOutput object",
        ):
            navigation.estimate([])


class TestNavigationPlaceholder:

    def test_estimate_raises_not_implemented_error(self):
        navigation = NavigationPlaceholder()

        with pytest.raises(
            NotImplementedError,
            match="NavigationPlaceholder does not implement the estimate method",
        ):
            navigation.estimate([])