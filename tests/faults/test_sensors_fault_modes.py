import numpy as np

from argos.faults.sensors_fault_modes import SensorStuck


class TestSensorStuck:

    def test_initial_state(self):
        fault = SensorStuck()

        assert fault.active is False
        assert fault.stuck_value is None

    def test_activate_sets_active(self):
        fault = SensorStuck()

        fault.activate()

        assert fault.active is True
        assert fault.stuck_value is None

    def test_deactivate_sets_inactive(self):
        fault = SensorStuck()

        fault.activate()
        fault.deactivate()

        assert fault.active is False
        assert fault.stuck_value is None

    def test_first_value_is_stored(self):
        fault = SensorStuck()

        value = np.array([1.0, 2.0, 3.0])

        result = fault.apply(value)

        np.testing.assert_array_equal(result, value)
        np.testing.assert_array_equal(fault.stuck_value, value)

    def test_subsequent_values_are_ignored(self):
        fault = SensorStuck()

        first_value = np.array([1.0, 2.0, 3.0])
        second_value = np.array([4.0, 5.0, 6.0])

        fault.apply(first_value)
        result = fault.apply(second_value)

        np.testing.assert_array_equal(result, first_value)

    def test_apply_returns_copy(self):
        fault = SensorStuck()

        value = np.array([1.0, 2.0, 3.0])

        result = fault.apply(value)

        assert result is not fault.stuck_value

    def test_modifying_returned_value_does_not_modify_stuck_value(self):
        fault = SensorStuck()

        value = np.array([1.0, 2.0, 3.0])

        result = fault.apply(value)
        result[0] = 100.0

        np.testing.assert_array_equal(
            fault.stuck_value,
            np.array([1.0, 2.0, 3.0]),
        )

    def test_activate_resets_stuck_value(self):
        fault = SensorStuck()

        first_value = np.array([1.0, 2.0, 3.0])
        second_value = np.array([4.0, 5.0, 6.0])

        fault.apply(first_value)

        fault.activate()

        result = fault.apply(second_value)

        np.testing.assert_array_equal(result, second_value)

    def test_deactivate_resets_stuck_value(self):
        fault = SensorStuck()

        first_value = np.array([1.0, 2.0, 3.0])
        second_value = np.array([4.0, 5.0, 6.0])

        fault.apply(first_value)
        fault.deactivate()

        result = fault.apply(second_value)

        np.testing.assert_array_equal(result, second_value)

    def test_context_is_ignored(self):
        fault = SensorStuck()

        value = np.array([1.0, 2.0, 3.0])
        context = object()

        result = fault.apply(value, context)

        np.testing.assert_array_equal(result, value)