import numpy as np
import pytest

from argos.controllers.basic_laws import (
    CustomController,
    CustomControlAllocator,
    BasicRCSAllocator,
    MultiRCSAllocator,
    ControlAllocatorPlaceholder,
    ControllerPlaceholder,
)
from argos.general.dataclasses import ControlOutput


# ============================================================
# Helpers
# ============================================================


class MockActuator:
    def __init__(
        self,
        name,
        direction,
        nominal_thrust=10.0,
        override_torque_value=5.0,
    ):
        self.name = name
        self.direction = np.asarray(direction, dtype=float)
        self.nominal_thrust = nominal_thrust
        self.override_torque_value = override_torque_value

        self.commands = []

    def set_command(self, command):
        self.commands.append(command)

    @property
    def last_command(self):
        return self.commands[-1]


# ============================================================
# CustomController
# ============================================================


class TestCustomController:

    def test_initialization_with_callable(self):
        def control_function(navigation, guidance):
            return ControlOutput(
                force=np.zeros(3),
                torque=np.zeros(3),
            )

        controller = CustomController(control_function)

        assert controller.control_function is control_function

    def test_non_callable_is_rejected(self):
        with pytest.raises(ValueError):
            CustomController("not callable")

    def test_compute_control_calls_user_function(self):
        received = {}

        def control_function(navigation, guidance):
            received["navigation"] = navigation
            received["guidance"] = guidance

            return ControlOutput(
                force=np.array([1.0, 2.0, 3.0]),
                torque=np.array([4.0, 5.0, 6.0]),
            )

        controller = CustomController(control_function)

        navigation = object()
        guidance = object()

        output = controller.compute_control(
            navigation,
            guidance,
        )

        assert received["navigation"] is navigation
        assert received["guidance"] is guidance

        np.testing.assert_allclose(
            output.force,
            np.array([1.0, 2.0, 3.0]),
        )

        np.testing.assert_allclose(
            output.torque,
            np.array([4.0, 5.0, 6.0]),
        )

    def test_compute_control_rejects_invalid_output(self):
        def control_function(navigation, guidance):
            return np.zeros(3)

        controller = CustomController(control_function)

        with pytest.raises(TypeError):
            controller.compute_control(
                object(),
                object(),
            )


# ============================================================
# CustomControlAllocator
# ============================================================


class TestCustomControlAllocator:

    def test_initialization_with_callable(self):
        def allocation_function(control_output, actuators):
            pass

        allocator = CustomControlAllocator(allocation_function)

        assert allocator.allocation_function is allocation_function

    def test_non_callable_is_rejected(self):
        with pytest.raises(ValueError):
            CustomControlAllocator("not callable")

    def test_allocate_calls_user_function(self):
        received = {}

        def allocation_function(control_output, actuators):
            received["control_output"] = control_output
            received["actuators"] = actuators

        allocator = CustomControlAllocator(
            allocation_function
        )

        actuator = MockActuator(
            name="test",
            direction=[1, 0, 0],
        )

        allocator.actuators = [actuator]

        control_output = ControlOutput(
            force=np.array([1.0, 2.0, 3.0]),
            torque=np.array([4.0, 5.0, 6.0]),
        )

        allocator.allocate(control_output)

        assert received["control_output"] is control_output
        assert received["actuators"] is allocator.actuators


# ============================================================
# BasicRCSAllocator
# ============================================================


class TestBasicRCSAllocator:

    def test_zero_torque_commands_zero_to_all_actuators(self):
        allocator = BasicRCSAllocator()

        actuators = [
            MockActuator("a", [1, 0, 0]),
            MockActuator("b", [0, 1, 0]),
            MockActuator("c", [0, 0, 1]),
        ]

        allocator.actuators = actuators

        control_output = ControlOutput(
            force=np.zeros(3),
            torque=np.zeros(3),
        )

        allocator.allocate(control_output)

        for actuator in actuators:
            assert actuator.last_command == 0

    def test_only_positive_contributing_actuator_is_commanded(self):
        allocator = BasicRCSAllocator()

        positive_x = MockActuator(
            "positive_x",
            [1, 0, 0],
        )

        negative_x = MockActuator(
            "negative_x",
            [-1, 0, 0],
        )

        allocator.actuators = [
            positive_x,
            negative_x,
        ]

        control_output = ControlOutput(
            force=np.zeros(3),
            torque=np.array([4.0, 0.0, 0.0]),
        )

        allocator.allocate(control_output)

        assert positive_x.last_command == pytest.approx(4.0)
        assert negative_x.last_command == 0

    def test_negative_torque_selects_negative_direction(self):
        allocator = BasicRCSAllocator()

        positive_x = MockActuator(
            "positive_x",
            [1, 0, 0],
        )

        negative_x = MockActuator(
            "negative_x",
            [-1, 0, 0],
        )

        allocator.actuators = [
            positive_x,
            negative_x,
        ]

        control_output = ControlOutput(
            force=np.zeros(3),
            torque=np.array([-4.0, 0.0, 0.0]),
        )

        allocator.allocate(control_output)

        assert positive_x.last_command == 0
        assert negative_x.last_command == pytest.approx(4.0)

    def test_partial_direction_allocation(self):
        allocator = BasicRCSAllocator()

        actuator = MockActuator(
            "diagonal",
            [1 / np.sqrt(2), 1 / np.sqrt(2), 0],
        )

        allocator.actuators = [actuator]

        control_output = ControlOutput(
            force=np.zeros(3),
            torque=np.array([4.0, 4.0, 0.0]),
        )

        allocator.allocate(control_output)

        # Requested magnitude = sqrt(4² + 4²)
        # Direction projection = 1
        # Therefore allocated torque = sqrt(32)
        expected = np.linalg.norm([4.0, 4.0])

        assert actuator.last_command == pytest.approx(expected)

    def test_actuator_perpendicular_to_requested_torque_is_zero(self):
        allocator = BasicRCSAllocator()

        actuator = MockActuator(
            "y",
            [0, 1, 0],
        )

        allocator.actuators = [actuator]

        control_output = ControlOutput(
            force=np.zeros(3),
            torque=np.array([5.0, 0.0, 0.0]),
        )

        allocator.allocate(control_output)

        assert actuator.last_command == 0


# ============================================================
# MultiRCSAllocator
# ============================================================


class TestMultiRCSAllocator:

    def test_zero_commands_are_not_reallocated(self):
        allocator = MultiRCSAllocator()

        actuator = MockActuator(
            "translational_thruster_x",
            [1, 0, 0],
        )

        allocator.actuators = [actuator]

        control_output = ControlOutput(
            force=np.zeros(3),
            torque=np.zeros(3),
        )

        allocator.allocate(control_output)

        # allocate() first resets every actuator.
        assert actuator.commands == [0.0]

    def test_actuators_are_reset_before_allocation(self):
        allocator = MultiRCSAllocator()

        actuator = MockActuator(
            "translational_thruster_x",
            [1, 0, 0],
        )

        allocator.actuators = [actuator]

        control_output = ControlOutput(
            force=np.array([3.0, 0.0, 0.0]),
            torque=np.zeros(3),
        )

        allocator.allocate(control_output)

        assert actuator.commands[0] == 0.0
        assert actuator.commands[-1] == pytest.approx(3.0)

    def test_smallest_sufficient_translational_actuator_is_selected(self):
        allocator = MultiRCSAllocator()

        small = MockActuator(
            "translational_thruster_x_small",
            [1, 0, 0],
            nominal_thrust=5.0,
        )

        large = MockActuator(
            "translational_thruster_x_large",
            [1, 0, 0],
            nominal_thrust=10.0,
        )

        allocator.actuators = [large, small]

        control_output = ControlOutput(
            force=np.array([4.0, 0.0, 0.0]),
            torque=np.zeros(3),
        )

        allocator.allocate(control_output)

        assert small.last_command == pytest.approx(4.0)
        assert large.last_command == 0.0

    def test_largest_translational_actuator_is_used_when_none_is_sufficient(self):
        allocator = MultiRCSAllocator()

        small = MockActuator(
            "translational_thruster_x_small",
            [1, 0, 0],
            nominal_thrust=5.0,
        )

        large = MockActuator(
            "translational_thruster_x_large",
            [1, 0, 0],
            nominal_thrust=10.0,
        )

        allocator.actuators = [small, large]

        control_output = ControlOutput(
            force=np.array([20.0, 0.0, 0.0]),
            torque=np.zeros(3),
        )

        allocator.allocate(control_output)

        assert small.last_command == 0.0
        assert large.last_command == pytest.approx(20.0)

    def test_negative_force_selects_negative_direction(self):
        allocator = MultiRCSAllocator()

        positive = MockActuator(
            "translational_thruster_x_positive",
            [1, 0, 0],
        )

        negative = MockActuator(
            "translational_thruster_x_negative",
            [-1, 0, 0],
        )

        allocator.actuators = [positive, negative]

        control_output = ControlOutput(
            force=np.array([-4.0, 0.0, 0.0]),
            torque=np.zeros(3),
        )

        allocator.allocate(control_output)

        assert positive.last_command == 0.0
        assert negative.last_command == pytest.approx(4.0)

    def test_rotation_uses_override_torque_capacity(self):
        allocator = MultiRCSAllocator()

        small = MockActuator(
            "rotational_thruster_x_small",
            [1, 0, 0],
            override_torque_value=2.0,
        )

        large = MockActuator(
            "rotational_thruster_x_large",
            [1, 0, 0],
            override_torque_value=10.0,
        )

        allocator.actuators = [large, small]

        control_output = ControlOutput(
            force=np.zeros(3),
            torque=np.array([5.0, 0.0, 0.0]),
        )

        allocator.allocate(control_output)

        assert small.last_command == 0.0
        assert large.last_command == pytest.approx(5.0)

    def test_translation_and_rotation_are_allocated_independently(self):
        allocator = MultiRCSAllocator()

        translational = MockActuator(
            "translational_thruster_x",
            [1, 0, 0],
            nominal_thrust=10.0,
        )

        rotational = MockActuator(
            "rotational_thruster_x",
            [1, 0, 0],
            override_torque_value=10.0,
        )

        allocator.actuators = [
            translational,
            rotational,
        ]

        control_output = ControlOutput(
            force=np.array([4.0, 0.0, 0.0]),
            torque=np.array([3.0, 0.0, 0.0]),
        )

        allocator.allocate(control_output)

        assert translational.last_command == pytest.approx(4.0)
        assert rotational.last_command == pytest.approx(3.0)

    def test_no_capable_actuator_receives_only_reset_command(self):
        allocator = MultiRCSAllocator()

        actuator = MockActuator(
            "translational_thruster_x",
            [-1, 0, 0],
        )

        allocator.actuators = [actuator]

        control_output = ControlOutput(
            force=np.array([5.0, 0.0, 0.0]),
            torque=np.zeros(3),
        )

        allocator.allocate(control_output)

        assert actuator.commands == [0.0]

    def test_axes_are_allocated_independently(self):
        allocator = MultiRCSAllocator()

        x = MockActuator(
            "translational_thruster_x",
            [1, 0, 0],
        )

        y = MockActuator(
            "translational_thruster_y",
            [0, 1, 0],
        )

        z = MockActuator(
            "translational_thruster_z",
            [0, 0, 1],
        )

        allocator.actuators = [x, y, z]

        control_output = ControlOutput(
            force=np.array([2.0, 3.0, 4.0]),
            torque=np.zeros(3),
        )

        allocator.allocate(control_output)

        assert x.last_command == pytest.approx(2.0)
        assert y.last_command == pytest.approx(3.0)
        assert z.last_command == pytest.approx(4.0)


# ============================================================
# Placeholders
# ============================================================


class TestControlAllocatorPlaceholder:

    def test_allocate_raises_not_implemented(self):
        allocator = ControlAllocatorPlaceholder()

        control_output = ControlOutput(
            force=np.zeros(3),
            torque=np.zeros(3),
        )

        with pytest.raises(NotImplementedError):
            allocator.allocate(control_output)


class TestControllerPlaceholder:

    def test_compute_control_raises_not_implemented(self):
        controller = ControllerPlaceholder()

        with pytest.raises(NotImplementedError):
            controller.compute_control(
                object(),
                object(),
            )