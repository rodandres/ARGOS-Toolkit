import numpy as np
import pytest

from argos.actuators.RCS import RCSThruster


# ============================================================================
# Initialization
# ============================================================================

class TestRCSThrusterInitialization:

    def test_default_initialization(self):
        thruster = RCSThruster(nominal_thrust=10.0)

        assert thruster.nominal_thrust == 10.0
        assert np.allclose(thruster.position, np.zeros(3))
        assert np.allclose(thruster.direction, [1.0, 0.0, 0.0])
        assert thruster.command_mode == "PWM"
        assert thruster.modulation_window == 0.5
        assert thruster.minimum_on_time == 0.01
        assert thruster.activation_threshold == 0.0

    def test_custom_initialization(self):
        position = np.array([1.0, 2.0, 3.0])
        direction = np.array([1.0, 2.0, 2.0])

        thruster = RCSThruster(
            nominal_thrust=20.0,
            position=position,
            direction=direction,
            command_mode="bangbang",
            modulation_window=2.0,
            minimum_on_time=0.1,
            activation_threshold=5.0,
            name="RCS_1",
        )

        assert thruster.nominal_thrust == 20.0
        assert np.allclose(thruster.position, position)
        assert np.isclose(np.linalg.norm(thruster.direction), 1.0)
        assert thruster.command_mode == "bangbang"
        assert thruster.modulation_window == 2.0
        assert thruster.minimum_on_time == 0.1
        assert thruster.activation_threshold == 5.0
        assert thruster.name == "RCS_1"

    def test_direction_is_normalized(self):
        thruster = RCSThruster(
            nominal_thrust=10.0,
            direction=np.array([3.0, 4.0, 0.0]),
        )

        assert np.allclose(thruster.direction, [0.6, 0.8, 0.0])

    def test_default_direction_is_positive_x(self):
        thruster = RCSThruster(nominal_thrust=10.0)

        assert np.allclose(thruster.direction, [1.0, 0.0, 0.0])

    def test_default_position_is_origin(self):
        thruster = RCSThruster(nominal_thrust=10.0)

        assert np.allclose(thruster.position, np.zeros(3))

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"nominal_thrust": -1.0},
            {"nominal_thrust": 10.0, "modulation_window": 0.0},
            {"nominal_thrust": 10.0, "modulation_window": -1.0},
            {"nominal_thrust": 10.0, "minimum_on_time": -0.1},
            {"nominal_thrust": 10.0, "activation_threshold": -1.0},
            {"nominal_thrust": 10.0, "command_mode": "invalid"},
            {
                "nominal_thrust": 10.0,
                "modulation_window": 1.0,
                "minimum_on_time": 2.0,
            },
        ],
    )
    def test_invalid_initialization(self, kwargs):
        with pytest.raises(ValueError):
            RCSThruster(**kwargs)

    def test_zero_direction_is_invalid(self):
        with pytest.raises(ValueError):
            RCSThruster(
                nominal_thrust=10.0,
                direction=np.zeros(3),
            )


# ============================================================================
# Commands
# ============================================================================

class TestRCSThrusterCommands:

    def test_command_must_be_scalar(self):
        thruster = RCSThruster(nominal_thrust=10.0)

        with pytest.raises(ValueError):
            thruster.set_command(np.array([5.0, 0.0, 0.0]))

    def test_negative_command_is_set_to_zero(self):
        thruster = RCSThruster(nominal_thrust=10.0)

        with pytest.warns(UserWarning):
            thruster.set_command(-5.0)

        assert thruster.command_active is False
        assert thruster.on_time == 0.0

    def test_zero_command_deactivates_pwm(self):
        thruster = RCSThruster(
            nominal_thrust=10.0,
            modulation_window=1.0,
        )

        thruster.set_command(0.0)

        assert thruster.command_active is False
        assert thruster.on_time == 0.0

    def test_command_time_is_reset_when_new_command_is_set(self):
        thruster = RCSThruster(nominal_thrust=10.0)

        thruster.set_command(5.0)
        thruster.update(0.0)

        assert thruster.command_time == 0.0

        thruster.set_command(5.0)

        assert thruster.command_time is None


# ============================================================================
# PWM command computation
# ============================================================================

class TestRCSThrusterPWMCommand:

    def test_zero_command_produces_zero_on_time(self):
        thruster = RCSThruster(
            nominal_thrust=10.0,
            modulation_window=1.0,
        )

        thruster.set_command(0.0)

        assert thruster.on_time == 0.0
        assert thruster.command_active is False

    def test_half_command_produces_half_on_time(self):
        thruster = RCSThruster(
            nominal_thrust=10.0,
            modulation_window=1.0,
        )

        thruster.set_command(5.0)

        assert np.isclose(thruster.on_time, 0.5)
        assert thruster.command_active is True

    def test_full_command_produces_full_on_time(self):
        thruster = RCSThruster(
            nominal_thrust=10.0,
            modulation_window=1.0,
        )

        thruster.set_command(10.0)

        assert np.isclose(thruster.on_time, 1.0)
        assert thruster.command_active is True

    def test_command_above_nominal_thrust_is_saturated(self):
        thruster = RCSThruster(
            nominal_thrust=10.0,
            modulation_window=1.0,
        )

        thruster.set_command(20.0)

        assert np.isclose(thruster.on_time, 1.0)
        assert thruster.command_active is True

    def test_command_below_minimum_on_time_is_rejected(self):
        thruster = RCSThruster(
            nominal_thrust=10.0,
            modulation_window=1.0,
            minimum_on_time=0.2,
        )

        thruster.set_command(1.0)

        assert thruster.on_time == 0.0
        assert thruster.command_active is False

    def test_command_exactly_at_minimum_on_time_is_active(self):
        thruster = RCSThruster(
            nominal_thrust=10.0,
            modulation_window=1.0,
            minimum_on_time=0.1,
        )

        thruster.set_command(1.0)

        assert np.isclose(thruster.on_time, 0.1)
        assert thruster.command_active is True


# ============================================================================
# PWM temporal behavior
#
# These tests reproduce the actual ARGOS execution order:
#
#     set_command()
#     update(current_simulation_time)
#
# The first update establishes command_time.
# ============================================================================

class TestRCSThrusterPWMTiming:

    def test_first_update_starts_pwm_window(self):
        thruster = RCSThruster(
            nominal_thrust=10.0,
            modulation_window=1.0,
        )

        thruster.set_command(5.0)

        thruster.update(0.0)

        assert thruster.command_time == 0.0
        assert np.allclose(
            thruster.get_output().force,
            [10.0, 0.0, 0.0],
        )

    def test_pwm_remains_on_during_on_time(self):
        thruster = RCSThruster(
            nominal_thrust=10.0,
            modulation_window=1.0,
        )

        thruster.set_command(5.0)

        thruster.update(0.0)
        assert np.allclose(
            thruster.get_output().force,
            [10.0, 0.0, 0.0],
        )

        thruster.update(0.25)
        assert np.allclose(
            thruster.get_output().force,
            [10.0, 0.0, 0.0],
        )

        thruster.update(0.499999)
        assert np.allclose(
            thruster.get_output().force,
            [10.0, 0.0, 0.0],
        )

    def test_pwm_is_off_after_on_time(self):
        thruster = RCSThruster(
            nominal_thrust=10.0,
            modulation_window=1.0,
        )

        thruster.set_command(5.0)

        thruster.update(0.0)
        thruster.update(0.500001)

        assert np.allclose(
            thruster.get_output().force,
            np.zeros(3),
        )

    def test_pwm_is_off_exactly_at_on_time(self):
        thruster = RCSThruster(
            nominal_thrust=10.0,
            modulation_window=1.0,
        )

        thruster.set_command(5.0)

        thruster.update(0.0)
        thruster.update(0.5)

        assert np.allclose(
            thruster.get_output().force,
            np.zeros(3),
        )

    def test_pwm_remains_off_after_pulse(self):
        thruster = RCSThruster(
            nominal_thrust=10.0,
            modulation_window=1.0,
        )

        thruster.set_command(5.0)

        thruster.update(0.0)
        thruster.update(0.5)
        thruster.update(1.0)
        thruster.update(10.0)

        assert np.allclose(
            thruster.get_output().force,
            np.zeros(3),
        )

    def test_pwm_pulse_duration_matches_command(self):
        thruster = RCSThruster(
            nominal_thrust=20.0,
            modulation_window=2.0,
        )

        # 50% duty cycle -> 1 s ON
        thruster.set_command(10.0)

        thruster.update(5.0)

        assert np.allclose(
            thruster.get_output().force,
            [20.0, 0.0, 0.0],
        )

        thruster.update(5.999999)

        assert np.allclose(
            thruster.get_output().force,
            [20.0, 0.0, 0.0],
        )

        thruster.update(6.0)

        assert np.allclose(
            thruster.get_output().force,
            np.zeros(3),
        )


# ============================================================================
# PWM behavior when integrated with repeated simulation cycles
# ============================================================================

class TestRCSThrusterPWMSimulationFlow:

    def test_command_and_update_in_same_simulation_cycle(self):
        """
        Reproduce the actual Spacecraft flow:

            allocator -> set_command()
            compute_actuation -> update(t)
        """
        thruster = RCSThruster(
            nominal_thrust=10.0,
            modulation_window=1.0,
        )

        simulation_times = [0.0, 0.1, 0.2, 0.3, 0.4]

        for t in simulation_times:
            if t == 0.0:
                thruster.set_command(5.0)

            thruster.update(t)

            assert np.allclose(
                thruster.get_output().force,
                [10.0, 0.0, 0.0],
            )

    def test_pwm_becomes_inactive_without_new_command(self):
        thruster = RCSThruster(
            nominal_thrust=10.0,
            modulation_window=1.0,
        )

        thruster.set_command(5.0)

        thruster.update(0.0)
        assert np.linalg.norm(thruster.get_output().force) > 0.0

        thruster.update(0.6)
        assert np.allclose(
            thruster.get_output().force,
            np.zeros(3),
        )

        thruster.update(1.0)
        assert np.allclose(
            thruster.get_output().force,
            np.zeros(3),
        )

    def test_new_command_restarts_pwm_pulse(self):
        thruster = RCSThruster(
            nominal_thrust=10.0,
            modulation_window=1.0,
        )

        # First command at t = 0
        thruster.set_command(5.0)
        thruster.update(0.0)
        thruster.update(0.6)

        assert np.allclose(
            thruster.get_output().force,
            np.zeros(3),
        )

        # New command at t = 0.6
        thruster.set_command(5.0)
        thruster.update(0.6)

        assert np.allclose(
            thruster.get_output().force,
            [10.0, 0.0, 0.0],
        )

        thruster.update(1.0)

        assert np.allclose(
            thruster.get_output().force,
            [10.0, 0.0, 0.0],
        )

        thruster.update(1.1)

        assert np.allclose(
            thruster.get_output().force,
            np.zeros(3),
        )


# ============================================================================
# Force
# ============================================================================

class TestRCSThrusterForce:

    @pytest.mark.parametrize(
        "direction",
        [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
            np.array([1.0, 1.0, 0.0]),
        ],
    )
    def test_force_follows_thruster_direction(self, direction):
        thruster = RCSThruster(
            nominal_thrust=10.0,
            direction=direction,
            modulation_window=1.0,
        )

        expected_direction = direction / np.linalg.norm(direction)

        thruster.set_command(10.0)
        thruster.update(0.0)

        expected_force = 10.0 * expected_direction

        assert np.allclose(
            thruster.get_output().force,
            expected_force,
        )

    def test_force_magnitude_is_nominal_thrust(self):
        thruster = RCSThruster(
            nominal_thrust=10.0,
            direction=np.array([1.0, 2.0, 3.0]),
        )

        thruster.set_command(5.0)
        thruster.update(0.0)

        assert np.isclose(
            np.linalg.norm(thruster.get_output().force),
            10.0,
        )

    def test_force_is_zero_when_thruster_is_inactive(self):
        thruster = RCSThruster(nominal_thrust=10.0)

        thruster.update(0.0)

        assert np.allclose(
            thruster.get_output().force,
            np.zeros(3),
        )


# ============================================================================
# Torque
# ============================================================================

class TestRCSThrusterTorque:

    def test_torque_is_cross_product_of_position_and_force(self):
        position = np.array([0.0, 1.0, 0.0])

        thruster = RCSThruster(
            nominal_thrust=10.0,
            position=position,
            direction=np.array([1.0, 0.0, 0.0]),
            modulation_window=1.0,
        )

        thruster.set_command(10.0)
        thruster.update(0.0)

        expected_force = np.array([10.0, 0.0, 0.0])
        expected_torque = np.cross(position, expected_force)

        assert np.allclose(
            thruster.get_output().torque,
            expected_torque,
        )

    def test_torque_is_zero_when_position_is_origin(self):
        thruster = RCSThruster(
            nominal_thrust=10.0,
            position=np.zeros(3),
        )

        thruster.set_command(10.0)
        thruster.update(0.0)

        assert np.allclose(
            thruster.get_output().torque,
            np.zeros(3),
        )

    def test_torque_is_zero_when_thruster_is_off(self):
        position = np.array([1.0, 2.0, 3.0])

        thruster = RCSThruster(
            nominal_thrust=10.0,
            position=position,
            modulation_window=1.0,
        )

        thruster.set_command(5.0)

        thruster.update(0.0)
        thruster.update(0.5)

        assert np.allclose(
            thruster.get_output().torque,
            np.zeros(3),
        )


# ============================================================================
# Torque override
# ============================================================================

class TestRCSThrusterTorqueOverride:

    def test_override_torque_is_used_when_enabled(self):
        thruster = RCSThruster(
            nominal_thrust=10.0,
            position=np.array([0.0, 1.0, 0.0]),
            direction=np.array([0.0, 0.0, 1.0]),
            override_torque_value=5.0,
        )

        thruster.set_command(10.0)
        thruster.update(0.0)

        output = thruster.get_output()

        expected_force = np.array([0.0, 0.0, 10.0])
        expected_torque = np.array([0.0, 0.0, 5.0])

        np.testing.assert_allclose(output.force, expected_force)
        np.testing.assert_allclose(output.torque, expected_torque)
        
    def test_override_torque_does_not_change_force(self):
        thruster = RCSThruster(
            nominal_thrust=10.0,
            position=np.array([0.0, 1.0, 0.0]),
            direction=np.array([0.0, 0.0, 1.0]),
            override_torque_value=5.0,
        )

        # 50 % duty cycle
        thruster.set_command(5.0)
        thruster.update(0.0)

        output = thruster.get_output()

        expected_force = np.array([0.0, 0.0, 10.0])

        np.testing.assert_allclose(output.force, expected_force)

    def test_override_torque_is_zero_when_thruster_is_off(self):
        thruster = RCSThruster(
            nominal_thrust=10.0,
            position=np.array([0.0, 1.0, 0.0]),
            direction=np.array([0.0, 0.0, 1.0]),
            override_torque_value=5.0,
        )

        thruster.set_command(5.0)

        # First update establishes command_time = 0.0.
        thruster.update(0.0)

        # modulation_window = 0.5 by default,
        # so the pulse is already off at t = 0.5.
        thruster.update(0.5)

        output = thruster.get_output()

        np.testing.assert_allclose(output.force, np.zeros(3))
        np.testing.assert_allclose(output.torque, np.zeros(3))


# ============================================================================
# Bang-bang mode
# ============================================================================

class TestRCSThrusterBangBang:

    def test_bangbang_below_threshold_is_inactive(self):
        thruster = RCSThruster(
            nominal_thrust=10.0,
            command_mode="bangbang",
            activation_threshold=5.0,
        )

        thruster.set_command(4.999)

        assert thruster.command_active is False

        thruster.update(0.0)

        assert np.allclose(
            thruster.get_output().force,
            np.zeros(3),
        )

    def test_bangbang_at_threshold_is_active(self):
        thruster = RCSThruster(
            nominal_thrust=10.0,
            command_mode="bangbang",
            activation_threshold=5.0,
        )

        thruster.set_command(5.0)

        assert thruster.command_active is True

        thruster.update(0.0)

        assert np.allclose(
            thruster.get_output().force,
            [10.0, 0.0, 0.0],
        )

    def test_bangbang_above_threshold_is_active(self):
        thruster = RCSThruster(
            nominal_thrust=10.0,
            command_mode="bangbang",
            activation_threshold=5.0,
        )

        thruster.set_command(8.0)
        thruster.update(0.0)

        assert np.allclose(
            thruster.get_output().force,
            [10.0, 0.0, 0.0],
        )

    def test_bangbang_force_is_independent_of_command_magnitude(self):
        thruster = RCSThruster(
            nominal_thrust=10.0,
            command_mode="bangbang",
            activation_threshold=1.0,
        )

        thruster.set_command(2.0)
        thruster.update(0.0)

        force_1 = thruster.get_output().force.copy()

        thruster.set_command(100.0)
        thruster.update(0.1)

        force_2 = thruster.get_output().force.copy()

        assert np.allclose(force_1, force_2)

    def test_bangbang_new_command_can_deactivate_thruster(self):
        thruster = RCSThruster(
            nominal_thrust=10.0,
            command_mode="bangbang",
            activation_threshold=5.0,
        )

        thruster.set_command(10.0)
        thruster.update(0.0)

        assert np.allclose(
            thruster.get_output().force,
            [10.0, 0.0, 0.0],
        )

        thruster.set_command(1.0)
        thruster.update(0.1)

        assert np.allclose(
            thruster.get_output().force,
            np.zeros(3),
        )


# ============================================================================
# Update behavior
# ============================================================================

class TestRCSThrusterUpdate:

    def test_update_without_command_produces_zero_output(self):
        thruster = RCSThruster(nominal_thrust=10.0)

        thruster.update(0.0)

        output = thruster.get_output()

        assert np.allclose(output.force, np.zeros(3))
        assert np.allclose(output.torque, np.zeros(3))

    def test_update_clears_output_when_command_is_inactive(self):
        thruster = RCSThruster(nominal_thrust=10.0)

        thruster.set_command(0.0)
        thruster.update(0.0)

        assert np.allclose(
            thruster.get_output().force,
            np.zeros(3),
        )

        assert np.allclose(
            thruster.get_output().torque,
            np.zeros(3),
        )


# ============================================================================
# Output
# ============================================================================

class TestRCSThrusterOutput:

    def test_get_output_returns_actuator_output(self):
        thruster = RCSThruster(nominal_thrust=10.0)

        thruster.set_command(10.0)
        thruster.update(0.0)

        output = thruster.get_output()

        assert hasattr(output, "force")
        assert hasattr(output, "torque")

    def test_output_is_updated_after_each_update(self):
        thruster = RCSThruster(
            nominal_thrust=10.0,
            modulation_window=1.0,
        )

        thruster.set_command(5.0)

        thruster.update(0.0)

        assert np.allclose(
            thruster.get_output().force,
            [10.0, 0.0, 0.0],
        )

        thruster.update(0.6)

        assert np.allclose(
            thruster.get_output().force,
            np.zeros(3),
        )


# ============================================================================
# Information display
# ============================================================================

class TestRCSThrusterInformation:

    def test_print_information(self, capsys):
        thruster = RCSThruster(
            nominal_thrust=10.0,
            name="RCS_1",
        )

        thruster.print_information()

        captured = capsys.readouterr()

        assert "RCS Thruster Information:" in captured.out
        assert "Nominal Thrust: 10.0 N" in captured.out
        assert "Command Mode: PWM" in captured.out

    def test_print_information_with_override(self, capsys):
        thruster = RCSThruster(
            nominal_thrust=10.0,
            override_torque_value=np.array([1.0, 2.0, 3.0]),
        )

        thruster.print_information()

        captured = capsys.readouterr()

        assert "Override Torque Enabled: Yes" in captured.out
        assert "Override Torque Value:" in captured.out