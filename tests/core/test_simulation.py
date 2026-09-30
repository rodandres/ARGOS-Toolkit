from unittest.mock import Mock

import numpy as np
import pytest

from argos.core.simulation import Simulation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_simulation(max_sim_time=10.0, **kwargs):
    return Simulation(
        max_sim_time=max_sim_time,
        environment=Mock(),
        **kwargs,
    )


def make_mock_spacecraft(
    name="SC1",
    t=0.0,
    tick=0,
    dt_master=1.0,
    dts=(np.nan, np.nan, np.nan, 1.0),
):
    spacecraft = Mock()

    spacecraft.name = name
    spacecraft.spacecraft_data.t = t
    spacecraft.spacecraft_data.tick = tick
    spacecraft.spacecraft_data.current_master_dt = dt_master

    spacecraft.get_dts.return_value = dts

    return spacecraft


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def test_simulation_initialization():
    environment = Mock()

    simulation = Simulation(
        max_sim_time=100.0,
        environment=environment,
    )

    assert simulation.environment is environment
    assert simulation.history_chunk_size == 4096
    assert simulation.history_data_file_path == "tmp"
    assert simulation.auto_save_csv is True
    assert simulation.csv_folder_path == "sim_data"
    assert simulation.verbose is False

    assert simulation.simulation_data.max_sim_time == 100.0
    assert simulation.simulation_data.spacecrafts == []


def test_simulation_custom_configuration():
    environment = Mock()

    simulation = Simulation(
        max_sim_time=50.0,
        environment=environment,
        history_chunk_size=128,
        history_data_file_path="history",
        auto_save_csv=False,
        csv_folder_path="results",
        verbose=True,
    )

    assert simulation.environment is environment
    assert simulation.history_chunk_size == 128
    assert simulation.history_data_file_path == "history"
    assert simulation.auto_save_csv is False
    assert simulation.csv_folder_path == "results"
    assert simulation.verbose is True


def test_simulation_uses_default_csv_folder():
    simulation = make_simulation()

    assert simulation.csv_folder_path == "sim_data"


# ---------------------------------------------------------------------------
# add_spacecraft
# ---------------------------------------------------------------------------

def test_add_spacecraft_with_explicit_values():
    simulation = make_simulation()

    initial_position = np.array([1.0, 2.0, 3.0])
    initial_velocity = np.array([4.0, 5.0, 6.0])
    initial_attitude = np.array([1.0, 0.0, 0.0, 0.0])
    initial_angular_velocity = np.array([0.1, 0.2, 0.3])
    inertia_tensor = np.eye(3)

    simulation.add_spacecraft(
        name="SC1",
        mass=10.0,
        initial_position=initial_position,
        initial_velocity=initial_velocity,
        initial_attitude=initial_attitude,
        initial_angular_velocity=initial_angular_velocity,
        inertia_tensor=inertia_tensor,
    )

    assert len(simulation.simulation_data.spacecrafts) == 1

    spacecraft = simulation.simulation_data.spacecrafts[0]

    assert spacecraft.name == "SC1"
    assert spacecraft.mass == 10.0

    np.testing.assert_array_equal(
        spacecraft.spacecraft_data.true_state.position,
        initial_position,
    )

    np.testing.assert_array_equal(
        spacecraft.spacecraft_data.true_state.velocity,
        initial_velocity,
    )

    np.testing.assert_array_equal(
        spacecraft.spacecraft_data.true_state.attitude,
        initial_attitude,
    )

    np.testing.assert_array_equal(
        spacecraft.spacecraft_data.true_state.angular_velocity,
        initial_angular_velocity,
    )

    np.testing.assert_array_equal(
        spacecraft.inertia_tensor,
        inertia_tensor,
    )


def test_add_spacecraft_uses_default_values():
    simulation = make_simulation()

    simulation.add_spacecraft(name="SC1")

    spacecraft = simulation.simulation_data.spacecrafts[0]

    assert spacecraft.mass == 0.0

    np.testing.assert_array_equal(
        spacecraft.spacecraft_data.true_state.position,
        np.zeros(3),
    )

    np.testing.assert_array_equal(
        spacecraft.spacecraft_data.true_state.velocity,
        np.zeros(3),
    )

    np.testing.assert_array_equal(
        spacecraft.spacecraft_data.true_state.attitude,
        np.array([0, 0, 0, 1]),
    )

    np.testing.assert_array_equal(
        spacecraft.spacecraft_data.true_state.angular_velocity,
        np.zeros(3),
    )

    np.testing.assert_array_equal(
        spacecraft.inertia_tensor,
        np.eye(3),
    )


def test_add_spacecraft_generates_name():
    simulation = make_simulation()

    simulation.add_spacecraft()
    simulation.add_spacecraft()

    assert simulation.simulation_data.spacecrafts[0].name == "Spacecraft_1"
    assert simulation.simulation_data.spacecrafts[1].name == "Spacecraft_2"


def test_add_spacecraft_preserves_explicit_name():
    simulation = make_simulation()

    simulation.add_spacecraft(name="Chaser")

    assert simulation.simulation_data.spacecrafts[0].name == "Chaser"


def test_add_spacecraft_registers_history():
    simulation = make_simulation()

    history = Mock()
    simulation.simulation_data.simulation_history = history

    simulation.add_spacecraft(name="SC1")

    history.add_spacecraft_history.assert_called_once_with("SC1")


def test_add_spacecraft_registers_target():
    simulation = make_simulation()

    history = Mock()
    simulation.simulation_data.simulation_history = history

    simulation.add_spacecraft(
        name="Chaser",
        target_name="Target",
    )

    spacecraft = simulation.simulation_data.spacecrafts[0]

    assert spacecraft.spacecraft_data.target_name == "Target"

    assert history.record_target.call_count == 2


# ---------------------------------------------------------------------------
# Target management
# ---------------------------------------------------------------------------

def test_add_spacecraft_target():
    simulation = make_simulation()

    simulation.add_spacecraft(name="Chaser")

    simulation.add_spacecraft_target(
        spacecraft_name="Chaser",
        target_name="Target",
    )

    spacecraft = simulation.simulation_data.spacecrafts[0]

    assert spacecraft.spacecraft_data.target_name == "Target"


def test_add_spacecraft_target_records_history():
    simulation = make_simulation()

    simulation.add_spacecraft(name="Chaser")

    history = Mock()
    simulation.simulation_data.simulation_history = history

    simulation.add_spacecraft_target(
        spacecraft_name="Chaser",
        target_name="Target",
    )

    history.record_target.assert_called_once_with(
        spacecraft_name="Chaser",
        target_name="Target",
    )


def test_add_spacecraft_target_does_nothing_if_spacecraft_not_found():
    simulation = make_simulation()

    simulation.add_spacecraft_target(
        spacecraft_name="Unknown",
        target_name="Target",
    )

    assert simulation.simulation_data.spacecrafts == []


# ---------------------------------------------------------------------------
# _set_dt_master
# ---------------------------------------------------------------------------

def test_set_dt_master_uses_minimum_dt():
    simulation = make_simulation()

    sc1 = make_mock_spacecraft(
        name="SC1",
        dts=(np.nan, 1.0, np.nan, 0.5),
    )

    sc2 = make_mock_spacecraft(
        name="SC2",
        dts=(0.2, np.nan, np.nan, 1.0),
    )

    simulation.simulation_data.spacecrafts = [sc1, sc2]

    simulation._set_dt_master()

    assert simulation.simulation_data.dt_master == 0.2


def test_set_dt_master_ignores_nan_dts():
    simulation = make_simulation()

    spacecraft = make_mock_spacecraft(
        dts=(np.nan, np.nan, np.nan, 0.5),
    )

    simulation.simulation_data.spacecrafts = [spacecraft]

    simulation._set_dt_master()

    assert simulation.simulation_data.dt_master == 0.5


def test_set_dt_master_with_single_spacecraft():
    simulation = make_simulation()

    spacecraft = make_mock_spacecraft(
        dts=(0.1, 0.2, 0.5, 1.0),
    )

    simulation.simulation_data.spacecrafts = [spacecraft]

    simulation._set_dt_master()

    assert simulation.simulation_data.dt_master == 0.1


def test_set_dt_master_includes_propagation_dt():
    simulation = make_simulation()

    spacecraft = make_mock_spacecraft(
        dts=(np.nan, np.nan, np.nan, 0.25),
    )

    simulation.simulation_data.spacecrafts = [spacecraft]

    simulation._set_dt_master()

    assert simulation.simulation_data.dt_master == 0.25


# ---------------------------------------------------------------------------
# record_transition
# ---------------------------------------------------------------------------

def test_record_transition():
    simulation = make_simulation()

    event_info = Mock()

    simulation.simulation_data.simulation_history.record_event = Mock()

    simulation.record_transition(
        spacecraft_name="SC1",
        event_info=event_info,
    )

    simulation.simulation_data.simulation_history.record_event.assert_called_once_with(
        "SC1",
        event_info,
    )


# ---------------------------------------------------------------------------
# _init_simulation
# ---------------------------------------------------------------------------

def test_init_simulation_sets_master_dt():
    simulation = make_simulation()

    spacecraft = make_mock_spacecraft(
        dts=(0.1, np.nan, np.nan, 1.0),
    )

    simulation.simulation_data.spacecrafts = [spacecraft]
    simulation.simulation_data.simulation_history.record_spacecraft = Mock()

    simulation._init_simulation()

    assert simulation.simulation_data.dt_master == 0.1

def test_init_simulation_records_initial_spacecraft_state():
    simulation = make_simulation()

    spacecraft = make_mock_spacecraft(
        dts=(0.1, np.nan, np.nan, 1.0),
    )

    simulation.simulation_data.spacecrafts = [spacecraft]

    history = Mock()
    simulation.simulation_data.simulation_history = history

    simulation._init_simulation()

    history.record_spacecraft.assert_called_once_with(
        spacecraft.name,
        spacecraft.spacecraft_data,
    )


def test_init_simulation_records_all_spacecrafts():
    simulation = make_simulation()

    sc1 = make_mock_spacecraft(
        name="SC1",
        dts=(0.1, np.nan, np.nan, 1.0),
    )

    sc2 = make_mock_spacecraft(
        name="SC2",
        dts=(0.2, np.nan, np.nan, 1.0),
    )

    simulation.simulation_data.spacecrafts = [sc1, sc2]

    history = Mock()
    simulation.simulation_data.simulation_history = history

    simulation._init_simulation()

    assert history.record_spacecraft.call_count == 2

    history.record_spacecraft.assert_any_call(
        "SC1",
        sc1.spacecraft_data,
    )

    history.record_spacecraft.assert_any_call(
        "SC2",
        sc2.spacecraft_data,
    )


# ---------------------------------------------------------------------------
# simulate
# ---------------------------------------------------------------------------

def test_simulate_requires_at_least_one_spacecraft():
    simulation = make_simulation()

    with pytest.raises(ValueError, match="spacecraft"):
        simulation.simulate()


def test_simulate_updates_spacecraft():
    simulation = make_simulation(max_sim_time=1.0)

    spacecraft = make_mock_spacecraft()

    simulation.simulation_data.spacecrafts = [spacecraft]
    simulation.simulation_data.simulation_history = Mock()

    simulation.simulate()

    spacecraft.update_mission_manager.assert_called()
    spacecraft.update_sensors.assert_called()
    spacecraft.update_navigation.assert_called()
    spacecraft.update_guidance.assert_called()
    spacecraft.update_control.assert_called()
    spacecraft.compute_actuation.assert_called()
    spacecraft.propagate_translational.assert_called()
    spacecraft.propagate_rotational.assert_called()


def test_simulate_calls_spacecraft_operations_in_order():
    simulation = make_simulation(max_sim_time=1.0)

    spacecraft = make_mock_spacecraft()

    simulation.simulation_data.spacecrafts = [spacecraft]
    simulation.simulation_data.simulation_history = Mock()

    simulation.simulate()

    expected = [
        "update_mission_manager",
        "update_sensors",
        "update_navigation",
        "update_guidance",
        "update_control",
        "compute_actuation",
        "propagate_translational",
        "propagate_rotational",
    ]

    actual = [
        call[0]
        for call in spacecraft.method_calls
        if call[0] in expected
    ]

    assert actual == expected


def test_simulate_advances_spacecraft_time_and_tick():
    simulation = make_simulation(max_sim_time=2.0)

    spacecraft = make_mock_spacecraft(
        t=0.0,
        tick=0,
        dt_master=1.0,
    )

    simulation.simulation_data.spacecrafts = [spacecraft]
    simulation.simulation_data.simulation_history = Mock()

    simulation.simulate()

    assert spacecraft.spacecraft_data.t == 2.0
    assert spacecraft.spacecraft_data.tick == 2


def test_simulate_records_spacecraft_state_each_step():
    simulation = make_simulation(max_sim_time=2.0)

    spacecraft = make_mock_spacecraft()

    simulation.simulation_data.spacecrafts = [spacecraft]

    history = Mock()
    simulation.simulation_data.simulation_history = history

    simulation.simulate()

    # One initial record plus one record per simulation step.
    assert history.record_spacecraft.call_count == 3


def test_simulate_finalizes_history_and_returns_metadata():
    simulation = make_simulation(max_sim_time=1.0)

    spacecraft = make_mock_spacecraft()

    simulation.simulation_data.spacecrafts = [spacecraft]

    history = Mock()
    history.finalize.return_value = {"test": "metadata"}

    simulation.simulation_data.simulation_history = history

    metadata = simulation.simulate()

    history.finalize.assert_called_once()

    assert metadata == {"test": "metadata"}


# ---------------------------------------------------------------------------
# Simulation state
# ---------------------------------------------------------------------------

def test_simulate_initializes_simulation_time_and_tick():
    simulation = make_simulation(max_sim_time=1.0)

    spacecraft = make_mock_spacecraft()

    simulation.simulation_data.spacecrafts = [spacecraft]
    simulation.simulation_data.simulation_history = Mock()

    simulation.simulate()

    assert np.isnan(simulation.simulation_data.t)
    assert np.isnan(simulation.simulation_data.tick)


# ---------------------------------------------------------------------------
# Multiple spacecraft
# ---------------------------------------------------------------------------

def test_simulate_updates_all_spacecrafts():
    simulation = make_simulation(max_sim_time=1.0)

    sc1 = make_mock_spacecraft(name="SC1")
    sc2 = make_mock_spacecraft(name="SC2")

    simulation.simulation_data.spacecrafts = [sc1, sc2]
    simulation.simulation_data.simulation_history = Mock()

    simulation.simulate()

    for spacecraft in [sc1, sc2]:
        spacecraft.update_mission_manager.assert_called()
        spacecraft.update_sensors.assert_called()
        spacecraft.update_navigation.assert_called()
        spacecraft.update_guidance.assert_called()
        spacecraft.update_control.assert_called()
        spacecraft.compute_actuation.assert_called()
        spacecraft.propagate_translational.assert_called()
        spacecraft.propagate_rotational.assert_called()


def test_simulate_records_all_spacecrafts():
    simulation = make_simulation(max_sim_time=1.0)

    sc1 = make_mock_spacecraft(name="SC1")
    sc2 = make_mock_spacecraft(name="SC2")

    simulation.simulation_data.spacecrafts = [sc1, sc2]

    history = Mock()
    simulation.simulation_data.simulation_history = history

    simulation.simulate()

    # Initial state + one simulation step for each spacecraft.
    assert history.record_spacecraft.call_count == 4

    history.record_spacecraft.assert_any_call(
        "SC1",
        sc1.spacecraft_data,
    )

    history.record_spacecraft.assert_any_call(
        "SC2",
        sc2.spacecraft_data,
    )


# ---------------------------------------------------------------------------
# Spacecraft already beyond simulation time
# ---------------------------------------------------------------------------

def test_simulate_skips_spacecraft_past_max_sim_time():
    simulation = make_simulation(max_sim_time=1.0)

    spacecraft = make_mock_spacecraft(
        t=2.0,
        dt_master=1.0,
    )

    simulation.simulation_data.spacecrafts = [spacecraft]

    history = Mock()
    simulation.simulation_data.simulation_history = history

    simulation.simulate()

    spacecraft.update_mission_manager.assert_not_called()
    spacecraft.update_sensors.assert_not_called()
    spacecraft.update_navigation.assert_not_called()
    spacecraft.update_guidance.assert_not_called()
    spacecraft.update_control.assert_not_called()
    spacecraft.compute_actuation.assert_not_called()
    spacecraft.propagate_translational.assert_not_called()
    spacecraft.propagate_rotational.assert_not_called()


# ---------------------------------------------------------------------------
# Verbose output
# ---------------------------------------------------------------------------

def test_simulation_verbose_prints_initialization_and_completion(capsys):
    simulation = make_simulation(
        max_sim_time=1.0,
        verbose=True,
    )

    spacecraft = make_mock_spacecraft()

    simulation.simulation_data.spacecrafts = [spacecraft]
    simulation.simulation_data.simulation_history = Mock()

    simulation.simulate()

    captured = capsys.readouterr()

    assert "Initializing simulation..." in captured.out
    assert "Starting simulation..." in captured.out
    assert "Simulation initialized." in captured.out
    assert "Simulation completed." in captured.out