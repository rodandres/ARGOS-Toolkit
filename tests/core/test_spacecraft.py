import numpy as np
import pytest
from types import SimpleNamespace
from unittest.mock import Mock, MagicMock, patch

from argos.core.spacecraft import Spacecraft
from argos.general.dataclasses import SpacecraftData
from argos.sensors.sensor_base import SensorBase
from argos.actuators.actuators_base import ActuatorBase
from argos.core.mission_manager import MissionManager


def make_initial_state():
    return np.array([
        1.0, 2.0, 3.0,          # position
        4.0, 5.0, 6.0,          # velocity
        1.0, 0.0, 0.0, 0.0,     # attitude
        0.1, 0.2, 0.3,          # angular velocity
    ])


def make_phase(
    name="phase_1",
    guidance=None,
    navigation=None,
    controller=None,
    allocator=None,
    dt_nav=None,
    dt_guid=None,
    dt_control=None,
    translational_model=None,
    rotational_model=None,
    dt_propagation=1.0,
):
    return SimpleNamespace(
        name=name,
        guidance=guidance,
        navigation=navigation,
        controller=controller,
        allocator=allocator,
        dt_nav=dt_nav,
        dt_guid=dt_guid,
        dt_control=dt_control,
        translational_model=translational_model,
        rotational_model=rotational_model,
        dt_propagation=dt_propagation,
    )


def make_spacecraft_without_init():
    spacecraft = Spacecraft.__new__(Spacecraft)
    spacecraft.name = "SC"
    spacecraft.mass = 100.0
    spacecraft.inertia_tensor = np.eye(3)
    spacecraft.inertia_tensor_inv = np.eye(3)

    spacecraft.spacecraft_data = SpacecraftData()

    spacecraft.sensors = []
    spacecraft.actuators = []

    spacecraft.mission_manager = None
    spacecraft.parent = Mock()

    spacecraft.rotational_model = None
    spacecraft.translational_model = None

    spacecraft.current_phase = None
    spacecraft.current_guidance_law = None
    spacecraft.current_navigation_law = None
    spacecraft.current_control_law = None
    spacecraft.current_allocator = None

    spacecraft.current_navigation_dt = np.nan
    spacecraft.current_guidance_dt = np.nan
    spacecraft.current_control_dt = np.nan

    spacecraft.has_sensors = False
    spacecraft.has_actuators = False
    spacecraft.has_mission_manager = False
    spacecraft.has_navigation_law = False
    spacecraft.has_guidance_law = False
    spacecraft.has_control_law = False
    spacecraft.has_translational_model = False
    spacecraft.has_rotational_model = False

    spacecraft.verbose = False
    return spacecraft


# ============================================================================
# __init__
# ============================================================================

def test_init_sets_basic_attributes():
    inertia = np.diag([2.0, 3.0, 4.0])
    state = make_initial_state()

    spacecraft = Spacecraft(
        name="Chaser",
        mass=250.0,
        initial_state=state,
        inertia_tensor=inertia,
    )

    assert spacecraft.name == "Chaser"
    assert spacecraft.mass == 250.0
    np.testing.assert_allclose(spacecraft.inertia_tensor, inertia)
    np.testing.assert_allclose(
        spacecraft.inertia_tensor_inv,
        np.linalg.inv(inertia),
    )


def test_init_sets_initial_state():
    state = make_initial_state()

    spacecraft = Spacecraft(
        name="Chaser",
        mass=100.0,
        initial_state=state,
        inertia_tensor=np.eye(3),
    )

    np.testing.assert_allclose(
        spacecraft.spacecraft_data.true_state.position,
        state[0:3],
    )
    np.testing.assert_allclose(
        spacecraft.spacecraft_data.true_state.velocity,
        state[3:6],
    )
    np.testing.assert_allclose(
        spacecraft.spacecraft_data.true_state.attitude,
        state[6:10],
    )
    np.testing.assert_allclose(
        spacecraft.spacecraft_data.true_state.angular_velocity,
        state[10:13],
    )


def test_init_without_optional_components_sets_flags_false():
    spacecraft = Spacecraft(
        name="Chaser",
        mass=100.0,
        initial_state=make_initial_state(),
        inertia_tensor=np.eye(3),
    )

    assert spacecraft.has_sensors is False
    assert spacecraft.has_actuators is False
    assert spacecraft.has_mission_manager is False
    assert spacecraft.has_navigation_law is False
    assert spacecraft.has_guidance_law is False
    assert spacecraft.has_control_law is False
    assert spacecraft.has_translational_model is False
    assert spacecraft.has_rotational_model is False


def test_init_computes_inverse_inertia():
    inertia = np.diag([2.0, 4.0, 5.0])

    spacecraft = Spacecraft(
        name="Chaser",
        mass=100.0,
        initial_state=make_initial_state(),
        inertia_tensor=inertia,
    )

    np.testing.assert_array_almost_equal(
        spacecraft.inertia_tensor_inv,
        np.diag([0.5, 0.25, 0.2]),
    )


# ============================================================================
# _init_state_variables
# ============================================================================

def test_init_state_variables_assigns_state_components():
    spacecraft = make_spacecraft_without_init()
    state = make_initial_state()

    spacecraft._init_state_variables(state)

    np.testing.assert_allclose(
        spacecraft.spacecraft_data.true_state.position,
        state[0:3],
    )
    np.testing.assert_allclose(
        spacecraft.spacecraft_data.true_state.velocity,
        state[3:6],
    )
    np.testing.assert_allclose(
        spacecraft.spacecraft_data.true_state.attitude,
        state[6:10],
    )
    np.testing.assert_allclose(
        spacecraft.spacecraft_data.true_state.angular_velocity,
        state[10:13],
    )


# ============================================================================
# Naming sensors / actuators
# ============================================================================

def test_set_sensors_names_assigns_names_to_unnamed_sensors():
    spacecraft = make_spacecraft_without_init()

    sensor_1 = Mock(spec=SensorBase)
    sensor_1.name = None

    sensor_2 = Mock(spec=SensorBase)
    sensor_2.name = "Existing"

    sensor_3 = Mock(spec=SensorBase)
    sensor_3.name = None

    spacecraft.sensors = [sensor_1, sensor_2, sensor_3]
    spacecraft.has_sensors = True

    spacecraft._set_sensors_names()

    assert sensor_1.name == "Sensor_0"
    assert sensor_2.name == "Existing"
    assert sensor_3.name == "Sensor_1"


def test_set_sensors_names_does_nothing_without_sensors():
    spacecraft = make_spacecraft_without_init()
    spacecraft.has_sensors = False

    spacecraft._set_sensors_names()


def test_set_actuators_names_assigns_names_to_unnamed_actuators():
    spacecraft = make_spacecraft_without_init()

    actuator_1 = Mock(spec=ActuatorBase)
    actuator_1.name = None

    actuator_2 = Mock(spec=ActuatorBase)
    actuator_2.name = "Existing"

    actuator_3 = Mock(spec=ActuatorBase)
    actuator_3.name = None

    spacecraft.actuators = [actuator_1, actuator_2, actuator_3]
    spacecraft.has_actuators = True

    spacecraft._set_actuators_names()

    assert actuator_1.name == "Actuator_0"
    assert actuator_2.name == "Existing"
    assert actuator_3.name == "Actuator_1"


# ============================================================================
# _check_initialization
# ============================================================================

def test_check_initialization_detects_valid_sensors():
    spacecraft = make_spacecraft_without_init()

    sensor = Mock(spec=SensorBase)
    spacecraft.sensors = [sensor]

    spacecraft._check_initialization()

    assert spacecraft.has_sensors is True


def test_check_initialization_detects_valid_actuators():
    spacecraft = make_spacecraft_without_init()

    actuator = Mock(spec=ActuatorBase)
    spacecraft.actuators = [actuator]

    spacecraft._check_initialization()

    assert spacecraft.has_actuators is True


def test_check_initialization_ignores_invalid_sensor_list():
    spacecraft = make_spacecraft_without_init()

    spacecraft.sensors = [Mock()]

    spacecraft._check_initialization()

    assert spacecraft.has_sensors is False


def test_check_initialization_ignores_invalid_actuator_list():
    spacecraft = make_spacecraft_without_init()

    spacecraft.actuators = [Mock()]

    spacecraft._check_initialization()

    assert spacecraft.has_actuators is False


# ============================================================================
# update_sensor_name
# ============================================================================

def test_update_sensor_name_changes_matching_sensor():
    spacecraft = make_spacecraft_without_init()

    sensor = Mock(spec=SensorBase)
    sensor.name = "IMU"

    spacecraft.sensors = [sensor]

    spacecraft.update_sensor_name("IMU", "IMU_1")

    assert sensor.name == "IMU_1"


def test_update_sensor_name_only_changes_first_match():
    spacecraft = make_spacecraft_without_init()

    sensor_1 = Mock(spec=SensorBase)
    sensor_1.name = "IMU"

    sensor_2 = Mock(spec=SensorBase)
    sensor_2.name = "IMU"

    spacecraft.sensors = [sensor_1, sensor_2]

    spacecraft.update_sensor_name("IMU", "IMU_1")

    assert sensor_1.name == "IMU_1"
    assert sensor_2.name == "IMU"


def test_update_sensor_name_missing_sensor_does_not_change_anything(capsys):
    spacecraft = make_spacecraft_without_init()

    sensor = Mock(spec=SensorBase)
    sensor.name = "IMU"

    spacecraft.sensors = [sensor]

    spacecraft.update_sensor_name("GPS", "GPS_1")

    assert sensor.name == "IMU"
    assert "No sensor found" in capsys.readouterr().out


# ============================================================================
# update_actuator_name
# ============================================================================

def test_update_actuator_name_changes_matching_actuator():
    spacecraft = make_spacecraft_without_init()

    actuator = Mock(spec=ActuatorBase)
    actuator.name = "RCS"

    spacecraft.actuators = [actuator]

    spacecraft.update_actuator_name("RCS", "RCS_1")

    assert actuator.name == "RCS_1"


def test_update_actuator_name_missing_actuator_does_not_change_anything(capsys):
    spacecraft = make_spacecraft_without_init()

    actuator = Mock(spec=ActuatorBase)
    actuator.name = "RCS"

    spacecraft.actuators = [actuator]

    spacecraft.update_actuator_name("OTHER", "RCS_1")

    assert actuator.name == "RCS"
    assert "No actuator found" in capsys.readouterr().out


# ============================================================================
# _check_and_update_phase_info
# ============================================================================

def test_phase_info_updates_guidance():
    spacecraft = make_spacecraft_without_init()

    guidance = Mock()
    phase = make_phase(
        guidance=guidance,
        dt_guid=0.2,
        dt_propagation=1.0,
    )

    spacecraft.parent = Mock()

    spacecraft._check_and_update_phase_info(phase)

    assert spacecraft.current_phase is phase
    assert spacecraft.has_guidance_law is True
    assert spacecraft.current_guidance_law is guidance
    assert spacecraft.current_guidance_dt == 0.2


def test_phase_info_clears_guidance_when_absent():
    spacecraft = make_spacecraft_without_init()

    spacecraft.current_guidance_law = Mock()
    spacecraft.current_guidance_dt = 0.5
    spacecraft.has_guidance_law = True

    phase = make_phase(
        guidance=None,
        dt_guid=None,
        dt_propagation=1.0,
    )

    spacecraft._check_and_update_phase_info(phase)

    assert spacecraft.has_guidance_law is False
    assert spacecraft.current_guidance_law is None
    assert np.isnan(spacecraft.current_guidance_dt)


def test_phase_info_updates_navigation():
    spacecraft = make_spacecraft_without_init()

    navigation = Mock()
    phase = make_phase(
        navigation=navigation,
        dt_nav=0.3,
        dt_propagation=1.0,
    )

    spacecraft._check_and_update_phase_info(phase)

    assert spacecraft.has_navigation_law is True
    assert spacecraft.current_navigation_law is navigation
    assert spacecraft.current_navigation_dt == 0.3


def test_phase_info_clears_navigation_when_absent():
    spacecraft = make_spacecraft_without_init()

    spacecraft.current_navigation_law = Mock()
    spacecraft.current_navigation_dt = 0.5
    spacecraft.has_navigation_law = True

    phase = make_phase(
        navigation=None,
        dt_nav=None,
        dt_propagation=1.0,
    )

    spacecraft._check_and_update_phase_info(phase)

    assert spacecraft.has_navigation_law is False
    assert spacecraft.current_navigation_law is None
    assert np.isnan(spacecraft.current_navigation_dt)


def test_phase_info_updates_controller_and_allocator():
    spacecraft = make_spacecraft_without_init()

    controller = Mock()
    allocator = Mock()

    phase = make_phase(
        controller=controller,
        allocator=allocator,
        dt_control=0.4,
        dt_propagation=1.0,
    )

    spacecraft.actuators = ["actuator_1", "actuator_2"]

    spacecraft._check_and_update_phase_info(phase)

    assert spacecraft.has_control_law is True
    assert spacecraft.current_control_law is controller
    assert spacecraft.current_allocator is allocator
    assert spacecraft.current_control_dt == 0.4

    allocator.set_actuators.assert_called_once_with(spacecraft.actuators)


def test_phase_info_clears_controller_when_absent():
    spacecraft = make_spacecraft_without_init()

    spacecraft.current_control_law = Mock()
    spacecraft.current_allocator = Mock()
    spacecraft.current_control_dt = 0.5
    spacecraft.has_control_law = True

    phase = make_phase(
        controller=None,
        allocator=None,
        dt_control=None,
        dt_propagation=1.0,
    )

    spacecraft._check_and_update_phase_info(phase)

    assert spacecraft.has_control_law is False
    assert spacecraft.current_control_law is None
    assert spacecraft.current_allocator is None
    assert np.isnan(spacecraft.current_control_dt)


def test_phase_info_updates_translational_model():
    spacecraft = make_spacecraft_without_init()

    model = Mock()

    phase = make_phase(
        translational_model=model,
        dt_propagation=1.0,
    )

    spacecraft._check_and_update_phase_info(phase)

    assert spacecraft.has_translational_model is True
    assert spacecraft.translational_model is model


def test_phase_info_clears_translational_model_when_absent():
    spacecraft = make_spacecraft_without_init()

    spacecraft.has_translational_model = True
    spacecraft.translational_model = Mock()

    phase = make_phase(
        translational_model=None,
        dt_propagation=1.0,
    )

    spacecraft._check_and_update_phase_info(phase)

    assert spacecraft.has_translational_model is False
    assert spacecraft.translational_model is None


def test_phase_info_updates_rotational_model():
    spacecraft = make_spacecraft_without_init()

    model = Mock()

    phase = make_phase(
        rotational_model=model,
        dt_propagation=1.0,
    )

    spacecraft._check_and_update_phase_info(phase)

    assert spacecraft.has_rotational_model is True
    assert spacecraft.rotational_model is model


def test_phase_info_clears_rotational_model_when_absent():
    spacecraft = make_spacecraft_without_init()

    spacecraft.has_rotational_model = True
    spacecraft.rotational_model = Mock()

    phase = make_phase(
        rotational_model=None,
        dt_propagation=1.0,
    )

    spacecraft._check_and_update_phase_info(phase)

    assert spacecraft.has_rotational_model is False
    assert spacecraft.rotational_model is None


def test_phase_info_updates_propagation_dt_and_master_dt():
    spacecraft = make_spacecraft_without_init()

    phase = make_phase(
        dt_propagation=0.5,
    )

    spacecraft._check_and_update_phase_info(phase)

    assert spacecraft.spacecraft_data.current_propagation_dt == 0.5
    assert spacecraft.spacecraft_data.current_master_dt == 0.5


def test_phase_info_records_initial_transition():
    spacecraft = make_spacecraft_without_init()

    phase = make_phase(
        name="initial",
        dt_propagation=1.0,
    )

    spacecraft._check_and_update_phase_info(phase)

    spacecraft.parent.record_transition.assert_called_once()

    args = spacecraft.parent.record_transition.call_args.args

    assert args[0] == spacecraft.name
    event_info = args[1]

    assert event_info.from_phase == "Initial State"
    assert event_info.to_phase == "initial"
    assert event_info.transition_name == "Initial Setup"


def test_phase_info_records_transition_update():
    spacecraft = make_spacecraft_without_init()

    phase = make_phase(
        name="phase_2",
        dt_propagation=1.0,
    )

    update_info = {
        "from_phase": "phase_1",
        "to_phase": "phase_2",
        "via": "condition_met",
    }

    spacecraft._check_and_update_phase_info(
        phase,
        update_info,
    )

    event_info = spacecraft.parent.record_transition.call_args.args[1]

    assert event_info.from_phase == "phase_1"
    assert event_info.to_phase == "phase_2"
    assert event_info.transition_name == "condition_met"


# ============================================================================
# get_dts / get_min_dt
# ============================================================================

def test_get_dts_returns_all_component_dts():
    spacecraft = make_spacecraft_without_init()

    spacecraft.current_navigation_dt = 0.1
    spacecraft.current_guidance_dt = 0.2
    spacecraft.current_control_dt = 0.3
    spacecraft.spacecraft_data.current_propagation_dt = 0.4

    assert spacecraft.get_dts() == (0.1, 0.2, 0.3, 0.4)


def test_get_min_dt_returns_minimum_component_dt():
    spacecraft = make_spacecraft_without_init()

    spacecraft.current_navigation_dt = 0.2
    spacecraft.current_guidance_dt = 0.5
    spacecraft.current_control_dt = 0.1
    spacecraft.spacecraft_data.current_propagation_dt = 1.0

    assert spacecraft.get_min_dt() == 0.1


def test_get_min_dt_includes_sensor_sample_rate():
    spacecraft = make_spacecraft_without_init()

    sensor = Mock(spec=SensorBase)
    sensor.sample_rate_sec = 0.05

    spacecraft.sensors = [sensor]
    spacecraft.has_sensors = True

    spacecraft.current_navigation_dt = 0.2
    spacecraft.current_guidance_dt = 0.5
    spacecraft.current_control_dt = 0.1
    spacecraft.spacecraft_data.current_propagation_dt = 1.0

    assert spacecraft.get_min_dt() == 0.05


def test_get_min_dt_ignores_sensor_without_sample_rate():
    spacecraft = make_spacecraft_without_init()

    sensor = Mock(spec=SensorBase)
    sensor.sample_rate_sec = None

    spacecraft.sensors = [sensor]
    spacecraft.has_sensors = True

    spacecraft.current_navigation_dt = 0.2
    spacecraft.current_guidance_dt = 0.5
    spacecraft.current_control_dt = 0.1
    spacecraft.spacecraft_data.current_propagation_dt = 1.0

    assert spacecraft.get_min_dt() == 0.1


def test_get_min_dt_returns_none_when_no_dt_available():
    spacecraft = make_spacecraft_without_init()

    spacecraft.current_navigation_dt = None
    spacecraft.current_guidance_dt = None
    spacecraft.current_control_dt = None
    spacecraft.spacecraft_data.current_propagation_dt = None

    spacecraft.has_sensors = False

    assert spacecraft.get_min_dt() is None


# ============================================================================
# _should_compute
# ============================================================================

def test_should_compute_returns_true_on_matching_tick():
    spacecraft = make_spacecraft_without_init()

    spacecraft.spacecraft_data.tick = 10
    spacecraft.spacecraft_data.current_master_dt = 0.1

    assert spacecraft._should_compute(None, 0.5) is True


def test_should_compute_returns_false_on_non_matching_tick():
    spacecraft = make_spacecraft_without_init()

    spacecraft.spacecraft_data.tick = 7
    spacecraft.spacecraft_data.current_master_dt = 0.1

    assert spacecraft._should_compute(None, 0.5) is False


# ============================================================================
# update_mission_manager
# ============================================================================

def test_update_mission_manager_does_nothing_without_manager():
    spacecraft = make_spacecraft_without_init()

    spacecraft.has_mission_manager = False

    spacecraft.update_mission_manager(Mock())

    assert spacecraft.current_phase is None


def test_update_mission_manager_does_nothing_when_phase_does_not_change():
    spacecraft = make_spacecraft_without_init()

    manager = Mock(spec=MissionManager)
    manager.update.return_value = (False, None)

    spacecraft.mission_manager = manager
    spacecraft.has_mission_manager = True

    spacecraft.update_mission_manager("simulation_data")

    manager.update.assert_called_once_with("simulation_data")


def test_update_mission_manager_updates_phase_when_changed():
    spacecraft = make_spacecraft_without_init()

    manager = Mock(spec=MissionManager)

    new_phase = make_phase(
        name="phase_2",
        dt_propagation=1.0,
    )

    update_info = {
        "from_phase": "phase_1",
        "to_phase": "phase_2",
        "via": "transition_1",
    }

    manager.current_phase = new_phase
    manager.update.return_value = (True, update_info)

    spacecraft.mission_manager = manager
    spacecraft.has_mission_manager = True

    spacecraft.update_mission_manager("simulation_data")

    assert spacecraft.current_phase is new_phase


# ============================================================================
# update_sensors
# ============================================================================

def test_update_sensors_does_nothing_without_sensors():
    spacecraft = make_spacecraft_without_init()

    spacecraft.has_sensors = False

    spacecraft.update_sensors("simulation_data")


def test_update_sensors_updates_each_sensor():
    spacecraft = make_spacecraft_without_init()

    sensor_1 = Mock(spec=SensorBase)
    sensor_2 = Mock(spec=SensorBase)

    spacecraft.sensors = [sensor_1, sensor_2]
    spacecraft.has_sensors = True

    simulation_data = Mock()

    spacecraft.update_sensors(simulation_data)

    sensor_1.update.assert_called_once_with(
        spacecraft.spacecraft_data,
        simulation_data,
    )
    sensor_2.update.assert_called_once_with(
        spacecraft.spacecraft_data,
        simulation_data,
    )


# ============================================================================
# update_navigation
# ============================================================================

def test_update_navigation_does_nothing_without_navigation():
    spacecraft = make_spacecraft_without_init()

    spacecraft.has_navigation_law = False

    spacecraft.update_navigation(Mock())


def test_update_navigation_computes_when_due():
    spacecraft = make_spacecraft_without_init()

    spacecraft.has_navigation_law = True
    spacecraft.current_navigation_dt = 0.5
    spacecraft.spacecraft_data.tick = 10
    spacecraft.spacecraft_data.current_master_dt = 0.1

    navigation = Mock()
    estimation = Mock()

    navigation.estimate.return_value = estimation
    spacecraft.current_navigation_law = navigation

    spacecraft.update_navigation("simulation_data")

    navigation.estimate.assert_called_once_with(spacecraft.sensors)
    assert spacecraft.spacecraft_data.navigation_data is estimation


def test_update_navigation_does_not_compute_when_not_due():
    spacecraft = make_spacecraft_without_init()

    spacecraft.has_navigation_law = True
    spacecraft.current_navigation_dt = 0.5
    spacecraft.spacecraft_data.tick = 7
    spacecraft.spacecraft_data.current_master_dt = 0.1

    navigation = Mock()
    spacecraft.current_navigation_law = navigation

    spacecraft.update_navigation("simulation_data")

    navigation.estimate.assert_not_called()


# ============================================================================
# update_guidance
# ============================================================================

def test_update_guidance_does_nothing_without_guidance():
    spacecraft = make_spacecraft_without_init()

    spacecraft.has_guidance_law = False

    spacecraft.update_guidance(Mock())


def test_update_guidance_computes_when_due():
    spacecraft = make_spacecraft_without_init()

    spacecraft.has_guidance_law = True
    spacecraft.current_guidance_dt = 0.5
    spacecraft.spacecraft_data.tick = 10
    spacecraft.spacecraft_data.current_master_dt = 0.1

    guidance = Mock()
    reference = Mock()

    guidance.compute_reference.return_value = reference
    spacecraft.current_guidance_law = guidance

    simulation_data = Mock()
    spacecraft.spacecraft_data.navigation_data = Mock()

    spacecraft.update_guidance(simulation_data)

    guidance.compute_reference.assert_called_once_with(
        spacecraft.spacecraft_data.navigation_data,
        simulation_data,
    )

    assert spacecraft.spacecraft_data.guidance_data is reference


def test_update_guidance_does_not_compute_when_not_due():
    spacecraft = make_spacecraft_without_init()

    spacecraft.has_guidance_law = True
    spacecraft.current_guidance_dt = 0.5
    spacecraft.spacecraft_data.tick = 7
    spacecraft.spacecraft_data.current_master_dt = 0.1

    guidance = Mock()
    spacecraft.current_guidance_law = guidance

    spacecraft.update_guidance(Mock())

    guidance.compute_reference.assert_not_called()


# ============================================================================
# update_control
# ============================================================================

def test_update_control_does_nothing_without_control():
    spacecraft = make_spacecraft_without_init()

    spacecraft.has_control_law = False

    spacecraft.update_control(Mock())


def test_update_control_computes_and_allocates_when_due():
    spacecraft = make_spacecraft_without_init()

    spacecraft.has_control_law = True
    spacecraft.current_control_dt = 0.5
    spacecraft.spacecraft_data.tick = 10
    spacecraft.spacecraft_data.current_master_dt = 0.1

    controller = Mock()
    allocator = Mock()

    control_output = Mock()

    controller.compute_control.return_value = control_output

    spacecraft.current_control_law = controller
    spacecraft.current_allocator = allocator

    spacecraft.spacecraft_data.navigation_data = Mock()
    spacecraft.spacecraft_data.guidance_data = Mock()

    simulation_data = Mock()

    spacecraft.update_control(simulation_data)

    controller.compute_control.assert_called_once_with(
        spacecraft.spacecraft_data.navigation_data,
        spacecraft.spacecraft_data.guidance_data,
    )

    allocator.allocate.assert_called_once_with(control_output)

    assert spacecraft.spacecraft_data.control_data is control_output


def test_update_control_does_not_compute_when_not_due():
    spacecraft = make_spacecraft_without_init()

    spacecraft.has_control_law = True
    spacecraft.current_control_dt = 0.5
    spacecraft.spacecraft_data.tick = 7
    spacecraft.spacecraft_data.current_master_dt = 0.1

    controller = Mock()
    allocator = Mock()

    spacecraft.current_control_law = controller
    spacecraft.current_allocator = allocator

    spacecraft.update_control(Mock())

    controller.compute_control.assert_not_called()
    allocator.allocate.assert_not_called()


# ============================================================================
# compute_actuation
# ============================================================================

def test_compute_actuation_does_nothing_without_actuators():
    spacecraft = make_spacecraft_without_init()

    initial_force = spacecraft.spacecraft_data.current_force_exerted.copy()
    initial_torque = spacecraft.spacecraft_data.current_torque_exerted.copy()

    spacecraft.has_actuators = False

    spacecraft.compute_actuation(Mock())

    np.testing.assert_allclose(
        spacecraft.spacecraft_data.current_force_exerted,
        initial_force,
    )
    np.testing.assert_allclose(
        spacecraft.spacecraft_data.current_torque_exerted,
        initial_torque,
    )


def test_compute_actuation_accumulates_actuator_outputs():
    spacecraft = make_spacecraft_without_init()

    actuator_1 = Mock(spec=ActuatorBase)
    actuator_2 = Mock(spec=ActuatorBase)

    output_1 = Mock()
    output_1.force = np.array([1.0, 2.0, 3.0])
    output_1.torque = np.array([0.1, 0.2, 0.3])

    output_2 = Mock()
    output_2.force = np.array([4.0, 5.0, 6.0])
    output_2.torque = np.array([0.4, 0.5, 0.6])

    actuator_1.get_output.return_value = output_1
    actuator_2.get_output.return_value = output_2

    spacecraft.actuators = [actuator_1, actuator_2]
    spacecraft.has_actuators = True
    spacecraft.spacecraft_data.t = 12.0

    spacecraft.compute_actuation(Mock())

    actuator_1.update.assert_called_once_with(12.0)
    actuator_2.update.assert_called_once_with(12.0)

    np.testing.assert_allclose(
        spacecraft.spacecraft_data.current_force_exerted,
        np.array([5.0, 7.0, 9.0]),
    )

    np.testing.assert_allclose(
        spacecraft.spacecraft_data.current_torque_exerted,
        np.array([0.5, 0.7, 0.9]),
    )


# ============================================================================
# Propagation
# ============================================================================

def test_propagate_translational_does_nothing_without_model():
    spacecraft = make_spacecraft_without_init()

    spacecraft.has_translational_model = False

    spacecraft.propagate_translational(
        "simulation_data",
        "environment",
    )


def test_propagate_translational_calls_model_when_due():
    spacecraft = make_spacecraft_without_init()

    model = Mock()

    spacecraft.translational_model = model
    spacecraft.has_translational_model = True

    spacecraft.spacecraft_data.tick = 10
    spacecraft.spacecraft_data.current_master_dt = 0.1
    spacecraft.spacecraft_data.current_propagation_dt = 0.5

    simulation_data = Mock()
    environment = Mock()

    spacecraft.propagate_translational(
        simulation_data,
        environment,
    )

    model.propagate.assert_called_once_with(
        spacecraft,
        simulation_data,
        environment,
    )


def test_propagate_translational_does_not_call_model_when_not_due():
    spacecraft = make_spacecraft_without_init()

    model = Mock()

    spacecraft.translational_model = model
    spacecraft.has_translational_model = True

    spacecraft.spacecraft_data.tick = 7
    spacecraft.spacecraft_data.current_master_dt = 0.1
    spacecraft.spacecraft_data.current_propagation_dt = 0.5

    spacecraft.propagate_translational(
        Mock(),
        Mock(),
    )

    model.propagate.assert_not_called()


def test_propagate_rotational_does_nothing_without_model():
    spacecraft = make_spacecraft_without_init()

    spacecraft.has_rotational_model = False

    spacecraft.propagate_rotational(
        "simulation_data",
        "environment",
    )


def test_propagate_rotational_calls_model_when_due():
    spacecraft = make_spacecraft_without_init()

    model = Mock()

    spacecraft.rotational_model = model
    spacecraft.has_rotational_model = True

    spacecraft.spacecraft_data.tick = 10
    spacecraft.spacecraft_data.current_master_dt = 0.1
    spacecraft.spacecraft_data.current_propagation_dt = 0.5

    simulation_data = Mock()
    environment = Mock()

    spacecraft.propagate_rotational(
        simulation_data,
        environment,
    )

    model.propagate.assert_called_once_with(
        spacecraft,
        simulation_data,
        environment,
    )


def test_propagate_rotational_does_not_call_model_when_not_due():
    spacecraft = make_spacecraft_without_init()

    model = Mock()

    spacecraft.rotational_model = model
    spacecraft.has_rotational_model = True

    spacecraft.spacecraft_data.tick = 7
    spacecraft.spacecraft_data.current_master_dt = 0.1
    spacecraft.spacecraft_data.current_propagation_dt = 0.5

    spacecraft.propagate_rotational(
        Mock(),
        Mock(),
    )

    model.propagate.assert_not_called()


# ============================================================================
# change_target
# ============================================================================

def test_change_target_updates_target_name():
    spacecraft = make_spacecraft_without_init()

    spacecraft.change_target("Target")

    assert spacecraft.spacecraft_data.target_name == "Target"


# ============================================================================
# Information methods
# ============================================================================

def test_show_spacecraft_basic_info(capsys):
    spacecraft = make_spacecraft_without_init()

    spacecraft.show_spacecraft_basic_info()

    output = capsys.readouterr().out

    assert "Spacecraft Name: SC" in output
    assert "Mass: 100.0 kg" in output
    assert "Inertia Tensor:" in output


def test_show_spacecraft_gnc_info(capsys):
    spacecraft = make_spacecraft_without_init()

    spacecraft.show_spacecraft_gnc_info()

    assert "NOT IMPLEMENTED" in capsys.readouterr().out


def test_show_spacecraft_state_info(capsys):
    spacecraft = make_spacecraft_without_init()

    spacecraft.show_spacecraft_state_info()

    assert "NOT IMPLEMENTED" in capsys.readouterr().out


def test_show_spacecraft_reference_info(capsys):
    spacecraft = make_spacecraft_without_init()

    spacecraft.show_spacecraft_reference_info()

    assert "NOT IMPLEMENTED" in capsys.readouterr().out


def test_show_sensors_info_without_sensors(capsys):
    spacecraft = make_spacecraft_without_init()

    spacecraft.has_sensors = False

    spacecraft.show_sensors_info()

    assert "No sensors available" in capsys.readouterr().out


def test_show_sensors_info_calls_print_info():
    spacecraft = make_spacecraft_without_init()

    sensor = Mock(spec=SensorBase)

    spacecraft.sensors = [sensor]
    spacecraft.has_sensors = True

    spacecraft.show_sensors_info()

    sensor.print_info.assert_called_once()


def test_show_faults_info_delegates_to_fault_manager():
    spacecraft = make_spacecraft_without_init()

    spacecraft.fault_manager = Mock()

    spacecraft.show_faults_info()

    spacecraft.fault_manager.show_faults_info.assert_called_once()


def test_show_spacecraft_all_info_calls_all_sections():
    spacecraft = make_spacecraft_without_init()

    spacecraft.show_spacecraft_basic_info = Mock()
    spacecraft.show_spacecraft_gnc_info = Mock()
    spacecraft.show_spacecraft_state_info = Mock()
    spacecraft.show_spacecraft_reference_info = Mock()
    spacecraft.show_sensors_info = Mock()
    spacecraft.show_faults_info = Mock()

    spacecraft.show_spacecraft_all_info()

    spacecraft.show_spacecraft_basic_info.assert_called_once()
    spacecraft.show_spacecraft_gnc_info.assert_called_once()
    spacecraft.show_spacecraft_state_info.assert_called_once()
    spacecraft.show_spacecraft_reference_info.assert_called_once()
    spacecraft.show_sensors_info.assert_called_once()
    spacecraft.show_faults_info.assert_called_once()

def test_change_target_prints_when_verbose(capsys):
    spacecraft = make_spacecraft_without_init()
    spacecraft.verbose = True

    spacecraft.change_target("Target")

    assert spacecraft.spacecraft_data.target_name == "Target"
    assert "target changed to 'Target'" in capsys.readouterr().out    

def test_change_target_does_not_print_when_not_verbose(capsys):
    spacecraft = make_spacecraft_without_init()
    spacecraft.verbose = False

    spacecraft.change_target("Target")

    assert spacecraft.spacecraft_data.target_name == "Target"
    assert capsys.readouterr().out == ""    