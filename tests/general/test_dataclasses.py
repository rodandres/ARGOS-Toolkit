import numpy as np

from argos.general.dataclasses import (
    ActuatorOutput,
    ControlOutput,
    GuidanceOutput,
    MissionPhase,
    NavigationOutput,
    SimulationData,
    SpacecraftData,
    StateVariables,
)


def test_mission_phase_defaults():
    phase = MissionPhase(name="initial")

    assert phase.name == "initial"

    assert phase.rotational_model is None
    assert phase.translational_model is None

    assert phase.guidance is None
    assert phase.navigation is None
    assert phase.controller is None
    assert phase.allocator is None

    assert phase.dt_nav is None
    assert phase.dt_guid is None
    assert phase.dt_control is None
    assert phase.dt_propagation is None


def test_mission_phase_stores_configuration():
    phase = MissionPhase(
        name="coast",
        dt_nav=0.5,
        dt_guid=1.0,
        dt_control=0.1,
        dt_propagation=0.01,
    )

    assert phase.name == "coast"
    assert phase.dt_nav == 0.5
    assert phase.dt_guid == 1.0
    assert phase.dt_control == 0.1
    assert phase.dt_propagation == 0.01


def test_simulation_data_defaults():
    simulation_data = SimulationData(
        max_sim_time=100.0,
        spacecrafts=[],
    )

    assert simulation_data.max_sim_time == 100.0
    assert simulation_data.spacecrafts == []

    assert simulation_data.dt_master == 0.1
    assert simulation_data.t == 0.0
    assert simulation_data.tick == 0
    assert simulation_data.verbose is False

    assert simulation_data.simulation_history is None

    assert np.array_equal(
        simulation_data.DCM_inertial_to_body,
        np.eye(3),
    )


def test_simulation_data_stores_configuration():
    spacecrafts = ["SC1", "SC2"]

    simulation_data = SimulationData(
        max_sim_time=500.0,
        spacecrafts=spacecrafts,
        dt_master=0.5,
        t=10.0,
        tick=20,
        verbose=True,
    )

    assert simulation_data.max_sim_time == 500.0
    assert simulation_data.spacecrafts is spacecrafts
    assert simulation_data.dt_master == 0.5
    assert simulation_data.t == 10.0
    assert simulation_data.tick == 20
    assert simulation_data.verbose is True


def test_simulation_data_default_DCM_is_independent():
    data_1 = SimulationData(
        max_sim_time=100.0,
        spacecrafts=[],
    )

    data_2 = SimulationData(
        max_sim_time=100.0,
        spacecrafts=[],
    )

    data_1.DCM_inertial_to_body[0, 0] = 99.0

    assert data_2.DCM_inertial_to_body[0, 0] == 1.0


def test_state_variables_defaults():
    state = StateVariables()

    assert np.array_equal(
        state.position,
        np.zeros(3),
    )

    assert np.array_equal(
        state.velocity,
        np.zeros(3),
    )

    assert np.array_equal(
        state.acceleration,
        np.zeros(3),
    )

    assert np.array_equal(
        state.attitude,
        np.array([0, 0, 0, 1]),
    )

    assert np.array_equal(
        state.angular_velocity,
        np.zeros(3),
    )

    assert np.array_equal(
        state.angular_acceleration,
        np.zeros(3),
    )

    assert state.frame_type is None


def test_state_variables_default_arrays_are_independent():
    state_1 = StateVariables()
    state_2 = StateVariables()

    state_1.position[0] = 100.0
    state_1.velocity[1] = 200.0
    state_1.attitude[3] = 0.0

    assert state_2.position[0] == 0.0
    assert state_2.velocity[1] == 0.0
    assert state_2.attitude[3] == 1


def test_state_variables_stores_values():
    state = StateVariables(
        position=np.array([1.0, 2.0, 3.0]),
        velocity=np.array([4.0, 5.0, 6.0]),
        acceleration=np.array([7.0, 8.0, 9.0]),
        attitude=np.array([0.0, 0.0, 0.0, 1.0]),
        angular_velocity=np.array([0.1, 0.2, 0.3]),
        angular_acceleration=np.array([0.4, 0.5, 0.6]),
        frame_type="inertial",
    )

    assert np.array_equal(
        state.position,
        [1.0, 2.0, 3.0],
    )

    assert np.array_equal(
        state.velocity,
        [4.0, 5.0, 6.0],
    )

    assert np.array_equal(
        state.acceleration,
        [7.0, 8.0, 9.0],
    )

    assert np.array_equal(
        state.attitude,
        [0.0, 0.0, 0.0, 1.0],
    )

    assert np.array_equal(
        state.angular_velocity,
        [0.1, 0.2, 0.3],
    )

    assert np.array_equal(
        state.angular_acceleration,
        [0.4, 0.5, 0.6],
    )

    assert state.frame_type == "inertial"


def test_navigation_output_defaults():
    navigation = NavigationOutput()

    assert isinstance(
        navigation.spacecraft_state,
        StateVariables,
    )

    assert isinstance(
        navigation.target_state,
        StateVariables,
    )

    assert np.array_equal(
        navigation.spacecraft_state.position,
        np.zeros(3),
    )

    assert np.array_equal(
        navigation.target_state.position,
        np.zeros(3),
    )


def test_navigation_output_states_are_independent():
    navigation_1 = NavigationOutput()
    navigation_2 = NavigationOutput()

    navigation_1.spacecraft_state.position[0] = 100.0

    assert navigation_2.spacecraft_state.position[0] == 0.0


def test_guidance_output_defaults():
    guidance = GuidanceOutput()

    assert isinstance(
        guidance.state,
        StateVariables,
    )

    assert np.array_equal(
        guidance.state.position,
        np.zeros(3),
    )


def test_guidance_outputs_are_independent():
    guidance_1 = GuidanceOutput()
    guidance_2 = GuidanceOutput()

    guidance_1.state.position[0] = 100.0

    assert guidance_2.state.position[0] == 0.0


def test_control_output_defaults():
    control = ControlOutput()

    assert np.array_equal(
        control.force,
        np.zeros(3),
    )

    assert np.array_equal(
        control.torque,
        np.zeros(3),
    )


def test_control_output_arrays_are_independent():
    control_1 = ControlOutput()
    control_2 = ControlOutput()

    control_1.force[0] = 100.0
    control_1.torque[1] = 200.0

    assert control_2.force[0] == 0.0
    assert control_2.torque[1] == 0.0


def test_spacecraft_data_defaults():
    data = SpacecraftData()

    assert data.t == 0.0
    assert data.tick == 0
    assert data.current_master_dt == 0.1
    assert data.current_propagation_dt == 0.1

    assert isinstance(
        data.true_state,
        StateVariables,
    )

    assert isinstance(
        data.navigation_data,
        NavigationOutput,
    )

    assert isinstance(
        data.guidance_data,
        GuidanceOutput,
    )

    assert isinstance(
        data.control_data,
        ControlOutput,
    )

    assert np.array_equal(
        data.current_force_exerted,
        np.zeros(3),
    )

    assert np.array_equal(
        data.current_torque_exerted,
        np.zeros(3),
    )

    assert data.target_name is None


def test_spacecraft_data_nested_defaults_are_independent():
    data_1 = SpacecraftData()
    data_2 = SpacecraftData()

    data_1.true_state.position[0] = 100.0
    data_1.navigation_data.spacecraft_state.position[0] = 200.0
    data_1.guidance_data.state.position[0] = 300.0
    data_1.control_data.force[0] = 400.0
    data_1.current_force_exerted[0] = 500.0

    assert data_2.true_state.position[0] == 0.0
    assert data_2.navigation_data.spacecraft_state.position[0] == 0.0
    assert data_2.guidance_data.state.position[0] == 0.0
    assert data_2.control_data.force[0] == 0.0
    assert data_2.current_force_exerted[0] == 0.0


def test_spacecraft_data_stores_values():
    true_state = StateVariables(
        position=np.array([1.0, 2.0, 3.0]),
    )

    data = SpacecraftData(
        t=10.0,
        tick=100,
        current_master_dt=0.2,
        current_propagation_dt=0.01,
        true_state=true_state,
        target_name="Target",
    )

    assert data.t == 10.0
    assert data.tick == 100
    assert data.current_master_dt == 0.2
    assert data.current_propagation_dt == 0.01
    assert data.true_state is true_state
    assert data.target_name == "Target"


def test_actuator_output_defaults():
    actuator = ActuatorOutput()

    assert np.array_equal(
        actuator.force,
        np.zeros(3),
    )

    assert np.array_equal(
        actuator.torque,
        np.zeros(3),
    )


def test_actuator_output_arrays_are_independent():
    actuator_1 = ActuatorOutput()
    actuator_2 = ActuatorOutput()

    actuator_1.force[0] = 100.0
    actuator_1.torque[1] = 200.0

    assert actuator_2.force[0] == 0.0
    assert actuator_2.torque[1] == 0.0