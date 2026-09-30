import numpy as np

from argos.core.simulation import Simulation
from argos.core.spacecraft import Spacecraft
from argos.core.mission_manager import MissionManager

from argos.general.dataclasses import MissionPhase

from argos.enviroments.environments import ClassicalEnvironment

from argos.propagators.native_propagator import (
    NativeTranslationalPropagator,
)

from argos.sensors.generic_sensor import AbsoluteSensor
from argos.navigation.basic_laws import IdealNavigation
from argos.guidance.basic_laws import ConstantReferenceGuidance
from argos.controllers.classic_controllers import PDController
from argos.controllers.basic_laws import MultiRCSAllocator
from argos.actuators.RCS import RCSThruster


def create_gnc_simulation(tmp_path, max_sim_time=1.0):
    return Simulation(
        max_sim_time=max_sim_time,
        environment=ClassicalEnvironment(),
        history_data_file_path=str(tmp_path / "history"),
        csv_folder_path=str(tmp_path / "csv"),
        auto_save_csv=False,
    )


def create_gnc_spacecraft(simulation):
    translational_propagator = NativeTranslationalPropagator(
        dynamics="NEWTON",
    )

    sensor = AbsoluteSensor(
        sample_rate_freq=10.0,
        name="absolute_sensor",
    )

    navigation = IdealNavigation(
        sensor_to_be_use=1,
    )

    guidance = ConstantReferenceGuidance(
        desired_position=np.array([10.0, 0.0, 0.0]),
        desired_velocity=np.zeros(3),
        desired_acceleration=np.zeros(3),
        desired_quaternion=np.array([0.0, 0.0, 0.0, 1.0]),
        desired_angular_velocity=np.zeros(3),
        desired_angular_acceleration=np.zeros(3),
    )

    # The controller accepts scalar gains or vectors of shape (3,).
    # Vectors are used here explicitly so that the control output also
    # has shape (3,).
    controller = PDController(
        Kp_rotational=np.ones(3),
        Kd_rotational=np.ones(3),
        Kp_translational=np.ones(3),
        Kd_translational=np.ones(3),
    )

    translational_thruster = RCSThruster(
        name="translational_thruster_x",
        nominal_thrust=1.0,
        direction=np.array([1.0, 0.0, 0.0]),
    )

    allocator = MultiRCSAllocator()

    allocator.set_actuators([translational_thruster])

    phase = MissionPhase(
        name="gnc",
        translational_model=translational_propagator,
        guidance=guidance,
        navigation=navigation,
        controller=controller,
        allocator=allocator,
        dt_nav=0.1,
        dt_guid=0.1,
        dt_control=0.1,
        dt_propagation=0.1,
    )

    mission_manager = MissionManager(phase)

    name = "GNC_Spacecraft"

    # Must exist before Spacecraft initialization because the mission
    # manager can record the initial transition.
    simulation.simulation_data.simulation_history.add_spacecraft_history(
        name
    )

    spacecraft = Spacecraft(
        name=name,
        mass=1.0,
        initial_state=np.array(
            [
                0.0, 0.0, 0.0,       # position
                0.0, 0.0, 0.0,       # velocity
                0.0, 0.0, 0.0, 1.0, # attitude
                0.0, 0.0, 0.0,       # angular velocity
            ]
        ),
        inertia_tensor=np.eye(3),
        actuators=[translational_thruster],
        sensors=[sensor],
        mission_manager=mission_manager,
        parent=simulation,
    )

    simulation.simulation_data.spacecrafts.append(spacecraft)

    return spacecraft


def test_gnc_pipeline_produces_guidance_output(tmp_path):
    simulation = create_gnc_simulation(tmp_path)
    spacecraft = create_gnc_spacecraft(simulation)

    simulation.simulate()

    guidance = spacecraft.spacecraft_data.guidance_data

    np.testing.assert_allclose(
        guidance.state.position,
        np.array([10.0, 0.0, 0.0]),
    )

    np.testing.assert_allclose(
        guidance.state.velocity,
        np.zeros(3),
    )


def test_gnc_pipeline_generates_control_command(tmp_path):
    simulation = create_gnc_simulation(tmp_path)
    spacecraft = create_gnc_spacecraft(simulation)

    simulation.simulate()

    control = spacecraft.spacecraft_data.control_data

    assert control.force.shape == (3,)
    assert control.torque.shape == (3,)
    assert np.linalg.norm(control.force) > 0.0


def test_gnc_pipeline_exerts_translational_force(tmp_path):
    simulation = create_gnc_simulation(tmp_path)
    spacecraft = create_gnc_spacecraft(simulation)

    simulation.simulate()

    force = spacecraft.spacecraft_data.current_force_exerted

    assert force.shape == (3,)
    assert np.linalg.norm(force) > 0.0


def test_gnc_pipeline_changes_spacecraft_velocity(tmp_path):
    simulation = create_gnc_simulation(
        tmp_path,
        max_sim_time=1.0,
    )

    spacecraft = create_gnc_spacecraft(simulation)

    initial_velocity = (
        spacecraft.spacecraft_data.true_state.velocity.copy()
    )

    simulation.simulate()

    final_velocity = (
        spacecraft.spacecraft_data.true_state.velocity.copy()
    )

    assert final_velocity[0] > initial_velocity[0]

def test_gnc_pipeline_navigation_tracks_spacecraft(tmp_path):
    """
    Navigation estimates remain consistent with the spacecraft state.
    """
    simulation = create_gnc_simulation(tmp_path)
    spacecraft = create_gnc_spacecraft(simulation)

    simulation.simulate()

    navigation = spacecraft.spacecraft_data.navigation_data
    true_state = spacecraft.spacecraft_data.true_state

    position_error = (
        navigation.spacecraft_state.position
        - true_state.position
    )

    velocity_error = (
        navigation.spacecraft_state.velocity
        - true_state.velocity
    )

    assert np.linalg.norm(position_error) < 0.2
    assert np.linalg.norm(velocity_error) < 0.2


def test_gnc_pipeline_produces_guidance_output(tmp_path):
    """
    The guidance system generates the configured reference state.
    """
    simulation = create_gnc_simulation(tmp_path)
    spacecraft = create_gnc_spacecraft(simulation)

    simulation.simulate()

    guidance = spacecraft.spacecraft_data.guidance_data

    np.testing.assert_allclose(
        guidance.state.position,
        np.array([10.0, 0.0, 0.0]),
    )

    np.testing.assert_allclose(
        guidance.state.velocity,
        np.zeros(3),
    )


def test_gnc_pipeline_generates_control_command(tmp_path):
    """
    Navigation and guidance information are converted into a control command.
    """
    simulation = create_gnc_simulation(tmp_path)
    spacecraft = create_gnc_spacecraft(simulation)

    simulation.simulate()

    control = spacecraft.spacecraft_data.control_data

    assert control.force.shape == (3,)
    assert control.torque.shape == (3,)

    assert np.linalg.norm(control.force) > 0.0


def test_gnc_pipeline_exerts_translational_force(tmp_path):
    """
    The allocator and actuator produce an actual force on the spacecraft.
    """
    simulation = create_gnc_simulation(tmp_path)
    spacecraft = create_gnc_spacecraft(simulation)

    simulation.simulate()

    force = spacecraft.spacecraft_data.current_force_exerted

    assert force.shape == (3,)
    assert np.linalg.norm(force) > 0.0


def test_gnc_pipeline_changes_spacecraft_velocity(tmp_path):
    """
    The complete GNC chain produces a physical translational effect.
    """
    simulation = create_gnc_simulation(
        tmp_path,
        max_sim_time=1.0,
    )

    spacecraft = create_gnc_spacecraft(simulation)

    initial_velocity = (
        spacecraft.spacecraft_data.true_state.velocity.copy()
    )

    simulation.simulate()

    final_velocity = (
        spacecraft.spacecraft_data.true_state.velocity.copy()
    )

    assert final_velocity[0] > initial_velocity[0]