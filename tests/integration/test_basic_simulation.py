import numpy as np
import pytest

from argos.core.simulation import Simulation
from argos.core.spacecraft import Spacecraft
from argos.general.dataclasses import MissionPhase

from argos.core.mission_manager import MissionManager
from argos.propagators.native_propagator import (
    NativeTranslationalPropagator,
    NativeRotationalPropagator
)

from argos.enviroments.environments import ClassicalEnvironment


def create_spacecraft(
    simulation,
    name="SC1",
    initial_position=None,
    initial_velocity=None,
    initial_attitude=None,
    initial_angular_velocity=None,
    mass=1.0,
    inertia_tensor=None,
    translational_model=None,
    rotational_model=None,
    dt_propagation=1.0,
):
    """Create a real spacecraft and register it in a real simulation."""

    if initial_position is None:
        initial_position = np.zeros(3)

    if initial_velocity is None:
        initial_velocity = np.zeros(3)

    if initial_attitude is None:
        initial_attitude = np.array([0.0, 0.0, 0.0, 1.0])

    if initial_angular_velocity is None:
        initial_angular_velocity = np.zeros(3)

    if inertia_tensor is None:
        inertia_tensor = np.eye(3)

    initial_state = np.concatenate(
        (
            initial_position,
            initial_velocity,
            initial_attitude,
            initial_angular_velocity,
        )
    )

    phase = MissionPhase(
        name="initial",
        translational_model=translational_model,
        rotational_model=rotational_model,
        dt_propagation=dt_propagation,
    )

    mission_manager = MissionManager(phase)

    # Spacecraft initialization can record the initial mission transition,
    # so its history entry must exist before creating the spacecraft.
    simulation.simulation_data.simulation_history.add_spacecraft_history(name)

    spacecraft = Spacecraft(
        name=name,
        mass=mass,
        initial_state=initial_state,
        inertia_tensor=inertia_tensor,
        mission_manager=mission_manager,
        parent=simulation,
        verbose=False,
    )

    simulation.simulation_data.spacecrafts.append(spacecraft)

    return spacecraft


def test_basic_translational_simulation():
    """
    Verify that a spacecraft with a real translational propagator
    advances according to its initial velocity.
    """

    environment = ClassicalEnvironment()

    simulation = Simulation(
        max_sim_time=2.0,
        environment=environment,
        auto_save_csv=False,
        verbose=False,
    )

    spacecraft = create_spacecraft(
        simulation,
        initial_position=np.zeros(3),
        initial_velocity=np.array([1.0, 0.0, 0.0]),
        translational_model=NativeTranslationalPropagator(
            dynamics="NEWTON"
        ),
        dt_propagation=1.0,
    )

    metadata = simulation.simulate()

    state = spacecraft.spacecraft_data.true_state

    np.testing.assert_allclose(
        state.position,
        np.array([2.0, 0.0, 0.0]),
    )

    np.testing.assert_allclose(
        state.velocity,
        np.array([1.0, 0.0, 0.0]),
    )

    np.testing.assert_allclose(
        state.acceleration,
        np.zeros(3),
    )

    assert spacecraft.spacecraft_data.t == pytest.approx(2.0)
    assert spacecraft.spacecraft_data.tick == 2

    assert simulation.simulation_data.dt_master == pytest.approx(1.0)

    assert metadata is not None


def test_basic_rotational_simulation():
    """
    Verify that a spacecraft with a real rotational propagator
    can propagate its rotational state.
    """

    environment = ClassicalEnvironment()

    simulation = Simulation(
        max_sim_time=2.0,
        environment=environment,
        auto_save_csv=False,
        verbose=False,
    )

    initial_attitude = np.array([0.0, 0.0, 0.0, 1.0])
    initial_angular_velocity = np.zeros(3)

    spacecraft = create_spacecraft(
        simulation,
        initial_attitude=initial_attitude,
        initial_angular_velocity=initial_angular_velocity,
        rotational_model=NativeRotationalPropagator(),
        dt_propagation=1.0,
    )

    metadata = simulation.simulate()

    state = spacecraft.spacecraft_data.true_state

    np.testing.assert_allclose(
        state.attitude,
        initial_attitude,
    )

    np.testing.assert_allclose(
        state.angular_velocity,
        initial_angular_velocity,
    )

    np.testing.assert_allclose(
        state.angular_acceleration,
        np.zeros(3),
    )

    assert spacecraft.spacecraft_data.t == pytest.approx(2.0)
    assert spacecraft.spacecraft_data.tick == 2

    assert simulation.simulation_data.dt_master == pytest.approx(1.0)

    assert metadata is not None


def test_translational_and_rotational_simulation():
    """
    Verify that translational and rotational propagation can run
    simultaneously within the same mission phase.
    """

    environment = ClassicalEnvironment()

    simulation = Simulation(
        max_sim_time=2.0,
        environment=environment,
        auto_save_csv=False,
        verbose=False,
    )

    spacecraft = create_spacecraft(
        simulation,
        initial_position=np.zeros(3),
        initial_velocity=np.array([1.0, 0.0, 0.0]),
        initial_attitude=np.array([0.0, 0.0, 0.0, 1.0]),
        initial_angular_velocity=np.zeros(3),
        translational_model=NativeTranslationalPropagator(
            dynamics="NEWTON"
        ),
        rotational_model=NativeRotationalPropagator(),
        dt_propagation=1.0,
    )

    simulation.simulate()

    state = spacecraft.spacecraft_data.true_state

    np.testing.assert_allclose(
        state.position,
        np.array([2.0, 0.0, 0.0]),
    )

    np.testing.assert_allclose(
        state.velocity,
        np.array([1.0, 0.0, 0.0]),
    )

    np.testing.assert_allclose(
        state.attitude,
        np.array([0.0, 0.0, 0.0, 1.0]),
    )

    np.testing.assert_allclose(
        state.angular_velocity,
        np.zeros(3),
    )

    assert spacecraft.spacecraft_data.t == pytest.approx(2.0)
    assert spacecraft.spacecraft_data.tick == 2


def test_simulation_dt_master_is_taken_from_spacecraft():
    """
    Verify that the simulation master timestep is determined from
    the spacecraft execution rates.
    """

    environment = ClassicalEnvironment()

    simulation = Simulation(
        max_sim_time=2.0,
        environment=environment,
        auto_save_csv=False,
        verbose=False,
    )

    phase = MissionPhase(
        name="initial",
        translational_model=NativeTranslationalPropagator(
            dynamics="NEWTON"
        ),
        dt_propagation=2.0,
    )

    mission_manager = MissionManager(phase)

    simulation.simulation_data.simulation_history.add_spacecraft_history("SC1")

    spacecraft = Spacecraft(
        name="SC1",
        mass=1.0,
        initial_state=np.concatenate(
            (
                np.zeros(3),
                np.array([1.0, 0.0, 0.0]),
                np.array([0.0, 0.0, 0.0, 1.0]),
                np.zeros(3),
            )
        ),
        inertia_tensor=np.eye(3),
        mission_manager=mission_manager,
        parent=simulation,
        verbose=False,
    )

    simulation.simulation_data.spacecrafts.append(spacecraft)

    simulation.simulate()

    assert simulation.simulation_data.dt_master == pytest.approx(2.0)
    assert spacecraft.spacecraft_data.current_master_dt == pytest.approx(2.0)


def test_simulation_respects_max_sim_time():
    """
    Verify that propagation does not advance beyond max_sim_time.
    """

    environment = ClassicalEnvironment()

    simulation = Simulation(
        max_sim_time=1.5,
        environment=environment,
        auto_save_csv=False,
        verbose=False,
    )

    spacecraft = create_spacecraft(
        simulation,
        initial_position=np.zeros(3),
        initial_velocity=np.array([1.0, 0.0, 0.0]),
        translational_model=NativeTranslationalPropagator(
            dynamics="NEWTON"
        ),
        dt_propagation=1.0,
    )

    simulation.simulate()

    state = spacecraft.spacecraft_data.true_state

    # First propagation: 0 -> 1
    # Second propagation: 1 -> 1.5
    np.testing.assert_allclose(
        state.position,
        np.array([1.5, 0.0, 0.0]),
    )

    np.testing.assert_allclose(
        state.velocity,
        np.array([1.0, 0.0, 0.0]),
    )

    assert spacecraft.spacecraft_data.t == pytest.approx(2.0)
    assert spacecraft.spacecraft_data.tick == 2


def test_multiple_spacecrafts_simulate_independently():
    """
    Verify that multiple real spacecraft can participate in the same
    simulation and maintain independent states.
    """

    environment = ClassicalEnvironment()

    simulation = Simulation(
        max_sim_time=2.0,
        environment=environment,
        auto_save_csv=False,
        verbose=False,
    )

    spacecraft_1 = create_spacecraft(
        simulation,
        name="SC1",
        initial_position=np.zeros(3),
        initial_velocity=np.array([1.0, 0.0, 0.0]),
        translational_model=NativeTranslationalPropagator(
            dynamics="NEWTON"
        ),
        dt_propagation=1.0,
    )

    spacecraft_2 = create_spacecraft(
        simulation,
        name="SC2",
        initial_position=np.zeros(3),
        initial_velocity=np.array([0.0, 2.0, 0.0]),
        translational_model=NativeTranslationalPropagator(
            dynamics="NEWTON"
        ),
        dt_propagation=1.0,
    )

    simulation.simulate()

    np.testing.assert_allclose(
        spacecraft_1.spacecraft_data.true_state.position,
        np.array([2.0, 0.0, 0.0]),
    )

    np.testing.assert_allclose(
        spacecraft_2.spacecraft_data.true_state.position,
        np.array([0.0, 4.0, 0.0]),
    )

    np.testing.assert_allclose(
        spacecraft_1.spacecraft_data.true_state.velocity,
        np.array([1.0, 0.0, 0.0]),
    )

    np.testing.assert_allclose(
        spacecraft_2.spacecraft_data.true_state.velocity,
        np.array([0.0, 2.0, 0.0]),
    )

    assert spacecraft_1.spacecraft_data.t == pytest.approx(2.0)
    assert spacecraft_2.spacecraft_data.t == pytest.approx(2.0)


def test_simulation_without_spacecraft_raises_error():
    """
    Verify that a simulation cannot be executed without spacecraft.
    """

    simulation = Simulation(
        max_sim_time=1.0,
        environment=ClassicalEnvironment(),
        auto_save_csv=False,
        verbose=False,
    )

    with pytest.raises(
        ValueError,
        match="No spacecrafts have been added to the simulation",
    ):
        simulation.simulate()