import numpy as np

from argos.core.simulation import Simulation
from argos.core.spacecraft import Spacecraft
from argos.core.mission_manager import MissionManager
from argos.general.dataclasses import MissionPhase
from argos.enviroments.environments import ClassicalEnvironment
from argos.propagators.native_propagator import (
    NativeTranslationalPropagator,
    NativeRotationalPropagator,
)
from argos.guidance.basic_laws import ConstantReferenceGuidance


def create_mission_simulation(tmp_path):
    simulation = Simulation(
        max_sim_time=0.5,
        environment=ClassicalEnvironment(),
        history_data_file_path=str(tmp_path),
        auto_save_csv=False,
    )

    translational_model = NativeTranslationalPropagator(
        dynamics="NEWTON",
        integration_method="NATIVE_RK45",
    )

    rotational_model = NativeRotationalPropagator(
        integration_method="NATIVE_RK45",
    )

    guidance_1 = ConstantReferenceGuidance(
        desired_position=np.array([10.0, 0.0, 0.0]),
        desired_velocity=np.zeros(3),
        desired_acceleration=np.zeros(3),
        desired_attitude=np.array([0.0, 0.0, 0.0, 1.0]),
        desired_angular_velocity=np.zeros(3),
        desired_angular_acceleration=np.zeros(3),
    )

    guidance_2 = ConstantReferenceGuidance(
        desired_position=np.array([20.0, 0.0, 0.0]),
        desired_velocity=np.zeros(3),
        desired_acceleration=np.zeros(3),
        desired_attitude=np.array([0.0, 0.0, 0.0, 1.0]),
        desired_angular_velocity=np.zeros(3),
        desired_angular_acceleration=np.zeros(3),
    )

    phase_1 = MissionPhase(
        name="phase_1",
        translational_model=translational_model,
        rotational_model=rotational_model,
        guidance=guidance_1,
        dt_guid=0.1,
        dt_propagation=0.1,
    )

    phase_2 = MissionPhase(
        name="phase_2",
        translational_model=translational_model,
        rotational_model=rotational_model,
        guidance=guidance_2,
        dt_guid=0.2,
        dt_propagation=0.2,
    )

    mission_manager = MissionManager(phase_1)
    mission_manager.add_phases(phase_2)

    mission_manager.add_transition(
        from_phase=phase_1,
        target_phase=phase_2,
        condition=lambda data: (
            data.spacecrafts[0].spacecraft_data.t >= 0.2
        ),
    )

    simulation.simulation_data.simulation_history.add_spacecraft_history(
        "TestSpacecraft"
    )

    spacecraft = Spacecraft(
        name="TestSpacecraft",
        mass=1.0,
        initial_state=np.array([
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
            0.0, 0.0, 0.0,
        ]),
        inertia_tensor=np.eye(3),
        mission_manager=mission_manager,
        parent=simulation,
    )

    simulation.simulation_data.spacecrafts.append(spacecraft)

    return simulation, spacecraft, phase_1, phase_2