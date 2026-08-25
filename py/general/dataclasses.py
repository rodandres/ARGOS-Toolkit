from dataclasses import dataclass, field

import numpy as np

from typing import TYPE_CHECKING

from py.general.data_save import SimulationHistory

if TYPE_CHECKING:
    from py.modules.controllers.controller_base import ControllerBase, ControlAllocatorBase
    from py.modules.guidance.guidance_base import GuidanceBase
    from py.modules.navigation.navigation_base import NavigationBase
    from py.modules.propagators.propagator_base import TranslationalPropagatorBase, RotationalPropagatorBase

@dataclass(slots=True)
class MissionPhase:
    
    name: str

    rotational_model: RotationalPropagatorBase | None = None
    translational_model: TranslationalPropagatorBase | None = None

    guidance: GuidanceBase | None = None
    navigation: NavigationBase | None = None
    controller: ControllerBase | None = None

    allocator: ControlAllocatorBase | None = None

    dt_nav: float | None = None
    dt_guid: float | None = None
    dt_control: float | None = None

    dt_propagation: float | None = None  # Optional propagation time step for this phase

@dataclass(slots=True)
class SimulationData:
    max_sim_time: float

    spacecrafts: list

    dt_master: float = field(default=0.1)

    t: float =  field(default=0.0)
    tick: int = field(default=0)
    verbose: bool = False

    simulation_history: "SimulationHistory" = field(default_factory=lambda: SimulationHistory())

    DCM_inertial_to_body: np.ndarray = field(default_factory=lambda: np.eye(3))

@dataclass(slots=True)
class StateVariables:
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    acceleration: np.ndarray = field(default_factory=lambda: np.zeros(3))
    attitude: np.ndarray = field(default_factory=lambda: np.array([0, 0, 0, 1]))  # Quaternion
    angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    angular_acceleration: np.ndarray = field(default_factory=lambda: np.zeros(3))

    frame_type: None = None  # 'inertial' or 'body', to be defined later

@dataclass(slots=True)
class EstimationOutput:
    spacecraft_state: StateVariables = field(default_factory=StateVariables)
    reference_state: StateVariables = field(default_factory=StateVariables)

@dataclass(slots=True)
class GuidanceReference:
    state: StateVariables = field(default_factory=StateVariables)    

@dataclass(slots=True)
class ControlOutput:
    force: np.ndarray = field(default_factory=lambda: np.zeros(3))
    torque: np.ndarray = field(default_factory=lambda: np.zeros(3))

@dataclass(slots=True)
class SpacecraftData():

    t: float = 0.0
    tick: int = 0
    current_master_dt: float = 0.1  # Time step for the current master simulation

    # Time step for the current propagation
    current_propagation_dt: float = 0.1

    # Spacraft state variables in inertial frame (same as the inertial frame of the simulation)
    true_state: StateVariables = field(default_factory=StateVariables)    
    
    estimated_data: EstimationOutput = field(default_factory=EstimationOutput)
    reference_data: GuidanceReference = field(default_factory=GuidanceReference)
    control_output_data: ControlOutput = field(default_factory=ControlOutput)

    current_force_exerted: np.ndarray = field(default_factory=lambda: np.zeros(3))
    current_torque_exerted: np.ndarray = field(default_factory=lambda: np.zeros(3))

    target_name: str | None = None  # Name of the reference spacecraft, if any

@dataclass(slots=True)
class ActuatorOutput:
    force: np.ndarray = field(default_factory=lambda: np.zeros(3))
    torque: np.ndarray = field(default_factory=lambda: np.zeros(3))