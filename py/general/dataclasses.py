from dataclasses import dataclass, field

import numpy as np

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from py.modules.controllers.controller_base import ControllerBase, ControlAllocatorBase
    from py.modules.guidance.guidance_base import GuidanceBase
    from py.modules.navigation.navigation_base import NavigationBase

@dataclass
class MissionPhase:
    
    name: str
    guidance: GuidanceBase | None = None
    navigation: NavigationBase | None = None
    controller: ControllerBase | None = None

    allocator: ControlAllocatorBase | None = None

    dt_nav: float | None = None
    dt_guid: float | None = None
    dt_control: float | None = None

@dataclass
class SimulationData:
    max_sim_time: float

    dt_master: float
    dt_propagation: float
    
    spacecrafts: list

    t = 0.0
    tick = 0
    verbose: bool = False

    DCM_inertial_to_body: np.ndarray = field(default_factory=lambda: np.eye(3))

@dataclass
class EstimationOutput:
    spacecraft_position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    spacecraft_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    spacecraft_acceleration: np.ndarray = field(default_factory=lambda: np.zeros(3))
    spacecraft_attitude: np.ndarray = field(default_factory=lambda: np.array([1, 0, 0, 0]))
    spacecraft_angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    spacecraft_angular_acceleration: np.ndarray = field(default_factory=lambda: np.zeros(3))

    reference_position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    reference_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    reference_acceleration: np.ndarray = field(default_factory=lambda: np.zeros(3))
    reference_attitude: np.ndarray = field(default_factory=lambda: np.array([1, 0, 0, 0]))
    reference_angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    reference_angular_acceleration: np.ndarray = field(default_factory=lambda: np.zeros(3))

@dataclass
class GuidanceReference:
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    acceleration: np.ndarray = field(default_factory=lambda: np.zeros(3))
    attitude: np.ndarray = field(default_factory=lambda: np.array([1, 0, 0, 0]))
    angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    angular_acceleration: np.ndarray = field(default_factory=lambda: np.zeros(3))

@dataclass
class ControlOutput:
    force: np.ndarray = field(default_factory=lambda: np.zeros(3))
    torque: np.ndarray = field(default_factory=lambda: np.zeros(3))

@dataclass
class SpacecraftData:

    # Spacraft state variables in inertial frame (same as the inertial frame of the simulation)
    true_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    true_vel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    true_accel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    true_q: np.ndarray = field(default_factory=lambda: np.array([1, 0, 0, 0]))
    true_omega: np.ndarray = field(default_factory=lambda: np.zeros(3))
    true_alpha: np.ndarray = field(default_factory=lambda: np.zeros(3))
    
    # Spacraft state variables in body frame (same as the body frame of the spacecraft) # NOTE NOT YET USED OR DEFINED
    body_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    body_vel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    body_accel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    body_q: np.ndarray = field(default_factory=lambda: np.array([1, 0, 0, 0]))
    body_omega: np.ndarray = field(default_factory=lambda: np.zeros(3))
    body_alpha: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # Reference state variables in inertial frame (same as the inertial frame of the simulation)
    true_ref_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    true_ref_vel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    true_ref_accel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    true_ref_q: np.ndarray = field(default_factory=lambda: np.array([1, 0, 0, 0]))
    true_ref_omega: np.ndarray = field(default_factory=lambda: np.zeros(3))
    true_ref_alpha: np.ndarray = field(default_factory=lambda: np.zeros(3))

    navigation_estimated_data: EstimationOutput = field(default_factory=EstimationOutput)
    guidance_reference_data: GuidanceReference = field(default_factory=GuidanceReference)
    control_output_data: ControlOutput = field(default_factory=ControlOutput)

    current_force_exerted: np.ndarray = field(default_factory=lambda: np.zeros(3))
    current_torque_exerted: np.ndarray = field(default_factory=lambda: np.zeros(3))

    target_name: str | None = None  # Name of the reference spacecraft, if any



@dataclass
class ActuatorOutput:
    force: np.ndarray = field(default_factory=lambda: np.zeros(3))
    torque: np.ndarray = field(default_factory=lambda: np.zeros(3))