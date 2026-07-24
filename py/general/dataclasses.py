from dataclasses import dataclass, field

import numpy as np

from py.modules.controllers import ControllerBase
from py.modules.guidance import GuidanceBase
from py.modules.navigation import NavigationBase

@dataclass
class MissionPhase:
    
    name: str
    controller: ControllerBase
    guidance: GuidanceBase
    navigation: NavigationBase

    dt_nav: float
    dt_guid: float
    dt_control: float


@dataclass
class SpacecraftData:
    name: str

    mass: float
    inertia_tensor: np.ndarray

    # Spacraft state variables in inertial frame (same as the inertial frame of the simulation)
    true_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    true_vel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    true_accel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    true_q: np.ndarray = field(default_factory=lambda: np.array([1, 0, 0, 0]))
    true_omega: np.ndarray = field(default_factory=lambda: np.zeros(3))
    true_alpha: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # Spacecraft state variables in sensor frame (same as the sensor frame of the spacecraft)
    estimated_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    estimated_vel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    estimated_accel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    estimated_q: np.ndarray = field(default_factory=lambda: np.array([1, 0, 0, 0]))
    estimated_omega: np.ndarray = field(default_factory=lambda: np.zeros(3))
    estimated_alpha: np.ndarray = field(default_factory=lambda: np.zeros(3))

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

    # Reference state variables in sensor frame (same as the sensor frame of the spacecraft)
    estimated_ref_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    estimated_ref_vel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    estimated_ref_accel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    estimated_ref_q: np.ndarray = field(default_factory=lambda: np.array([1, 0, 0, 0]))
    estimated_ref_omega: np.ndarray = field(default_factory=lambda: np.zeros(3))
    estimated_ref_alpha: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # Guidance reference variables
    guidance_ref_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    guidance_ref_vel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    guidance_ref_accel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    guidance_ref_q: np.ndarray = field(default_factory=lambda: np.array([1, 0, 0, 0]))
    guidance_ref_omega: np.ndarray = field(default_factory=lambda: np.zeros(3))
    guidance_ref_alpha: np.ndarray = field(default_factory=lambda: np.zeros(3))


@dataclass
class estimationOutput:
    spacecraft_estimated_position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    spacecraft_estimated_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    spacecraft_estimated_acceleration: np.ndarray = field(default_factory=lambda: np.zeros(3))
    spacecraft_estimated_attitude: np.ndarray = field(default_factory=lambda: np.array([1, 0, 0, 0]))
    spacecraft_estimated_angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    spacecraft_estimated_angular_acceleration: np.ndarray = field(default_factory=lambda: np.zeros(3))

    reference_estimated_position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    reference_estimated_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    reference_estimated_acceleration: np.ndarray = field(default_factory=lambda: np.zeros(3))
    reference_estimated_attitude: np.ndarray = field(default_factory=lambda: np.array([1, 0, 0, 0]))
    reference_estimated_angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    reference_estimated_angular_acceleration: np.ndarray = field(default_factory=lambda: np.zeros(3))