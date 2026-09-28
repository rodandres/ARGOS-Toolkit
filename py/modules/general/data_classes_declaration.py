from pathlib import Path
import sys
from typing import TYPE_CHECKING

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
    
from dataclasses import dataclass, field
import numpy as np

if TYPE_CHECKING:
    from py.modules.sensors import SensorBase
    from py.modules.controllers.controller_base import ControllerBase
    from py.modules.actuators.actuators_base import ActuatorBase
    from py.modules.enviroments.perturbations import PerturbationBase

@dataclass
class SimSharedData:
    t: float
    tick: int
    
    actual_q: np.ndarray
    actual_omega: np.ndarray
    actual_alpha: np.ndarray
    
    sensors_measurements: dict
    
    reference_q: np.ndarray
    reference_omega: np.ndarray
    reference_alpha: np.ndarray
    
    estimated_q: np.ndarray
    estimated_omega: np.ndarray
    estimated_alpha: np.ndarray
    
    # ----- Legacy forms -----
    actual_state: np.ndarray
    measured_state: np.ndarray
    reference_state: np.ndarray
    estimated_state: np.ndarray
    # ------------------------
    
    tau_from_control: np.ndarray
    tau_to_apply: np.ndarray
    
    perturbation_tau: np.ndarray
    
@dataclass
class SimHistory:

    time: list
    actual_state: list
    measured_state: list
    estimated_state: list

    reference_state: list

    tau_control: list
    tau_applied: list
    perturbation_tau: list
    
@dataclass
class GeneralSimulationSettings:
    total_sim_time: float
    navigation_freq: float
    guidance_freq: float
    control_freq: float
    dynamics_freq: float
    
    q0: np.ndarray
    w0: np.ndarray
    
    # ----- Legacy form -----
    initial_state: np.ndarray
    # ------------------------
    
    t_nav: float = field(init=False)
    dt_guid: float = field(init=False)
    dt_control: float = field(init=False)
    dt_dynamics: float = field(init=False)

    def __post_init__(self):

        self.dt_nav = 1 / self.navigation_freq
        self.dt_guid = 1 / self.guidance_freq
        self.dt_control = 1 / self.control_freq
        self.dt_dynamics = 1 / self.dynamics_freq

# =================================================================
@dataclass
class DynamicsSettings:
    I: np.ndarray
    I_inv: np.ndarray
    
# =================================================================
@dataclass
class SimulationSettings:
    general_settings: GeneralSimulationSettings
    sensor: "SensorBase"
    controller: "ControllerBase"
    actuators: list["ActuatorBase"]
    perturbations: list["PerturbationBase"]
    dynamics_settings: DynamicsSettings