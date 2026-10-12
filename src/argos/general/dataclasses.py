from dataclasses import dataclass, field

import numpy as np

from typing import TYPE_CHECKING

from argos.general.data_save import SimulationHistory

if TYPE_CHECKING:
    from argos.controllers.controller_base import ControllerBase, ControlAllocatorBase
    from argos.guidance.guidance_base import GuidanceBase
    from argos.navigation.navigation_base import NavigationBase
    from argos.propagators.propagator_base import TranslationalPropagatorBase, RotationalPropagatorBase
    from argos.frames.frames_base import Frame

from argos.frames.frames import BODY_frame, NONE_frame

@dataclass(slots=True)
class MissionPhase:
    """
    Define the configuration of a spacecraft mission phase.

    A mission phase groups the propagation models, GNC components, and
    execution time steps used while the spacecraft is in that phase.

    Attributes
    ----------
    name : str
        Mission phase identifier.
    rotational_model : RotationalPropagatorBase, optional
        Rotational dynamics propagator used during the phase.
    translational_model : TranslationalPropagatorBase, optional
        Translational dynamics propagator used during the phase.
    guidance : GuidanceBase, optional
        Guidance algorithm used during the phase.
    navigation : NavigationBase, optional
        Navigation algorithm used during the phase.
    controller : ControllerBase, optional
        Controller used during the phase.
    allocator : ControlAllocatorBase, optional
        Control allocator used during the phase.
    dt_nav : float, optional
        Navigation update period [s].
    dt_guid : float, optional
        Guidance update period [s].
    dt_control : float, optional
        Control update period [s].
    dt_propagation : float, optional
        Propagation time step used during the phase [s].
    """
    
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
    """
    Store global data and configuration for a spacecraft simulation.

    Attributes
    ----------
    max_sim_time : float
        Maximum simulation time [s].
    spacecrafts : list
        Spacecraft instances participating in the simulation.
    dt_master : float
        Master simulation time step [s].
    t : float
        Current simulation time [s].
    tick : int
        Current simulation tick.
    verbose : bool
        If True, verbose simulation information is enabled.
    simulation_history : SimulationHistory, optional
        Simulation history manager used to record simulation data.
    """
    max_sim_time: float

    spacecrafts: list

    dt_master: float = field(default=0.1)

    t: float =  field(default=0.0)
    tick: int = field(default=0)
    verbose: bool = False
    
    simulation_history: SimulationHistory = None

@dataclass(slots=True)
class StateVariables:
    """
    Store translational and rotational spacecraft state variables.

    Translational and rotational states may be associated with different
    reference frames. Each frame specifies the reference used for the
    corresponding state variables.

    Attributes
    ----------
    position : np.ndarray, shape (3,)
        Position vector expressed in `translational_frame` [m].
    velocity : np.ndarray, shape (3,)
        Velocity vector expressed in `translational_frame` [m/s].
    acceleration : np.ndarray, shape (3,)
        Acceleration vector expressed in `translational_frame` [m/s²].

    attitude : np.ndarray, shape (4,)
        Quaternion representing the orientation of the spacecraft BODY frame
        with respect to `rotational_frame`. The quaternion convention,
        including component ordering and rotation direction, must be defined
        consistently throughout the simulation.

    angular_velocity : np.ndarray, shape (3,)
        Angular velocity vector expressed in `rotational_frame` [rad/s].
        The reference frame with respect to which the angular velocity is
        measured must be defined by the rotational dynamics convention.

    angular_acceleration : np.ndarray, shape (3,)
        Angular acceleration vector expressed in `rotational_frame` [rad/s²].
        Its interpretation must be consistent with the rotational dynamics
        convention.

    translational_frame : Frame or None
        Reference frame associated with the translational state variables
        (`position`, `velocity`, and `acceleration`). May be `None` if
        translational propagation is not used.

    rotational_frame : Frame or None
        Reference frame associated with the rotational state variables
        (`attitude`, `angular_velocity`, and `angular_acceleration`).
        Must be specified when rotational propagation is active.
    """

    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    acceleration: np.ndarray = field(default_factory=lambda: np.zeros(3))
    attitude: np.ndarray = field(default_factory=lambda: np.array([0, 0, 0, 1]))  # Quaternion
    angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    angular_acceleration: np.ndarray = field(default_factory=lambda: np.zeros(3))

    translational_frame: Frame = field(default=NONE_frame)
    rotational_frame: Frame = field(default=NONE_frame)


@dataclass(slots=True)
class NavigationOutput:
    """
    Store the navigation estimates for the spacecraft and its target.

    Attributes
    ----------
    spacecraft_state : StateVariables
        Estimated state of the spacecraft.
    target_state : StateVariables
        Estimated state of the target spacecraft.
    """
    spacecraft_state: StateVariables = field(default_factory=StateVariables)
    target_state: StateVariables = field(default_factory=StateVariables)
    frame: Frame = field(default=NONE_frame)

@dataclass(slots=True)
class GuidanceOutput:
    """
    Store the desired spacecraft state generated by the guidance system.

    Attributes
    ----------
    state : StateVariables
        Desired spacecraft state.
    """
    state: StateVariables = field(default_factory=StateVariables)    
    frame: Frame = field(default=NONE_frame)

@dataclass(slots=True)
class ControlOutput:
    """
    Store the force and torque commands generated by the controller.

    Attributes
    ----------
    force : np.ndarray, shape (3,)
        Commanded force vector [N].
    torque : np.ndarray, shape (3,)
        Commanded torque vector [N·m].
    """
    force: np.ndarray = field(default_factory=lambda: np.zeros(3))
    torque: np.ndarray = field(default_factory=lambda: np.zeros(3))
    frame: Frame = field(default=BODY_frame)

@dataclass(slots=True)
class ActuatorOutput:
    """
    Store the force and torque generated by an actuator.

    Attributes
    ----------
    force : np.ndarray, shape (3,)
        Force generated by the actuator [N].
    torque : np.ndarray, shape (3,)
        Torque generated by the actuator [N·m].
    """
    force: np.ndarray = field(default_factory=lambda: np.zeros(3))
    torque: np.ndarray = field(default_factory=lambda: np.zeros(3))
    frame: Frame = field(default=BODY_frame)  # Frame in which the actuator output is expressed

@dataclass(slots=True)
class PerturbationData:
    """
    Store the perturbation forces and torques acting ocurrent_force_exertedn a spacecraft.

    Attributes
    ----------
    force : np.ndarray, shape (3,)
        Perturbation force acting on the spacecraft [N].
    torque : np.ndarray, shape (3,)
        Perturbation torque acting on the spacecraft [N·m].
    """
    force: np.ndarray = field(default_factory=lambda: np.zeros(3))
    torque: np.ndarray = field(default_factory=lambda: np.zeros(3))
    frame: Frame = field(default=NONE_frame)  # Frame in which the perturbation data is expressed

@dataclass(slots=True)
class ExertedData:
    """
    Store the force and torque currently exerted on a spacecraft.

    Attributes
    ----------
    force : np.ndarray, shape (3,)
        Force currently exerted on the spacecraft [N].
    torque : np.ndarray, shape (3,)
        Torque currently exerted on the spacecraft [N·m].
    """
    force: np.ndarray = field(default_factory=lambda: np.zeros(3))
    torque: np.ndarray = field(default_factory=lambda: np.zeros(3))
    frame: Frame = field(default=BODY_frame)  # Frame in which the exerted data is expressed

@dataclass(slots=True)
class SpacecraftData:
    """
    Store the complete simulation data associated with a spacecraft.

    The data includes simulation timing, true spacecraft state, navigation
    estimates, guidance output, control commands, exerted force and torque,
    and the optional target spacecraft.

    Attributes
    ----------
    t : float
        Current simulation time [s].
    tick : int
        Current simulation tick.
    current_master_dt : float
        Current master simulation time step [s].
    current_propagation_dt : float
        Current propagation time step [s].
    true_state : StateVariables
        True spacecraft state in the simulation inertial frame.
    navigation_data : NavigationOutput
        Navigation estimates for the spacecraft and its target.
    guidance_data : GuidanceOutput
        Current guidance output.
    control_data : ControlOutput
        Current control command.
    exerted_data : ExertedData
        Current force and torque exerted on the spacecraft.
    target_name : str, optional
        Name of the reference spacecraft, if one is assigned.
    """
    t: float = 0.0
    tick: int = 0
    current_master_dt: float = 0.1  # Time step for the current master simulation

    # Time step for the current propagation
    current_propagation_dt: float = 0.1

    # Spacraft state variables in inertial frame (same as the inertial frame of the simulation)
    true_state: StateVariables = field(default_factory=StateVariables)    
    
    navigation_data: NavigationOutput = field(default_factory=NavigationOutput)
    guidance_data: GuidanceOutput = field(default_factory=GuidanceOutput)
    control_data: ControlOutput = field(default_factory=ControlOutput)

    # Current force and torque exerted on the spacecraft
    #current_force_exerted: np.ndarray = field(default_factory=lambda: np.zeros(3))
    #current_torque_exerted: np.ndarray = field(default_factory=lambda: np.zeros(3))
    exerted_data: ExertedData = field(default_factory=ExertedData)

    perturbation_data: PerturbationData = field(default_factory=PerturbationData)

    target_name: str | None = None  # Name of the reference spacecraft, if any