from collections.abc import Callable
from dataclasses import dataclass

from typing import TYPE_CHECKING
from py.general.dataclasses import MissionPhase

if TYPE_CHECKING:
    from py.modules.core.simulation import SimulationData


class MissionManager: # NOTE add method to print info about the transitions

    def __init__(self, initial_phase: MissionPhase, verbose: bool = False):
        self.phases = {initial_phase.name: initial_phase}
        self.current_phase = initial_phase
        self.verbose = verbose

        self.transitions = {}  # Dictionary to hold transitions between phases

    def _check_phase(self, phase: MissionPhase):
        if not isinstance(phase, MissionPhase):
            raise TypeError("phase must be an instance of MissionPhase.")

        if phase.name in self.phases:
            raise ValueError(f"Phase '{phase.name}' already exists.")

        guidance_law = phase.guidance
        navigation_law = phase.navigation
        control_law = phase.controller

        dt_nav = phase.dt_nav
        dt_guid = phase.dt_guid
        dt_control = phase.dt_control

        translational_model = phase.translational_model
        rotational_model = phase.rotational_model

        if guidance_law is not None and dt_guid is None:
            raise ValueError("dt_guid must be provided if guidance law is specified.")

        if navigation_law is not None and dt_nav is None:
            raise ValueError("dt_nav must be provided if navigation law is specified.")

        if control_law is not None and dt_control is None:
            raise ValueError("dt_control must be provided if control law is specified.")

        if dt_nav is not None and navigation_law is None:
            raise ValueError("Navigation law must be specified if dt_nav is provided.")

        if dt_guid is not None and guidance_law is None:
            raise ValueError("Guidance law must be specified if dt_guid is provided.")

        if dt_control is not None and control_law is None:
            raise ValueError("Control law must be specified if dt_control is provided.")

        if translational_model is None and rotational_model is None:
            raise ValueError("At least one of translational_model or rotational_model must be specified for the phase.")

        if (
            (translational_model is not None or rotational_model is not None)
            and phase.dt_propagation is None
        ):
            raise ValueError(
                "dt_propagation must be provided if either translational_model "
                "or rotational_model is specified."
            )
        

    def add_phases(self, phase: MissionPhase | list[MissionPhase]):
        if isinstance(phase, list):
            for p in phase:                
                self.__add_phase(p)
        else:            
            self.__add_phase(phase)

    def __add_phase(self, phase: MissionPhase):
        self._check_phase(phase)
        if phase.name in self.phases:
            raise ValueError(f"Phase '{phase.name}' already exists.")
        self.phases[phase.name] = phase

    def add_transition(self, from_phase: MissionPhase, target_phase: MissionPhase, condition: Callable[[SimulationData], bool], transition_name: str | None = None):
        if from_phase.name not in self.phases:
            raise ValueError(f"Initial phase '{from_phase.name}' does not exist.")

        if target_phase.name not in self.phases:
            raise ValueError(f"Target phase '{target_phase.name}' does not exist.")
        
        if transition_name is None:
            transition_name = f"{from_phase.name}_to_{target_phase.name}"

        phase_transitions = self.transitions.setdefault(from_phase.name, {})

        if transition_name in phase_transitions:
            raise ValueError(
                f"Transition '{transition_name}' already exists for phase '{from_phase.name}'."
            )

        if not callable(condition):
            raise TypeError(
                "condition must be callable."
            )

        phase_transitions[transition_name] = {
            "target_phase": target_phase,
            "condition": condition,
        }

    def update(self, simulation_data):

        current_phase = self.current_phase.name

        if current_phase not in self.transitions:
            return False

        for transition_name, transition in self.transitions[current_phase].items():

            if transition["condition"](simulation_data):

                self.current_phase = transition["target_phase"]

                if self.verbose:
                    print(
                        f"Transitioned from '{current_phase}' "
                        f"to '{self.current_phase.name}' "
                        f"via '{transition_name}'."
                    )

                return True

        return False