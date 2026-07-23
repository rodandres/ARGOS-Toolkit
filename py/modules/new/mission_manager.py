from collections.abc import Callable
from dataclasses import dataclass

from py.modules.new.simulation import SimulationData


@dataclass
class MissionPhase:
    
    name: str
    controller: object
    guidance: object
    navigation: object

    dt_nav: float
    dt_guid: float
    dt_control: float


class MissionManager: # NOTE add method to print info about the transitions

    def __init__(self, initial_phase: MissionPhase, verbose: bool = False):
        self.phases = {initial_phase.name: initial_phase}
        self.current_phase = initial_phase
        self.verbose = verbose

        self.transitions = {}  # Dictionary to hold transitions between phases

    def add_phases(self, phase: MissionPhase | list[MissionPhase]):
        if isinstance(phase, list):
            for p in phase:                
                self.__add_phase(p)
        else:            
            self.__add_phase(phase)

    def __add_phase(self, phase: MissionPhase):
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

    def update(self, simulation_data: SimulationData):
        current_phase_name = self.current_phase.name
        if current_phase_name in self.transitions:
            for transition_name, transition_info in self.transitions[current_phase_name].items():
                if transition_info["condition"](simulation_data):
                    self.current_phase = transition_info["target_phase"]
                    if self.verbose:
                        print(f"Transitioned from '{current_phase_name}' to '{self.current_phase.name}' via '{transition_name}'.")
                    break  # Exit after the first valid transition

    def get_current_phase(self) -> MissionPhase:
        return self.current_phase