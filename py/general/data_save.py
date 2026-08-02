import copy
from dataclasses import is_dataclass
import numpy as np


def _stack(obj_list):
    """
    Convierte una lista de dataclasses en un diccionario de numpy arrays
    de forma recursiva.
    """

    first = obj_list[0]

    if is_dataclass(first):
        return {
            field: _stack([getattr(obj, field) for obj in obj_list])
            for field in first.__dataclass_fields__
        }

    if isinstance(first, np.ndarray):
        return np.asarray(obj_list)

    return obj_list


class SpacecraftHistory:

    def __init__(self):

        self.t = []

        self.tick = []

        self.true_state = []

        self.estimated_data = []
        self.reference_data = []
        self.control_output_data = []

        self.current_force_exerted = []
        self.current_torque_exerted = []

    def record(self, data):

        self.t.append(data.t)
        self.tick.append(data.tick)

        self.true_state.append(copy.deepcopy(data.true_state))

        self.estimated_data.append(copy.deepcopy(data.estimated_data))
        self.reference_data.append(copy.deepcopy(data.reference_data))
        self.control_output_data.append(copy.deepcopy(data.control_output_data))

        self.current_force_exerted.append(data.current_force_exerted.copy())
        self.current_torque_exerted.append(data.current_torque_exerted.copy())

    def finalize(self):

        self.t = np.asarray(self.t)
        self.tick = np.asarray(self.tick)

        self.true_state = _stack(self.true_state)

        self.estimated_data = _stack(self.estimated_data)

        self.reference_data = _stack(self.reference_data)

        self.control_output_data = _stack(self.control_output_data)

        self.current_force_exerted = np.asarray(self.current_force_exerted)

        self.current_torque_exerted = np.asarray(self.current_torque_exerted)


class SimulationHistory:

    def __init__(self):

        self.time = []
        self.spacecrafts_history = {}    

    def record(self, simulation_data):

        self.time.append(simulation_data.t)

        for spacecraft in simulation_data.spacecrafts:

            history = self.spacecrafts_history.setdefault(
                spacecraft.name,
                SpacecraftHistory()
            )

            history.record(spacecraft.spacecraft_data)


    def record_spacecraft(self, spacecraft):

        history = self.spacecrafts_history.setdefault(
            spacecraft.name,
            SpacecraftHistory()
        )

        history.record(spacecraft.spacecraft_data)

    def finalize(self):

        self.time = np.asarray(self.time)

        for history in self.spacecrafts_history.values():
            history.finalize()