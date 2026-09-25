import copy
from dataclasses import is_dataclass
import numpy as np

def _stack(obj_list, path="root"):
    """
    Convierte recursivamente dataclasses y arrays en estructuras
    de NumPy, verificando consistencia de dimensiones.
    """

    if not obj_list:
        return np.asarray([])

    first = obj_list[0]

    if is_dataclass(first):

        result = {}

        for field in first.__dataclass_fields__:

            values = [
                getattr(obj, field)
                for obj in obj_list
            ]

            result[field] = _stack(
                values,
                path=f"{path}.{field}"
            )

        return result

    if isinstance(first, np.ndarray):

        shapes = [np.shape(value) for value in obj_list]

        if len(set(shapes)) != 1:

            print("\nINCONSISTENT ARRAY SHAPES")
            print(f"Field: {path}")
            print(f"Number of samples: {len(obj_list)}")

            unique_shapes = {}

            for i, shape in enumerate(shapes):
                unique_shapes.setdefault(shape, []).append(i)

            for shape, indices in unique_shapes.items():
                print(
                    f"  shape={shape}, "
                    f"samples={indices[:10]}"
                    f"{'...' if len(indices) > 10 else ''}"
                )

            raise ValueError(
                f"Inconsistent shapes in '{path}': "
                f"{list(unique_shapes.keys())}"
            )

        return np.asarray(obj_list)

    return obj_list


class SpacecraftHistory:

    def __init__(self):

        self.t = []

        self.tick = []

        self.true_state = []

        self.navigation_data = []
        self.guidance_data = []
        self.control_data = []

        self.current_force_exerted = []
        self.current_torque_exerted = []

        self.target_name = []

    def record(self, data):

        self.t.append(data.t)
        self.tick.append(data.tick)

        self.true_state.append(copy.deepcopy(data.true_state))

        self.navigation_data.append(copy.deepcopy(data.navigation_data))
        self.guidance_data.append(copy.deepcopy(data.guidance_data))
        self.control_data.append(copy.deepcopy(data.control_data))

        self.current_force_exerted.append(data.current_force_exerted.copy())
        self.current_torque_exerted.append(data.current_torque_exerted.copy())

        self.target_name.append(data.target_name)

    def finalize(self):

        # Remove initial sample (t = 0)
        self.t = self.t[1:]
        self.tick = self.tick[1:]

        self.true_state = self.true_state[1:]
        self.navigation_data = self.navigation_data[1:]
        self.guidance_data = self.guidance_data[1:]
        self.control_data = self.control_data[1:]

        self.current_force_exerted = self.current_force_exerted[1:]
        self.current_torque_exerted = self.current_torque_exerted[1:]

        self.target_name = self.target_name[1:]

        self.t = np.asarray(self.t)
        self.tick = np.asarray(self.tick)

        self.true_state = _stack(self.true_state)

        self.navigation_data = _stack(self.navigation_data)

        self.guidance_data = _stack(self.guidance_data)

        self.control_data = _stack(self.control_data)

        self.current_force_exerted = np.asarray(self.current_force_exerted)

        self.current_torque_exerted = np.asarray(self.current_torque_exerted)

        self.target_name = np.asarray(self.target_name)


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

        self.time = np.asarray(self.time)[1:]

        for history in self.spacecrafts_history.values():
            history.finalize()