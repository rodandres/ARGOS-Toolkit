import numpy as np

import json
import os
import time

from dataclasses import dataclass

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from py.general.dataclasses import SpacecraftData


@dataclass
class SimulationHistoryMetadata:
    data_file_path: str
    unique_id: int
    spacecraft_names: list[str]
    chunk_size: dict
    chunk_count: dict
    

@dataclass
class TransitionEventInfo:
    time: float
    tick: int
    from_phase: str
    to_phase: str
    transition_name: str

    dt_guidance: float
    dt_navigation: float
    dt_control: float
    dt_propagation: float
    dt_master: float

class SimulationHistory:
    def __init__(self,
                 chunk_size: int,
                 data_file_path: str,
                 auto_convert_to_csv:bool,
                 csv_folder_path: str = "sim_data"):        
        
        self.chunk_size = chunk_size                
        self.auto_convert_to_csv = auto_convert_to_csv
        self.csv_folder_path = csv_folder_path
        self.spacecraft_names = []  # List to hold the names of spacecrafts in the simulation
        
        self.unique_id = int(time.time() * 1000)  # Unique ID based on current time in milliseconds

        # Create Main directory
        main_path = os.path.join(data_file_path, f"simulation_{self.unique_id}")
        if not os.path.exists(main_path):
            os.makedirs(main_path)

        self.data_file_path = main_path

        # Create a dictionary to hold the history for each spacecraft
        spacecraft_histories = {}        

        self.spacecrafts_history = spacecraft_histories

    def add_spacecraft_history(self, spacecraft_name: str):
        if spacecraft_name in self.spacecrafts_history:
            raise ValueError(f"Spacecraft '{spacecraft_name}' already exists in simulation history.")

        spacecraft_history = SpacecraftHistory(
            spacecraft_name=spacecraft_name,
            chunk_size=self.chunk_size,
            data_file_path=self.data_file_path
        )

        self.spacecrafts_history[spacecraft_name] = spacecraft_history
        self.spacecraft_names.append(spacecraft_name)

        print(f"[INFO] Spacecraft '{spacecraft_name}' history initialized. Data will be saved in '{self.data_file_path}'.")
        
    def record_spacecraft(self, spacecraft_name: str, data: "SpacecraftData"):
        if spacecraft_name in self.spacecrafts_history:
            self.spacecrafts_history[spacecraft_name].record(data)
        else:
            raise ValueError(f"Spacecraft '{spacecraft_name}' not found in simulation history.")
        
    def record_event(self, spacecraft_name: str, event: TransitionEventInfo):
        if spacecraft_name in self.spacecrafts_history:
            self.spacecrafts_history[spacecraft_name].record_event(event)
        else:
            raise ValueError(f"Spacecraft '{spacecraft_name}' not found in simulation history.")

    def finalize(self):
        for spacecraft_history in self.spacecrafts_history.values():
            spacecraft_history.flush()

        metadata = SimulationHistoryMetadata(
            data_file_path=self.data_file_path,
            unique_id=self.unique_id,
            spacecraft_names=self.spacecraft_names,
            chunk_size= {name: history.chunk_size for name, history in self.spacecrafts_history.items()},
            chunk_count={name: history.chunk_index for name, history in self.spacecrafts_history.items()}
        )

        metadata_file_path = os.path.join(self.data_file_path, "simulation_history_metadata.json")
        with open(metadata_file_path, 'w') as f:
            json.dump(metadata.__dict__, f, indent=4)

        # Create transition events file
        for spacecraft_name, spacecraft_history in self.spacecrafts_history.items():
            events_file_path = os.path.join(self.data_file_path, f"{spacecraft_name}_transition_events.json")
            events_data = [event.__dict__ for event in spacecraft_history.events]
            with open(events_file_path, 'w') as f:
                json.dump(events_data, f, indent=4)

        if self.auto_convert_to_csv:
            save_simulation_history_csv(metadata_file_path, self.csv_folder_path)


class SpacecraftHistory:
    def __init__(self, spacecraft_name: str, chunk_size: int, data_file_path: str):
        self.spacecraft_name = spacecraft_name

        self.chunk_size = chunk_size
        self.data_file_path = data_file_path

        self.chunk_index = 0
        self.index = 0

        self.t = np.empty(self.chunk_size, dtype=np.float64)
        self.tick = np.empty(self.chunk_size, dtype=np.int64)

        # True state
        self.true_position = np.empty((self.chunk_size, 3), dtype=np.float64)
        self.true_velocity = np.empty((self.chunk_size, 3), dtype=np.float64)
        self.true_acceleration = np.empty((self.chunk_size, 3), dtype=np.float64)
        self.true_attitude = np.empty((self.chunk_size, 4), dtype=np.float64)  # Quaternion
        self.true_angular_velocity = np.empty((self.chunk_size, 3), dtype=np.float64)
        self.true_angular_acceleration = np.empty((self.chunk_size, 3), dtype=np.float64)

        # Estimated data
        self.estimated_sc_position = np.empty((self.chunk_size, 3), dtype=np.float64)
        self.estimated_sc_velocity = np.empty((self.chunk_size, 3), dtype=np.float64)
        self.estimated_sc_acceleration = np.empty((self.chunk_size, 3), dtype=np.float64)
        self.estimated_sc_attitude = np.empty((self.chunk_size, 4), dtype=np.float64)  # Quaternion
        self.estimated_sc_angular_velocity = np.empty((self.chunk_size, 3), dtype=np.float64)
        self.estimated_sc_angular_acceleration = np.empty((self.chunk_size, 3), dtype=np.float64)

        self.estimated_target_position = np.empty((self.chunk_size, 3), dtype=np.float64)
        self.estimated_target_velocity = np.empty((self.chunk_size, 3), dtype=np.float64)
        self.estimated_target_acceleration = np.empty((self.chunk_size, 3), dtype=np.float64)
        self.estimated_target_attitude = np.empty((self.chunk_size, 4), dtype=np.float64)  # Quaternion
        self.estimated_target_angular_velocity = np.empty((self.chunk_size, 3), dtype=np.float64)
        self.estimated_target_angular_acceleration = np.empty((self.chunk_size, 3), dtype=np.float64)

        # Guidance data
        self.guidance_position = np.empty((self.chunk_size, 3), dtype=np.float64)
        self.guidance_velocity = np.empty((self.chunk_size, 3), dtype=np.float64)
        self.guidance_acceleration = np.empty((self.chunk_size, 3), dtype=np.float64)
        self.guidance_attitude = np.empty((self.chunk_size, 4), dtype=np.float64)  # Quaternion
        self.guidance_angular_velocity = np.empty((self.chunk_size, 3), dtype=np.float64)
        self.guidance_angular_acceleration = np.empty((self.chunk_size, 3), dtype=np.float64)

        # Control output data
        self.control_force = np.empty((self.chunk_size, 3), dtype=np.float64)
        self.control_torque = np.empty((self.chunk_size, 3), dtype=np.float64)

        # Excerted forces and torques
        self.current_force_exerted = np.empty((self.chunk_size, 3), dtype=np.float64)
        self.current_torque_exerted = np.empty((self.chunk_size, 3), dtype=np.float64)

        # Events
        self.events = []  # List of TransitionEventInfo objects

    def record(self, data: SpacecraftData):

        # True state
        self.true_position[self.index] = data.true_state.position
        self.true_velocity[self.index] = data.true_state.velocity
        self.true_acceleration[self.index] = data.true_state.acceleration
        self.true_attitude[self.index] = data.true_state.attitude
        self.true_angular_velocity[self.index] = data.true_state.angular_velocity
        self.true_angular_acceleration[self.index] = data.true_state.angular_acceleration

        # Estimated data
        self.estimated_sc_position[self.index] = data.navigation_data.spacecraft_state.position
        self.estimated_sc_velocity[self.index] = data.navigation_data.spacecraft_state.velocity
        self.estimated_sc_acceleration[self.index] = data.navigation_data.spacecraft_state.acceleration
        self.estimated_sc_attitude[self.index] = data.navigation_data.spacecraft_state.attitude
        self.estimated_sc_angular_velocity[self.index] = data.navigation_data.spacecraft_state.angular_velocity
        self.estimated_sc_angular_acceleration[self.index] = data.navigation_data.spacecraft_state.angular_acceleration

        self.estimated_target_position[self.index] = data.navigation_data.target_state.position
        self.estimated_target_velocity[self.index] = data.navigation_data.target_state.velocity
        self.estimated_target_acceleration[self.index] = data.navigation_data.target_state.acceleration
        self.estimated_target_attitude[self.index] = data.navigation_data.target_state.attitude
        self.estimated_target_angular_velocity[self.index] = data.navigation_data.target_state.angular_velocity
        self.estimated_target_angular_acceleration[self.index] = data.navigation_data.target_state.angular_acceleration

        # Guidance data
        self.guidance_position[self.index] = data.guidance_data.state.position
        self.guidance_velocity[self.index] = data.guidance_data.state.velocity
        self.guidance_acceleration[self.index] = data.guidance_data.state.acceleration
        self.guidance_attitude[self.index] = data.guidance_data.state.attitude
        self.guidance_angular_velocity[self.index] = data.guidance_data.state.angular_velocity
        self.guidance_angular_acceleration[self.index] = data.guidance_data.state.angular_acceleration

        # Control output data
        self.control_force[self.index] = data.control_data.force
        self.control_torque[self.index] = data.control_data.torque

        # Excerted forces and torques
        self.current_force_exerted[self.index] = data.current_force_exerted
        self.current_torque_exerted[self.index] = data.current_torque_exerted

        self.t[self.index] = data.t
        self.tick[self.index] = data.tick

        self.index += 1

        if self.index >= self.chunk_size:
            self.flush()

    def record_event(self, event: TransitionEventInfo):
        self.events.append(event)

    def flush(self):
        #print(f"[DEBUG] Flushing history for spacecraft '{self.spacecraft_name}' to disk. Chunk index: {self.chunk_index}")

        if self.index == 0:
            return  # Nothing to flush

        filename = os.path.join(
           self.data_file_path,
             f"{self.spacecraft_name}_history_chunk_{self.chunk_index}.npz"
        )        

        np.savez(
            filename,

            # Time
            t=self.t[:self.index],
            tick=self.tick[:self.index],

            # True state
            true_position=self.true_position[:self.index],
            true_velocity=self.true_velocity[:self.index],
            true_acceleration=self.true_acceleration[:self.index],
            true_attitude=self.true_attitude[:self.index],
            true_angular_velocity=self.true_angular_velocity[:self.index],
            true_angular_acceleration=self.true_angular_acceleration[:self.index],

            # Estimated spacecraft state
            estimated_sc_position=self.estimated_sc_position[:self.index],
            estimated_sc_velocity=self.estimated_sc_velocity[:self.index],
            estimated_sc_acceleration=self.estimated_sc_acceleration[:self.index],
            estimated_sc_attitude=self.estimated_sc_attitude[:self.index],
            estimated_sc_angular_velocity=self.estimated_sc_angular_velocity[:self.index],
            estimated_sc_angular_acceleration=self.estimated_sc_angular_acceleration[:self.index],

            # Estimated target state
            estimated_target_position=self.estimated_target_position[:self.index],
            estimated_target_velocity=self.estimated_target_velocity[:self.index],
            estimated_target_acceleration=self.estimated_target_acceleration[:self.index],
            estimated_target_attitude=self.estimated_target_attitude[:self.index],
            estimated_target_angular_velocity=self.estimated_target_angular_velocity[:self.index],
            estimated_target_angular_acceleration=self.estimated_target_angular_acceleration[:self.index],

            # Guidance
            guidance_position=self.guidance_position[:self.index],
            guidance_velocity=self.guidance_velocity[:self.index],
            guidance_acceleration=self.guidance_acceleration[:self.index],
            guidance_attitude=self.guidance_attitude[:self.index],
            guidance_angular_velocity=self.guidance_angular_velocity[:self.index],
            guidance_angular_acceleration=self.guidance_angular_acceleration[:self.index],

            # Control
            control_force=self.control_force[:self.index],
            control_torque=self.control_torque[:self.index],

            # Exerted
            current_force_exerted=self.current_force_exerted[:self.index],
            current_torque_exerted=self.current_torque_exerted[:self.index],
        )

        self.chunk_index += 1
        self.index = 0

        #print(f"[DEBUG] History for spacecraft '{self.spacecraft_name}' flushed to '{filename}'.")

def save_simulation_history_csv(
    metadata_file: str,
    main_directory: str = "sim_data",
):
    """
    Convert simulation history NPZ chunks into one CSV file per spacecraft.

    Parameters
    ----------
    metadata_file : str
        Path to the simulation metadata.json file.

    main_directory : str
        Main directory where the CSV simulation directory will be created.
    """

    # ==========================================================================
    # Load metadata
    # ==========================================================================

    with open(
        metadata_file,
        "r",
        encoding="utf-8",
    ) as file:

        metadata = json.load(file)

    data_file_path = metadata["data_file_path"]
    unique_id = metadata["unique_id"]
    spacecraft_names = metadata["spacecraft_names"]
    chunk_count = metadata["chunk_count"]

    # ==========================================================================
    # Create output directory
    # ==========================================================================

    output_directory = os.path.join(
        main_directory,
        f"simulation_{unique_id}",
    )

    os.makedirs(
        output_directory,
        exist_ok=True,
    )

    # ==========================================================================
    # Variables to export
    # ==========================================================================

    variables = [
        "t",
        "tick",

        # True state
        "true_position",
        "true_velocity",
        "true_acceleration",
        "true_attitude",
        "true_angular_velocity",
        "true_angular_acceleration",

        # Estimated spacecraft state
        "estimated_sc_position",
        "estimated_sc_velocity",
        "estimated_sc_acceleration",
        "estimated_sc_attitude",
        "estimated_sc_angular_velocity",
        "estimated_sc_angular_acceleration",

        # Estimated target state
        "estimated_target_position",
        "estimated_target_velocity",
        "estimated_target_acceleration",
        "estimated_target_attitude",
        "estimated_target_angular_velocity",
        "estimated_target_angular_acceleration",

        # Guidance
        "guidance_position",
        "guidance_velocity",
        "guidance_acceleration",
        "guidance_attitude",
        "guidance_angular_velocity",
        "guidance_angular_acceleration",

        # Control
        "control_force",
        "control_torque",

        # Exerted
        "current_force_exerted",
        "current_torque_exerted",
    ]

    # ==========================================================================
    # Process each spacecraft
    # ==========================================================================

    for spacecraft_name in spacecraft_names:

        output_file = os.path.join(
            output_directory,
            f"{spacecraft_name}.csv",
        )

        number_of_chunks = chunk_count[spacecraft_name]

        first_chunk = True

        # ======================================================================
        # Process chunks
        # ======================================================================

        for chunk_index in range(number_of_chunks):

            chunk_file = os.path.join(
                data_file_path,
                f"{spacecraft_name}_history_chunk_{chunk_index}.npz",
            )

            if not os.path.exists(chunk_file):

                raise FileNotFoundError(
                    f"History chunk not found: '{chunk_file}'"
                )

            # ==================================================================
            # Load current chunk
            # ==================================================================

            with np.load(chunk_file) as chunk:

                arrays = []
                columns = []

                # ==============================================================
                # Extract variables
                # ==============================================================

                for variable in variables:

                    if variable not in chunk:
                        continue

                    data = chunk[variable]

                    # ----------------------------------------------------------
                    # 1D data
                    # ----------------------------------------------------------

                    if data.ndim == 1:

                        data = data.reshape(
                            -1,
                            1,
                        )

                        variable_columns = [
                            variable
                        ]

                    # ----------------------------------------------------------
                    # 2D or higher-dimensional data
                    # ----------------------------------------------------------

                    else:

                        data = data.reshape(
                            data.shape[0],
                            -1,
                        )

                        variable_columns = [
                            f"{variable}_{i}"
                            for i in range(data.shape[1])
                        ]

                    arrays.append(data)
                    columns.extend(variable_columns)

                # ==============================================================
                # Check that data exists
                # ==============================================================

                if not arrays:
                    continue

                # ==============================================================
                # Combine variables horizontally
                # ==============================================================

                chunk_data = np.column_stack(
                    arrays
                )

                # ==============================================================
                # Write CSV
                # ==============================================================

                with open(
                    output_file,
                    "ab" if not first_chunk else "wb",
                ) as file:

                    # ----------------------------------------------------------
                    # Header
                    # ----------------------------------------------------------

                    if first_chunk:

                        header = ",".join(
                            columns
                        ) + "\n"

                        file.write(
                            header.encode("utf-8")
                        )

                    # ----------------------------------------------------------
                    # Data
                    # ----------------------------------------------------------

                    np.savetxt(
                        file,
                        chunk_data,
                        delimiter=",",
                    )

            first_chunk = False

    return output_directory