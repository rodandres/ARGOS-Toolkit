import numpy as np

import json
import os
import time

from dataclasses import dataclass

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from argos.general.dataclasses import SpacecraftData


@dataclass
class SimulationHistoryMetadata:
    """
    Store metadata describing a completed simulation history.

    Attributes
    ----------
    data_file_path : str
        Directory containing the simulation history data.
    unique_id : int
        Unique identifier assigned to the simulation history.
    spacecraft_names : list of str
        Names of the spacecraft recorded in the simulation history.
    chunk_size : dict
        History chunk size associated with each spacecraft.
    chunk_count : dict
        Number of history chunks generated for each spacecraft.
    csv_folder_path : str, optional
        Directory containing the CSV representation of the simulation
        history, if CSV conversion was enabled.
    targets_info : dict, optional
        Mapping between spacecraft names and their assigned target
        spacecraft names.
    """
    data_file_path: str    
    unique_id: int
    spacecraft_names: list[str]
    chunk_size: dict
    chunk_count: dict
    csv_folder_path: str = ""
    targets_info: dict = None

@dataclass
class TransitionEventInfo:
    """
    Store information about a spacecraft mission phase transition.

    Attributes
    ----------
    time : float
        Simulation time at which the transition occurred [s].
    tick : int
        Simulation tick at which the transition occurred.
    from_phase : str
        Mission phase active before the transition.
    to_phase : str
        Mission phase active after the transition.
    transition_name : str
        Name of the transition that triggered the phase change.
    dt_guidance : float
        Guidance update period active at the transition [s].
    dt_navigation : float
        Navigation update period active at the transition [s].
    dt_control : float
        Control update period active at the transition [s].
    dt_propagation : float
        Propagation time step active at the transition [s].
    dt_master : float
        Master simulation time step active at the transition [s].
    """
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
    """
    Manage persistent simulation history data for multiple spacecraft.

    Simulation history is stored in memory using fixed-size chunks and
    periodically written to NPZ files. The history can optionally be
    converted to CSV files when the simulation is finalized.

    Parameters
    ----------
    chunk_size : int
        Number of simulation records stored in each history chunk.
    data_file_path : str
        Directory where the simulation history data will be stored.
    auto_convert_to_csv : bool
        If True, convert the recorded history to CSV files when finalized.
    csv_folder_path : str, optional
        Directory where CSV simulation data will be stored.
    """
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
        """
        Initialize history storage for a spacecraft.

        Parameters
        ----------
        spacecraft_name : str
            Name of the spacecraft whose history will be recorded.

        Raises
        ------
        ValueError
            If history for the spacecraft already exists.
        """
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
        """
        Record the current spacecraft data in its simulation history.

        Parameters
        ----------
        spacecraft_name : str
            Name of the spacecraft whose data is being recorded.
        data : SpacecraftData
            Current spacecraft simulation data.

        Raises
        ------
        ValueError
            If the spacecraft does not have an initialized history.
        """
        if spacecraft_name in self.spacecrafts_history:
            self.spacecrafts_history[spacecraft_name].record(data)
        else:
            raise ValueError(f"Spacecraft '{spacecraft_name}' not found in simulation history.")
        
    def record_event(self, spacecraft_name: str, event: TransitionEventInfo):
        """
        Record a mission phase transition event for a spacecraft.

        Parameters
        ----------
        spacecraft_name : str
            Name of the spacecraft associated with the event.
        event : TransitionEventInfo
            Mission phase transition information.

        Raises
        ------
        ValueError
            If the spacecraft does not have an initialized history.
        """
        if spacecraft_name in self.spacecrafts_history:
            self.spacecrafts_history[spacecraft_name].record_event(event)
        else:
            raise ValueError(f"Spacecraft '{spacecraft_name}' not found in simulation history.")

    def record_target(self, spacecraft_name: str, target_name: str):
        """
        Record a target spacecraft associated with a spacecraft history.

        Parameters
        ----------
        spacecraft_name : str
            Name of the spacecraft to which the target is assigned.
        target_name : str
            Name of the target spacecraft.

        Raises
        ------
        ValueError
            If the spacecraft does not have an initialized history.
        """
        if spacecraft_name in self.spacecrafts_history:
            self.spacecrafts_history[spacecraft_name].record_target(target_name)
        else:
            raise ValueError(f"Spacecraft '{spacecraft_name}' not found in simulation history.")

    def finalize(self):
        """
        Finalize and persist the simulation history.

        Remaining in-memory history chunks are written to disk, simulation
        metadata and mission transition events are saved as JSON files, and
        the history is optionally converted to CSV.

        Returns
        -------
        SimulationHistoryMetadata
            Metadata describing the finalized simulation history.
        """
        for spacecraft_history in self.spacecrafts_history.values():
            spacecraft_history.flush()
                                
        targets_dict = {}

        for spacecraft_name, spacecraft_history in self.spacecrafts_history.items():
            targets_dict[spacecraft_name] = spacecraft_history.targets

        metadata = SimulationHistoryMetadata(
            data_file_path=self.data_file_path,
            unique_id=self.unique_id,
            spacecraft_names=self.spacecraft_names,
            chunk_size= {name: history.chunk_size for name, history in self.spacecrafts_history.items()},
            chunk_count={name: history.chunk_index for name, history in self.spacecrafts_history.items()},
            targets_info=targets_dict
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
            metadata.csv_folder_path = self.csv_folder_path

            save_simulation_history_csv(metadata_file_path, self.csv_folder_path)

            # Save the metadata file in the CSV folder as well
            csv_metadata_file_path = os.path.join(self.csv_folder_path, f"simulation_{self.unique_id}", "simulation_history_metadata.json")
            os.makedirs(os.path.dirname(csv_metadata_file_path), exist_ok=True)
            with open(csv_metadata_file_path, 'w') as f:
                json.dump(metadata.__dict__, f, indent=4)

        return metadata


class SpacecraftHistory:
    """
    Store time-history data for a single spacecraft.

    The history contains true, estimated, guidance, control, and exerted
    force and torque data. Records are accumulated in fixed-size chunks and
    written to NPZ files when a chunk is full.

    Parameters
    ----------
    spacecraft_name : str
        Name of the spacecraft.
    chunk_size : int
        Number of records stored in each history chunk.
    data_file_path : str
        Directory where history chunk files are stored.
    """
    def __init__(
        self,
        spacecraft_name: str,
        chunk_size: int,
        data_file_path: str,
    ):
        """
        Initialize the history storage for a spacecraft.

        Parameters
        ----------
        spacecraft_name : str
            Name of the spacecraft.
        chunk_size : int
            Number of records stored in each history chunk.
        data_file_path : str
            Directory where history chunk files are stored.
        """
        self.spacecraft_name = spacecraft_name
        self.targets = []  # List to hold the names of targets for this spacecraft

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
        """
        Record one spacecraft simulation state.

        The current true state, navigation estimates, guidance state, control
        output, exerted force and torque, simulation time, and simulation tick
        are stored in the current history chunk.

        Parameters
        ----------
        data : SpacecraftData
            Spacecraft simulation data to record.
        """
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
        """
        Record a mission phase transition event.

        Parameters
        ----------
        event : TransitionEventInfo
            Transition event information to store.
        """
        self.events.append(event)

    def record_target(self, target_name: str):
        """
        Register a target spacecraft for this spacecraft history.

        Duplicate target names are ignored.

        Parameters
        ----------
        target_name : str
            Name of the target spacecraft.
        """
        if target_name not in self.targets:
            self.targets.append(target_name)

    def flush(self):
        """
        Write the current history chunk to an NPZ file.

        If the current chunk contains no records, no file is created.
        After a successful flush, the chunk index is incremented and the
        in-memory record index is reset.
        """
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
        Path to the simulation metadata JSON file.
    main_directory : str, optional
        Main directory where the CSV simulation directory will be created.

    Returns
    -------
    str
        Path to the directory containing the generated CSV files.

    Raises
    ------
    FileNotFoundError
        If a history chunk referenced by the metadata file does not exist.
    """
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