# NOTAS DE IMPLEMENTACION:
# Se debe añadir una verificacion antes de correr la sim garantizando que se tengan spacecrafts
# Se debe añadir una verificacion o manejo de los propagadores en caso de ser None

import time
from collections import defaultdict

import numpy as np

from argos.core.spacecraft import Spacecraft
from argos.general.dataclasses import SimulationData
from argos.general.data_save import SimulationHistory

from typing import TYPE_CHECKING

from argos.sensors.sensor_base import SensorBase

if TYPE_CHECKING:
    from argos.enviroments.environment_base import EnvironmentBase
    from argos.actuators.actuators_base import ActuatorBase 
    from argos.core.mission_manager import MissionManager
    from argos.general.data_save import TransitionEventInfo

class Simulation:
    """
    Manage and execute a spacecraft simulation.

    Parameters
    ----------
    max_sim_time : float
        Maximum simulation time [s].
    environment : EnvironmentBase
        Environment model used during the simulation.
    history_chunk_size : int, optional
        Number of simulation history records stored in each chunk before
        being written to disk.
    history_data_file_path : str, optional
        Path used for temporary simulation history data.
    auto_save_csv : bool, optional
        If True, automatically convert simulation history data to CSV.
    csv_folder_path : str, optional
        Directory where CSV simulation data is stored. Defaults to
        ``"sim_data"``.
    verbose : bool, optional
        If True, print simulation status and spacecraft information.
    """

    def __init__(
        self,
        max_sim_time: float,
        environment: EnvironmentBase,
        history_chunk_size: int = 4096,
        history_data_file_path: str = "tmp",
        auto_save_csv: bool = True,
        csv_folder_path: str = None,
        verbose: bool = False,
    ):   

        simulation_data = SimulationData(
            max_sim_time=max_sim_time,
            spacecrafts=[],            
            )        

        self.simulation_data = simulation_data

        self.environment = environment
        self.history_chunk_size = history_chunk_size
        self.history_data_file_path = history_data_file_path
        self.auto_save_csv = auto_save_csv
        self.csv_folder_path = csv_folder_path

        if self.csv_folder_path is None:
            self.csv_folder_path = "sim_data"

        self.simulation_data.simulation_history = SimulationHistory(
                                                    chunk_size=self.history_chunk_size,
                                                    data_file_path=self.history_data_file_path,
                                                    auto_convert_to_csv=self.auto_save_csv,
                                                    csv_folder_path=self.csv_folder_path
                                                )        

        self.verbose = verbose

    def add_spacecraft(self,
                       mass: float | None=None, # NOTE: Add the type of the mass               
                       initial_position: np.ndarray | None=None, # NOTE: Add the type of the initial position
                       initial_velocity: np.ndarray | None=None, # NOTE: Add the type of the initial
                       initial_attitude: np.ndarray | None=None, # NOTE: Add the type of the initial attitude
                       initial_angular_velocity: np.ndarray | None=None, # NOTE: Add the type of
                       inertia_tensor: np.ndarray | None=None,
                       actuators: list[ActuatorBase] | None=None, # NOTE: Add the type of the list
                       sensors: list[SensorBase] | None=None, # NOTE: Add the type of the list
                       mission_manager: MissionManager | None=None, # NOTE: Add the type of the mission manager
                       target_name: str | None=None, # NOTE: Add the type of the target name
                       name: str |None=None,
                       verbose=False):
        """
        Add a spacecraft to the simulation.

        Parameters
        ----------
        mass : float, optional
            Spacecraft mass [kg]. Defaults to 0.0.
        initial_position : ndarray, shape (3,), optional
            Initial spacecraft position [m]. Defaults to a zero vector.
        initial_velocity : ndarray, shape (3,), optional
            Initial spacecraft velocity [m/s]. Defaults to a zero vector.
        initial_attitude : ndarray, shape (4,), optional
            Initial spacecraft attitude represented as a quaternion.
            Defaults to [0, 0, 0, 1].
        initial_angular_velocity : ndarray, shape (3,), optional
            Initial spacecraft angular velocity [rad/s]. Defaults to a zero
            vector.
        inertia_tensor : ndarray, shape (3, 3), optional
            Spacecraft inertia tensor [kg·m²]. Defaults to the identity matrix.
        actuators : list of ActuatorBase, optional
            Actuators attached to the spacecraft.
        sensors : list of SensorBase, optional
            Sensors attached to the spacecraft.
        mission_manager : MissionManager, optional
            Mission manager associated with the spacecraft.
        target_name : str, optional
            Name of the spacecraft target, if applicable.
        name : str, optional
            Spacecraft identifier. If not provided, a name is generated
            automatically.
        verbose : bool, optional
            If True, print information about the spacecraft during creation.

        Returns
        -------
        None
            The spacecraft is added directly to the simulation.
        """
        

        if mass is None:
            if verbose: print("Mass not provided. Using default mass of 0 kg.")
            mass = 0.0  # Default mass

        if initial_position is None:
            if verbose: print("Initial position not provided. Using default position of [0, 0, 0].")
            initial_position = np.zeros(3)  # Default position

        if initial_velocity is None:
            if verbose: print("Initial velocity not provided. Using default velocity of [0, 0, 0].")
            initial_velocity = np.zeros(3)  # Default velocity

        if initial_attitude is None:
            if verbose: print("Initial attitude not provided. Using default quaternion of [0, 0, 0, 1].")
            initial_attitude = np.array([0, 0, 0, 1])  # Default quaternion

        if initial_angular_velocity is None:
            if verbose: print("Initial angular velocity not provided. Using default angular velocity of [0, 0, 0].")
            initial_angular_velocity = np.zeros(3)  # Default angular velocity

        if inertia_tensor is None:
            if verbose: print("Inertia tensor not provided. Using default inertia tensor of identity matrix.")
            inertia_tensor = np.eye(3)  # Default inertia tensor

        if mission_manager is None:
            if verbose: print("Mission manager not provided. Using default mission manager of None.")
            mission_manager = None  # Default mission manager

        if name is None:
            name = f"Spacecraft_{len(self.simulation_data.spacecrafts)+1}"        
        
        self.simulation_data.simulation_history.add_spacecraft_history(name)

        initial_state = np.concatenate((initial_position, initial_velocity, initial_attitude, initial_angular_velocity))

        spacecraft = Spacecraft(
            name=name,
            mass=mass,
            initial_state=initial_state,
            inertia_tensor=inertia_tensor,
            actuators=actuators,
            sensors=sensors,            
            mission_manager=mission_manager, 
            parent=self,
            verbose=verbose
        )

        self.simulation_data.spacecrafts.append(spacecraft)


        if target_name is not None:
            self.add_spacecraft_target(spacecraft_name=name, target_name=target_name)
            self.simulation_data.simulation_history.record_target(spacecraft_name=name, target_name=target_name)

        if self.verbose:
            print(f"Spacecraft '{name}' added to the simulation.")

    def add_spacecraft_target(self, spacecraft_name: str, target_name: str):
        """
        Assign a target spacecraft to a spacecraft in the simulation.

        Parameters
        ----------
        spacecraft_name : str
            Name of the spacecraft to which the target is assigned.
        target_name : str
            Name of the target spacecraft.
        """
        for spacecraft in self.simulation_data.spacecrafts:
            if spacecraft.name == spacecraft_name:
                spacecraft.change_target(target_name)
                self.simulation_data.simulation_history.record_target(spacecraft_name=spacecraft_name, target_name=target_name)
                if self.verbose:
                    print(f"Target '{target_name}' assigned to spacecraft '{spacecraft_name}'.")
                return

    def _set_dt_master(self): # POSSIBLE BUG: dts must be with a minimum common multiple
        dts = []
        for spacecraft in self.simulation_data.spacecrafts:
            dt_nav, dt_guid, dt_control, dt_propagation = spacecraft.get_dts()
            print(f"Spacecraft '{spacecraft.name}' dt_nav: {dt_nav}, dt_guid: {dt_guid}, dt_control: {dt_control}")
            if not np.isnan(dt_nav):
                dts.append(dt_nav)
            if not np.isnan(dt_guid):
                dts.append(dt_guid)
            if not np.isnan(dt_control):
                dts.append(dt_control)

            dts.append(dt_propagation)  # Add the spacecraft's propagation time step

        self.simulation_data.dt_master = min(dts)

    def record_transition(self, spacecraft_name: str, event_info: TransitionEventInfo):
        """
        Record a spacecraft mission transition in the simulation history.

        Parameters
        ----------
        spacecraft_name : str
            Name of the spacecraft associated with the transition.
        event_info : TransitionEventInfo
            Information describing the mission phase transition.
        """
        self.simulation_data.simulation_history.record_event(spacecraft_name, event_info)

    def _init_simulation(self):
        if self.verbose:
            print("Initializing simulation...")
        
        self._set_dt_master()

        # Record initial states of all spacecrafts in the simulation history
        for spacecraft in self.simulation_data.spacecrafts:
            self.simulation_data.simulation_history.record_spacecraft(spacecraft.name, spacecraft.spacecraft_data)

        if self.verbose:
            print("Simulation initialized.")

    # def simulate(self):
    #     """
    #     Execute the spacecraft simulation.

    #     The simulation is initialized before execution and advances each
    #     spacecraft according to its configured mission, GNC components,
    #     actuators, and dynamics models. Simulation states and events are
    #     recorded in the configured simulation history.

    #     Returns
    #     -------
    #     dict
    #         Metadata generated when the simulation history is finalized.
    #     """
    #     if not self.simulation_data.spacecrafts:
    #         raise ValueError("No spacecrafts have been added to the simulation. Please add at least one spacecraft before running the simulation.")
        
    #     self._init_simulation()


    #     if self.verbose:
    #         print("Starting simulation...")


    #     t = 0.0
    #     current_ts = []
    #     self.simulation_data.t = np.nan
    #     self.simulation_data.tick = np.nan

    #     while t < self.simulation_data.max_sim_time:

    #         for spacecraft in self.simulation_data.spacecrafts:
    #             if spacecraft.spacecraft_data.t >= self.simulation_data.max_sim_time:                    
    #                 continue  # Skip this spacecraft if its time exceeds the max simulation time

    #             # Update Mission Manager
    #             spacecraft.update_mission_manager(self.simulation_data)                

    #             # Update sensors
    #             spacecraft.update_sensors(self.simulation_data)

    #             # Update Navigation
    #             spacecraft.update_navigation(self.simulation_data)

    #             # Update Guidance
    #             spacecraft.update_guidance(self.simulation_data)

    #             # Update Control
    #             spacecraft.update_control(self.simulation_data)

    #             # Compute Actuation
    #             spacecraft.compute_actuation(self.simulation_data)  

    #             # Propagate Translational and Rotational Dynamics
    #             spacecraft.propagate_translational(self.simulation_data, self.environment)
    #             spacecraft.propagate_rotational(self.simulation_data, self.environment)

    #             t_spacecraft = spacecraft.spacecraft_data.t
    #             spacecraft.spacecraft_data.t += spacecraft.spacecraft_data.current_master_dt
    #             spacecraft.spacecraft_data.tick += 1
    #             current_ts.append(t_spacecraft)
                
    #             self.simulation_data.simulation_history.record_spacecraft(spacecraft.name, spacecraft.spacecraft_data)


    #         try:
    #             t = min(current_ts)
    #         except ValueError:
    #             break  # Exit the loop if there are no spacecrafts to simulate

    #         current_ts = []  # Reset for the next iteration
            
        
    #     if self.verbose: print("Simulation completed.")
    #     timer_end = time.perf_counter()        
        
    #     metadata = self.simulation_data.simulation_history.finalize()  # Finalize the history after the simulation is complete

    #     return metadata

    def simulate(self, profile: bool = True):
        """
        Execute the spacecraft simulation.

        Parameters
        ----------
        profile : bool, optional
            If True, measure execution time for each simulation module.

        Returns
        -------
        dict
            Metadata generated when the simulation history is finalized.
        """

        if not self.simulation_data.spacecrafts:
            raise ValueError(
                "No spacecrafts have been added to the simulation. "
                "Please add at least one spacecraft before running the simulation."
            )

        self._init_simulation()

        if self.verbose:
            print("Starting simulation...")

        # ---------------------------------------------------------
        # Profiling setup
        # ---------------------------------------------------------

        profile_data = defaultdict(lambda: {
            "total_time": 0.0,
            "calls": 0,
        })

        total_start = time.perf_counter()

        # ---------------------------------------------------------
        # Helper for timing individual modules
        # ---------------------------------------------------------

        def timed_call(name, function, *args):
            """
            Execute a function and record its execution time.
            """

            if profile:
                start = time.perf_counter()

                result = function(*args)

                elapsed = time.perf_counter() - start

                profile_data[name]["total_time"] += elapsed
                profile_data[name]["calls"] += 1

                return result

            return function(*args)

        # ---------------------------------------------------------
        # Simulation
        # ---------------------------------------------------------

        t = 0.0
        current_ts = []

        self.simulation_data.t = np.nan
        self.simulation_data.tick = np.nan

        while t < self.simulation_data.max_sim_time:

            iteration_start = time.perf_counter()

            for spacecraft in self.simulation_data.spacecrafts:

                if spacecraft.spacecraft_data.t >= self.simulation_data.max_sim_time:
                    continue

                # -------------------------------------------------
                # Mission Manager
                # -------------------------------------------------

                timed_call(
                    "mission_manager",
                    spacecraft.update_mission_manager,
                    self.simulation_data
                )

                # -------------------------------------------------
                # Sensors
                # -------------------------------------------------

                timed_call(
                    "sensors",
                    spacecraft.update_sensors,
                    self.simulation_data
                )

                # -------------------------------------------------
                # Navigation
                # -------------------------------------------------

                timed_call(
                    "navigation",
                    spacecraft.update_navigation,
                    self.simulation_data
                )

                # -------------------------------------------------
                # Guidance
                # -------------------------------------------------

                timed_call(
                    "guidance",
                    spacecraft.update_guidance,
                    self.simulation_data
                )

                # -------------------------------------------------
                # Control
                # -------------------------------------------------

                timed_call(
                    "control",
                    spacecraft.update_control,
                    self.simulation_data
                )

                # -------------------------------------------------
                # Actuation
                # -------------------------------------------------

                timed_call(
                    "actuation",
                    spacecraft.compute_actuation,
                    self.simulation_data
                )

                # -------------------------------------------------
                # Translational dynamics
                # -------------------------------------------------

                timed_call(
                    "translation_propagation",
                    spacecraft.propagate_translational,
                    self.simulation_data,
                    self.environment
                )

                # -------------------------------------------------
                # Rotational dynamics
                # -------------------------------------------------

                timed_call(
                    "rotation_propagation",
                    spacecraft.propagate_rotational,
                    self.simulation_data,
                    self.environment
                )

                # -------------------------------------------------
                # Advance spacecraft clock
                # -------------------------------------------------

                t_spacecraft = spacecraft.spacecraft_data.t

                spacecraft.spacecraft_data.t += (
                    spacecraft.spacecraft_data.current_master_dt
                )

                spacecraft.spacecraft_data.tick += 1

                current_ts.append(t_spacecraft)

                # -------------------------------------------------
                # History
                # -------------------------------------------------

                timed_call(
                    "history_recording",
                    self.simulation_data.simulation_history.record_spacecraft,
                    spacecraft.name,
                    spacecraft.spacecraft_data
                )

            # -----------------------------------------------------
            # Update global simulation time
            # -----------------------------------------------------

            try:
                t = min(current_ts)
            except ValueError:
                break

            current_ts = []

            # -----------------------------------------------------
            # Loop overhead
            # -----------------------------------------------------

            if profile:
                iteration_elapsed = time.perf_counter() - iteration_start

                profile_data["simulation_iteration"]["total_time"] += (
                    iteration_elapsed
                )

                profile_data["simulation_iteration"]["calls"] += 1

        # ---------------------------------------------------------
        # Finalization
        # ---------------------------------------------------------

        simulation_time = time.perf_counter() - total_start

        if self.verbose:
            print("Simulation completed.")

        metadata = self.simulation_data.simulation_history.finalize()

        # ---------------------------------------------------------
        # Profiling report
        # ---------------------------------------------------------

        if profile:
            print("\n" + "=" * 70)
            print("SIMULATION PERFORMANCE PROFILE")
            print("=" * 70)

            print(f"\nWall-clock simulation time: {simulation_time:.6f} s")
            print(
                f"Simulated time: "
                f"{self.simulation_data.max_sim_time:.6f} s"
            )

            speed_ratio = (
                self.simulation_data.max_sim_time /
                simulation_time
            )

            print(
                f"Simulation speed: "
                f"{speed_ratio:.3f}x real-time"
            )

            print("\nModule timing:")
            print("-" * 70)

            print(
                f"{'Module':<30}"
                f"{'Calls':>12}"
                f"{'Total [s]':>15}"
                f"{'Avg [ms]':>15}"
                f"{'% total':>12}"
            )

            print("-" * 70)

            # Do not use simulation_iteration for the module table
            module_items = [
                (name, data)
                for name, data in profile_data.items()
                if name != "simulation_iteration"
            ]

            # Sort by total execution time
            module_items.sort(
                key=lambda item: item[1]["total_time"],
                reverse=True
            )

            for name, data in module_items:

                total_time = data["total_time"]
                calls = data["calls"]

                avg_time = (
                    total_time / calls
                    if calls > 0
                    else 0.0
                )

                percentage = (
                    100.0 * total_time / simulation_time
                    if simulation_time > 0
                    else 0.0
                )

                print(
                    f"{name:<30}"
                    f"{calls:>12,}"
                    f"{total_time:>15.6f}"
                    f"{avg_time * 1000:>15.6f}"
                    f"{percentage:>11.2f}%"
                )

            print("-" * 70)

            # -----------------------------------------------------
            # Profiling consistency check
            # -----------------------------------------------------

            measured_module_time = sum(
                data["total_time"]
                for name, data in module_items
            )

            print(
                f"\nMeasured module time: "
                f"{measured_module_time:.6f} s"
            )

            print(
                f"Unaccounted time: "
                f"{simulation_time - measured_module_time:.6f} s"
            )

            print("=" * 70)


            for sc in self.simulation_data.spacecrafts:
                if sc.mission_manager.current_phase.translational_model is not None:
                    sc.mission_manager.current_phase.translational_model.print_profile()
                if sc.mission_manager.current_phase.rotational_model is not None:
                    sc.mission_manager.current_phase.rotational_model.print_profile()

        return metadata    