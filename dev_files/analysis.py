# %% [markdown]
# # Example 3 — CR3BP Trajectory Propagation Analysis
#
# This analysis compares the native Python RK45 implementation against
# the native C++ RK45 implementation for the same CR3BP trajectory.
#
# The simulation configuration is kept identical between both cases.
#
# The analysis includes:
# - Wall-clock simulation time
# - Translational propagation time
# - Solver statistics
# - Function evaluations
# - Accepted/rejected integration steps
# - Final-state comparison
# - 3D trajectory comparison
# - Position, velocity and acceleration errors
# - Statistical boxplots
#
# The two simulations are executed independently to avoid interactions
# between solver statistics or simulation state.


# %% [markdown]
# ## Imports

# %%
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# %% [markdown]
# ## CR3BP Parameters

# %%
MU = 1.215e-2

TIME_FACTOR_SEC = (
    27.321661 / (2.0 * np.pi) * 24 * 3600
)

LENGTH_FACTOR = 384400.0e3


# %% [markdown]
# ## Simulation Parameters

# %%
MAX_SIM_TIME = 0.75952417 * 2 * TIME_FACTOR_SEC * 1

DT = 5e-4 * TIME_FACTOR_SEC

INITIAL_POSITION = (
    np.array([1.02262383, 0.0, -0.18250869])
    * LENGTH_FACTOR
)

INITIAL_VELOCITY = (
    np.array([0.0, -0.10456466, 0.0])
    * LENGTH_FACTOR
    / TIME_FACTOR_SEC
)


# %% [markdown]
# ## Utility Functions
#
# The simulation history is stored in NPZ chunks. These functions reconstruct
# the complete trajectory and extract the final state from the simulation
# metadata returned by `Simulation.simulate()`.


# %%
def load_spacecraft_history(
    history_metadata,
    spacecraft_name="SC",
):
    """
    Load the complete spacecraft history from all NPZ chunks.

    Parameters
    ----------
    history_metadata
        SimulationHistoryMetadata returned by sim.simulate().

    spacecraft_name : str
        Name of the spacecraft.

    Returns
    -------
    dict
        Dictionary containing time, position, velocity and acceleration.
    """

    history_path = Path(history_metadata.data_file_path)

    chunk_count = history_metadata.chunk_count[spacecraft_name]

    if chunk_count == 0:
        raise RuntimeError(
            f"No history chunks found for spacecraft '{spacecraft_name}'."
        )

    t = []
    position = []
    velocity = []
    acceleration = []

    for chunk_index in range(chunk_count):

        chunk_path = history_path / (
            f"{spacecraft_name}_history_chunk_{chunk_index}.npz"
        )

        if not chunk_path.exists():
            raise FileNotFoundError(
                f"History chunk not found: {chunk_path}"
            )

        with np.load(chunk_path) as data:

            t.append(data["t"].copy())
            position.append(data["true_position"].copy())
            velocity.append(data["true_velocity"].copy())
            acceleration.append(
                data["true_acceleration"].copy()
            )

    return {
        "t": np.concatenate(t),
        "position": np.concatenate(position),
        "velocity": np.concatenate(velocity),
        "acceleration": np.concatenate(acceleration),
    }


# %%
def load_final_state(
    history_metadata,
    spacecraft_name="SC",
):
    """
    Load the final translational state from the simulation history.
    """

    history_path = Path(history_metadata.data_file_path)

    chunk_count = history_metadata.chunk_count[spacecraft_name]

    if chunk_count == 0:
        raise RuntimeError(
            f"No history chunks found for spacecraft '{spacecraft_name}'."
        )

    last_chunk_index = chunk_count - 1

    chunk_path = history_path / (
        f"{spacecraft_name}_history_chunk_{last_chunk_index}.npz"
    )

    with np.load(chunk_path) as data:

        return {
            "t": float(data["t"][-1]),
            "position": data["true_position"][-1].copy(),
            "velocity": data["true_velocity"][-1].copy(),
            "acceleration": data["true_acceleration"][-1].copy(),
        }


# %% [markdown]
# ## Solver Statistics

# %%
from argos.solvers.solvers_base import (
    reset_solver_statistics,
    get_python_rk45_statistics,
    get_cpp_rk45_statistics,
)


# %% [markdown]
# ## Simulation Factory
#
# A fresh simulation is created for each integration method.
#
# This is important because the Python and C++ cases must not share the same
# spacecraft, mission manager, propagator or solver state.


# %%
def create_simulation(integration_method):
    """
    Create a fresh Example 3 simulation.

    Parameters
    ----------
    integration_method : str
        Either "NATIVE_RK45" or "CPP_RK45".

    Returns
    -------
    Simulation
    """

    from argos.enviroments.environments import ClassicalEnvironment
    from argos.core.simulation import Simulation
    from argos.propagators.native_propagator import (
        NativeTranslationalPropagator,
    )
    from argos.sensors.generic_sensor import AbsoluteSensor
    from argos.navigation.basic_laws import IdealNavigation
    from argos.general.dataclasses import MissionPhase
    from argos.core.mission_manager import MissionManager

    # Environment
    env = ClassicalEnvironment()

    # Simulation
    sim = Simulation(
        max_sim_time=MAX_SIM_TIME,
        environment=env,
        verbose=True,
    )

    # Translational propagator
    translational_propagator = NativeTranslationalPropagator(
        orbital_model="CR3BP",
        integration_method=integration_method,
    )

    # Sensor
    sensors = [
        AbsoluteSensor(
            1 / DT,
            verbose=False,
        )
    ]

    # Navigation
    nav_law = IdealNavigation()

    # Mission phase
    phase = MissionPhase(
        name="Phase 1",
        navigation=nav_law,
        dt_nav=DT,
        dt_propagation=DT,
        translational_model=translational_propagator,
    )

    # Mission manager
    mission_manager = MissionManager(
        initial_phase=phase
    )

    # Spacecraft
    sim.add_spacecraft(
        name="SC",
        initial_position=INITIAL_POSITION.copy(),
        initial_velocity=INITIAL_VELOCITY.copy(),
        sensors=sensors,
        mission_manager=mission_manager,
    )

    return sim


# %% [markdown]
# ## Run Python RK45 Simulation

# %%
print("=" * 80)
print("RUNNING PYTHON RK45")
print("=" * 80)

reset_solver_statistics()

sim_python = create_simulation("NATIVE_RK45")

python_wall_start = time.perf_counter()

python_result = sim_python.simulate()

python_wall_time = (
    time.perf_counter()
    - python_wall_start
)

python_statistics = get_python_rk45_statistics()

print()
print(f"Wall-clock simulation time: {python_wall_time:.6f} s")


# %% [markdown]
# ## Load Python Results

# %%
python_history = load_spacecraft_history(
    python_result,
    spacecraft_name="SC",
)

python_final_state = load_final_state(
    python_result,
    spacecraft_name="SC",
)

print(
    f"Final simulation time: "
    f"{python_final_state['t']:.12f} s"
)

print(
    "Final position:",
    python_final_state["position"],
)

print(
    "Final velocity:",
    python_final_state["velocity"],
)

print(
    "Final acceleration:",
    python_final_state["acceleration"],
)


# %% [markdown]
# ## Run C++ RK45 Simulation

# %%
print()
print("=" * 80)
print("RUNNING C++ RK45")
print("=" * 80)

reset_solver_statistics()

sim_cpp = create_simulation("CPP_RK45")

cpp_wall_start = time.perf_counter()

cpp_result = sim_cpp.simulate()

cpp_wall_time = (
    time.perf_counter()
    - cpp_wall_start
)

cpp_statistics = get_cpp_rk45_statistics()

print()
print(f"Wall-clock simulation time: {cpp_wall_time:.6f} s")


# %% [markdown]
# ## Load C++ Results

# %%
cpp_history = load_spacecraft_history(
    cpp_result,
    spacecraft_name="SC",
)

cpp_final_state = load_final_state(
    cpp_result,
    spacecraft_name="SC",
)

print(
    f"Final simulation time: "
    f"{cpp_final_state['t']:.12f} s"
)

print(
    "Final position:",
    cpp_final_state["position"],
)

print(
    "Final velocity:",
    cpp_final_state["velocity"],
)

print(
    "Final acceleration:",
    cpp_final_state["acceleration"],
)


# %% [markdown]
# # Solver Statistics
#
# The statistics are accumulated over every translational propagation call
# performed during the simulation.


# %%
def print_solver_statistics(
    name,
    statistics,
):
    nfev = np.asarray(
        statistics["nfev"],
        dtype=float,
    )

    accepted = np.asarray(
        statistics["accepted_steps"],
        dtype=float,
    )

    rejected = np.asarray(
        statistics["rejected_steps"],
        dtype=float,
    )

    elapsed = np.asarray(
        statistics["elapsed_time"],
        dtype=float,
    )

    print()
    print("=" * 80)
    print(f"{name} RK45 STATISTICS")
    print("=" * 80)

    print(
        f"Propagation calls: "
        f"{len(nfev):,}"
    )

    print(
        f"Total function evaluations: "
        f"{int(np.sum(nfev)):,}"
    )

    print(
        f"Total accepted steps: "
        f"{int(np.sum(accepted)):,}"
    )

    print(
        f"Total rejected steps: "
        f"{int(np.sum(rejected)):,}"
    )

    print(
        f"Total measured RK45 time: "
        f"{np.sum(elapsed):.6f} s"
    )

    if len(nfev) > 0:

        print(
            f"Average function evaluations / propagation: "
            f"{np.mean(nfev):.6f}"
        )

        print(
            f"Average accepted steps / propagation: "
            f"{np.mean(accepted):.6f}"
        )

        print(
            f"Average rejected steps / propagation: "
            f"{np.mean(rejected):.6f}"
        )

        print(
            f"Average RK45 time / propagation: "
            f"{np.mean(elapsed):.9f} s"
        )


# %%
print_solver_statistics(
    "Python",
    python_statistics,
)

print_solver_statistics(
    "C++",
    cpp_statistics,
)


# %% [markdown]
# # Performance Comparison


# %%
def compare_statistics(
    python_statistics,
    cpp_statistics,
):
    py_nfev = np.asarray(
        python_statistics["nfev"],
        dtype=float,
    )

    cpp_nfev = np.asarray(
        cpp_statistics["nfev"],
        dtype=float,
    )

    py_accepted = np.asarray(
        python_statistics["accepted_steps"],
        dtype=float,
    )

    cpp_accepted = np.asarray(
        cpp_statistics["accepted_steps"],
        dtype=float,
    )

    py_rejected = np.asarray(
        python_statistics["rejected_steps"],
        dtype=float,
    )

    cpp_rejected = np.asarray(
        cpp_statistics["rejected_steps"],
        dtype=float,
    )

    py_elapsed = np.asarray(
        python_statistics["elapsed_time"],
        dtype=float,
    )

    cpp_elapsed = np.asarray(
        cpp_statistics["elapsed_time"],
        dtype=float,
    )

    print()
    print("=" * 80)
    print("PERFORMANCE COMPARISON")
    print("=" * 80)

    print(
        f"Function evaluations: "
        f"{int(np.sum(py_nfev)):,} Python vs "
        f"{int(np.sum(cpp_nfev)):,} C++"
    )

    if np.sum(py_nfev) > 0:
        print(
            f"Function-evaluation ratio "
            f"(C++ / Python): "
            f"{np.sum(cpp_nfev) / np.sum(py_nfev):.3f}x"
        )

    print(
        f"Accepted steps: "
        f"{int(np.sum(py_accepted)):,} Python vs "
        f"{int(np.sum(cpp_accepted)):,} C++"
    )

    print(
        f"Rejected steps: "
        f"{int(np.sum(py_rejected)):,} Python vs "
        f"{int(np.sum(cpp_rejected)):,} C++"
    )

    print(
        f"Measured RK45 time: "
        f"{np.sum(py_elapsed):.6f} s Python vs "
        f"{np.sum(cpp_elapsed):.6f} s C++"
    )

    if np.sum(cpp_elapsed) > 0:

        print(
            f"RK45 time ratio "
            f"(C++ / Python): "
            f"{np.sum(cpp_elapsed) / np.sum(py_elapsed):.3f}x"
        )

        print(
            f"RK45 speedup "
            f"(Python / C++): "
            f"{np.sum(py_elapsed) / np.sum(cpp_elapsed):.3f}x"
        )

    print()
    print(
        f"Wall-clock simulation time: "
        f"{python_wall_time:.6f} s Python vs "
        f"{cpp_wall_time:.6f} s C++"
    )

    if cpp_wall_time > 0:

        print(
            f"Wall-clock ratio "
            f"(C++ / Python): "
            f"{cpp_wall_time / python_wall_time:.3f}x"
        )

        print(
            f"Wall-clock speedup "
            f"(Python / C++): "
            f"{python_wall_time / cpp_wall_time:.3f}x"
        )


# %%
compare_statistics(
    python_statistics,
    cpp_statistics,
)


# %% [markdown]
# # Final-State Comparison


# %%
def compare_final_states(
    python_final,
    cpp_final,
):
    print()
    print("=" * 80)
    print("FINAL STATE COMPARISON")
    print("=" * 80)

    print(
        f"Python final time: "
        f"{python_final['t']:.12f} s"
    )

    print(
        f"C++ final time:    "
        f"{cpp_final['t']:.12f} s"
    )

    print()

    position_error = (
        cpp_final["position"]
        - python_final["position"]
    )

    velocity_error = (
        cpp_final["velocity"]
        - python_final["velocity"]
    )

    acceleration_error = (
        cpp_final["acceleration"]
        - python_final["acceleration"]
    )

    print("Position [m]")
    print(
        "Python:",
        np.array2string(
            python_final["position"],
            precision=12,
        ),
    )

    print(
        "C++:   ",
        np.array2string(
            cpp_final["position"],
            precision=12,
        ),
    )

    print(
        "Error: ",
        np.array2string(
            position_error,
            precision=12,
        ),
    )

    print(
        "Max abs error:",
        np.max(np.abs(position_error)),
        "m",
    )

    print()

    print("Velocity [m/s]")
    print(
        "Python:",
        np.array2string(
            python_final["velocity"],
            precision=12,
        ),
    )

    print(
        "C++:   ",
        np.array2string(
            cpp_final["velocity"],
            precision=12,
        ),
    )

    print(
        "Error: ",
        np.array2string(
            velocity_error,
            precision=12,
        ),
    )

    print(
        "Max abs error:",
        np.max(np.abs(velocity_error)),
        "m/s",
    )

    print()

    print("Acceleration [m/s²]")
    print(
        "Python:",
        np.array2string(
            python_final["acceleration"],
            precision=12,
        ),
    )

    print(
        "C++:   ",
        np.array2string(
            cpp_final["acceleration"],
            precision=12,
        ),
    )

    print(
        "Error: ",
        np.array2string(
            acceleration_error,
            precision=12,
        ),
    )

    print(
        "Max abs error:",
        np.max(np.abs(acceleration_error)),
        "m/s²",
    )

    return {
        "position_error": position_error,
        "velocity_error": velocity_error,
        "acceleration_error": acceleration_error,
    }


# %%
state_errors = compare_final_states(
    python_final_state,
    cpp_final_state,
)


# %% [markdown]
# # Trajectory Comparison
#
# The complete trajectory is reconstructed from the simulation history chunks.
#
# Both trajectories are plotted in SI units.


# %%
def plot_trajectory_comparison(
    python_history,
    cpp_history,
):
    python_position = python_history["position"]
    cpp_position = cpp_history["position"]

    fig = plt.figure(
        figsize=(10, 8)
    )

    ax = fig.add_subplot(
        111,
        projection="3d",
    )

    ax.plot(
        python_position[:, 0],
        python_position[:, 1],
        python_position[:, 2],
        label="Python RK45",
    )

    ax.plot(
        cpp_position[:, 0],
        cpp_position[:, 1],
        cpp_position[:, 2],
        label="C++ RK45",
    )

    ax.scatter(
        python_position[0, 0],
        python_position[0, 1],
        python_position[0, 2],
        marker="o",
        label="Initial state",
    )

    ax.scatter(
        python_position[-1, 0],
        python_position[-1, 1],
        python_position[-1, 2],
        marker="x",
        label="Final Python",
    )

    ax.scatter(
        cpp_position[-1, 0],
        cpp_position[-1, 1],
        cpp_position[-1, 2],
        marker="x",
        label="Final C++",
    )

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")

    ax.set_title(
        "CR3BP trajectory comparison"
    )

    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.show()


# %%
plot_trajectory_comparison(
    python_history,
    cpp_history,
)


# %% [markdown]
# # Position Error Along the Trajectory
#
# Because the Python and C++ simulations may not contain exactly the same
# internal propagation timestamps, the trajectories are interpolated onto
# a common time grid before computing the error.


# %%
def interpolate_vector_history(
    time_source,
    values_source,
    time_target,
):
    """
    Interpolate a vector-valued history component by component.
    """

    values_source = np.asarray(
        values_source,
        dtype=float,
    )

    result = np.empty(
        (
            len(time_target),
            values_source.shape[1],
        ),
        dtype=float,
    )

    for component in range(
        values_source.shape[1]
    ):
        result[:, component] = np.interp(
            time_target,
            time_source,
            values_source[:, component],
        )

    return result


# %%
common_time = np.linspace(
    max(
        python_history["t"][0],
        cpp_history["t"][0],
    ),
    min(
        python_history["t"][-1],
        cpp_history["t"][-1],
    ),
    5000,
)

python_position_interp = interpolate_vector_history(
    python_history["t"],
    python_history["position"],
    common_time,
)

cpp_position_interp = interpolate_vector_history(
    cpp_history["t"],
    cpp_history["position"],
    common_time,
)

python_velocity_interp = interpolate_vector_history(
    python_history["t"],
    python_history["velocity"],
    common_time,
)

cpp_velocity_interp = interpolate_vector_history(
    cpp_history["t"],
    cpp_history["velocity"],
    common_time,
)

python_acceleration_interp = interpolate_vector_history(
    python_history["t"],
    python_history["acceleration"],
    common_time,
)

cpp_acceleration_interp = interpolate_vector_history(
    cpp_history["t"],
    cpp_history["acceleration"],
    common_time,
)


# %%
position_error_history = (
    cpp_position_interp
    - python_position_interp
)

velocity_error_history = (
    cpp_velocity_interp
    - python_velocity_interp
)

acceleration_error_history = (
    cpp_acceleration_interp
    - python_acceleration_interp
)

position_error_norm = np.linalg.norm(
    position_error_history,
    axis=1,
)

velocity_error_norm = np.linalg.norm(
    velocity_error_history,
    axis=1,
)

acceleration_error_norm = np.linalg.norm(
    acceleration_error_history,
    axis=1,
)


# %% [markdown]
# ## Position Error

# %%
plt.figure(
    figsize=(10, 5)
)

plt.plot(
    common_time,
    position_error_norm,
)

plt.xlabel("Time [s]")
plt.ylabel("Position error norm [m]")
plt.title(
    "Python vs C++ position error"
)

plt.grid(True)
plt.tight_layout()
plt.show()


# %% [markdown]
# ## Velocity Error

# %%
plt.figure(
    figsize=(10, 5)
)

plt.plot(
    common_time,
    velocity_error_norm,
)

plt.xlabel("Time [s]")
plt.ylabel("Velocity error norm [m/s]")
plt.title(
    "Python vs C++ velocity error"
)

plt.grid(True)
plt.tight_layout()
plt.show()


# %% [markdown]
# ## Acceleration Error

# %%
plt.figure(
    figsize=(10, 5)
)

plt.plot(
    common_time,
    acceleration_error_norm,
)

plt.xlabel("Time [s]")
plt.ylabel("Acceleration error norm [m/s²]")
plt.title(
    "Python vs C++ acceleration error"
)

plt.grid(True)
plt.tight_layout()
plt.show()


# %% [markdown]
# # Solver Statistics Plots


# %%
def plot_statistics(
    python_statistics,
    cpp_statistics,
):
    py_nfev = np.asarray(
        python_statistics["nfev"],
        dtype=float,
    )

    cpp_nfev = np.asarray(
        cpp_statistics["nfev"],
        dtype=float,
    )

    py_accepted = np.asarray(
        python_statistics["accepted_steps"],
        dtype=float,
    )

    cpp_accepted = np.asarray(
        cpp_statistics["accepted_steps"],
        dtype=float,
    )

    py_rejected = np.asarray(
        python_statistics["rejected_steps"],
        dtype=float,
    )

    cpp_rejected = np.asarray(
        cpp_statistics["rejected_steps"],
        dtype=float,
    )

    py_elapsed = np.asarray(
        python_statistics["elapsed_time"],
        dtype=float,
    )

    cpp_elapsed = np.asarray(
        cpp_statistics["elapsed_time"],
        dtype=float,
    )

    # ------------------------------------------------------------------
    # Function evaluations
    # ------------------------------------------------------------------

    plt.figure(
        figsize=(10, 5)
    )

    plt.boxplot(
        [py_nfev, cpp_nfev],
        tick_labels=["Python", "C++"],
    )

    plt.ylabel(
        "Function evaluations / propagation"
    )

    plt.title(
        "Function evaluations per propagation"
    )

    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------------
    # Accepted steps
    # ------------------------------------------------------------------

    plt.figure(
        figsize=(10, 5)
    )

    plt.boxplot(
        [py_accepted, cpp_accepted],
        tick_labels=["Python", "C++"],
    )

    plt.ylabel(
        "Accepted steps / propagation"
    )

    plt.title(
        "Accepted integration steps per propagation"
    )

    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------------
    # Rejected steps
    # ------------------------------------------------------------------

    plt.figure(
        figsize=(10, 5)
    )

    plt.boxplot(
        [py_rejected, cpp_rejected],
        tick_labels=["Python", "C++"],
    )

    plt.ylabel(
        "Rejected steps / propagation"
    )

    plt.title(
        "Rejected integration steps per propagation"
    )

    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------------
    # RK45 elapsed time
    # ------------------------------------------------------------------

    plt.figure(
        figsize=(10, 5)
    )

    plt.boxplot(
        [py_elapsed, cpp_elapsed],
        tick_labels=["Python", "C++"],
    )

    plt.ylabel(
        "RK45 time / propagation [s]"
    )

    plt.title(
        "RK45 execution time per propagation"
    )

    plt.grid(True)

    plt.tight_layout()
    plt.show()


# %%
plot_statistics(
    python_statistics,
    cpp_statistics,
)


# %% [markdown]
# # Summary


# %%
py_total_nfev = np.sum(
    python_statistics["nfev"]
)

cpp_total_nfev = np.sum(
    cpp_statistics["nfev"]
)

py_total_accepted = np.sum(
    python_statistics["accepted_steps"]
)

cpp_total_accepted = np.sum(
    cpp_statistics["accepted_steps"]
)

py_total_rejected = np.sum(
    python_statistics["rejected_steps"]
)

cpp_total_rejected = np.sum(
    cpp_statistics["rejected_steps"]
)

py_solver_time = np.sum(
    python_statistics["elapsed_time"]
)

cpp_solver_time = np.sum(
    cpp_statistics["elapsed_time"]
)

position_max_error = np.max(
    np.abs(
        state_errors["position_error"]
    )
)

velocity_max_error = np.max(
    np.abs(
        state_errors["velocity_error"]
    )
)

acceleration_max_error = np.max(
    np.abs(
        state_errors["acceleration_error"]
    )
)


# %%
print()
print("=" * 80)
print("SUMMARY")
print("=" * 80)

print()
print("Performance")
print("-" * 80)

print(
    f"Python wall-clock time : "
    f"{python_wall_time:.6f} s"
)

print(
    f"C++ wall-clock time    : "
    f"{cpp_wall_time:.6f} s"
)

print(
    f"Wall-clock speedup     : "
    f"{python_wall_time / cpp_wall_time:.3f}x"
)

print()

print(
    f"Python RK45 time       : "
    f"{py_solver_time:.6f} s"
)

print(
    f"C++ RK45 time          : "
    f"{cpp_solver_time:.6f} s"
)

print(
    f"RK45 speedup           : "
    f"{py_solver_time / cpp_solver_time:.3f}x"
)

print()
print("Integration work")
print("-" * 80)

print(
    f"Python function evals  : "
    f"{int(py_total_nfev):,}"
)

print(
    f"C++ function evals     : "
    f"{int(cpp_total_nfev):,}"
)

print(
    f"Function eval ratio    : "
    f"{cpp_total_nfev / py_total_nfev:.3f}x"
)

print()

print(
    f"Python accepted steps  : "
    f"{int(py_total_accepted):,}"
)

print(
    f"C++ accepted steps     : "
    f"{int(cpp_total_accepted):,}"
)

print()

print(
    f"Python rejected steps  : "
    f"{int(py_total_rejected):,}"
)

print(
    f"C++ rejected steps     : "
    f"{int(cpp_total_rejected):,}"
)

print()
print("Final-state discrepancy")
print("-" * 80)

print(
    f"Maximum position error     : "
    f"{position_max_error:.6e} m"
)

print(
    f"Maximum velocity error     : "
    f"{velocity_max_error:.6e} m/s"
)

print(                                                                          
    f"Maximum acceleration error : "
    f"{acceleration_max_error:.6e} m/s²"
)


# %% [markdown]
# ## Interpretation
#
# The performance measurements should not be interpreted independently from
# the amount of numerical work performed by each solver.
#
# In particular, a difference in the number of function evaluations or
# accepted integration steps indicates that the two RK45 implementations may
# not be using identical step-size controllers or error estimators.
#
# Therefore, a lower execution time alone does not establish numerical
# equivalence between the implementations.
#
# The trajectory and final-state comparisons above are included to identify
# numerical discrepancies before using the C++ implementation as a
# performance-equivalent replacement for the Python implementation.