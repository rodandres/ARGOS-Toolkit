# %% [markdown]
# # Example 3 — CR3BP Trajectory Propagation Analysis
#
# This analysis compares three RK45 implementations for the same CR3BP
# trajectory:
#
#     1. Native Python RK45
#     2. Native C++ RK45
#     3. SciPy RK45
#
# The simulation configuration is kept identical between all cases.
#
# The analysis includes:
# - Wall-clock simulation time
# - Translational propagation time
# - Solver statistics where available
# - Function evaluations
# - Accepted/rejected integration steps
# - Final-state comparison
# - 3D trajectory comparison
# - Position, velocity and acceleration errors
# - Statistical boxplots
#
# The simulations are executed independently to avoid interactions between
# solver statistics or simulation state.


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
#
# The simulation duration corresponds approximately to one revolution for the
# selected initial condition.


# %%
MAX_SIM_TIME = 0.75952417 * 2 * TIME_FACTOR_SEC

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
# the complete trajectory and extract the final state.


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

    history_path = Path(
        history_metadata.data_file_path
    )

    chunk_count = history_metadata.chunk_count[
        spacecraft_name
    ]

    if chunk_count == 0:
        raise RuntimeError(
            f"No history chunks found for spacecraft "
            f"'{spacecraft_name}'."
        )

    t = []
    position = []
    velocity = []
    acceleration = []

    for chunk_index in range(chunk_count):

        chunk_path = history_path / (
            f"{spacecraft_name}_history_chunk_"
            f"{chunk_index}.npz"
        )

        if not chunk_path.exists():
            raise FileNotFoundError(
                f"History chunk not found: {chunk_path}"
            )

        with np.load(chunk_path) as data:

            t.append(
                data["t"].copy()
            )

            position.append(
                data["true_position"].copy()
            )

            velocity.append(
                data["true_velocity"].copy()
            )

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

    history_path = Path(
        history_metadata.data_file_path
    )

    chunk_count = history_metadata.chunk_count[
        spacecraft_name
    ]

    if chunk_count == 0:
        raise RuntimeError(
            f"No history chunks found for spacecraft "
            f"'{spacecraft_name}'."
        )

    last_chunk_index = chunk_count - 1

    chunk_path = history_path / (
        f"{spacecraft_name}_history_chunk_"
        f"{last_chunk_index}.npz"
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
#
# Python and C++ currently expose native RK45 statistics through the solver
# statistics interface.
#
# SciPy statistics are queried dynamically. This allows the analysis to run
# even if no dedicated SciPy statistics interface has been implemented yet.


# %%
from argos.solvers import solvers_base

reset_solver_statistics = (
    solvers_base.reset_solver_statistics
)

get_python_rk45_statistics = (
    solvers_base.get_python_rk45_statistics
)

get_cpp_rk45_statistics = (
    solvers_base.get_cpp_rk45_statistics
)

get_scipy_rk45_statistics = getattr(
    solvers_base,
    "get_scipy_rk45_statistics",
    None,
)


# %% [markdown]
# ## Simulation Factory
#
# A fresh simulation is created for every integration method.
#
# Supported methods:
#
#     NATIVE_RK45
#     CPP_RK45
#     SCIPY_RK45


# %%
def create_simulation(
    integration_method,
):
    """
    Create a fresh Example 3 simulation.

    Parameters
    ----------
    integration_method : str
        Integration backend:
        "NATIVE_RK45", "CPP_RK45", or "SCIPY_RK45".

    Returns
    -------
    Simulation
    """

    from argos.enviroments.environments import (
        ClassicalEnvironment,
    )

    from argos.core.simulation import (
        Simulation,
    )

    from argos.propagators.native_propagator import (
        NativeTranslationalPropagator,
    )

    from argos.sensors.generic_sensor import (
        AbsoluteSensor,
    )

    from argos.navigation.basic_laws import (
        IdealNavigation,
    )

    from argos.general.dataclasses import (
        MissionPhase,
    )

    from argos.core.mission_manager import (
        MissionManager,
    )

    # ------------------------------------------------------------------
    # Environment
    # ------------------------------------------------------------------

    env = ClassicalEnvironment()

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    sim = Simulation(
        max_sim_time=MAX_SIM_TIME,
        environment=env,
        verbose=True,
    )

    # ------------------------------------------------------------------
    # Translational propagator
    # ------------------------------------------------------------------

    translational_propagator = (
        NativeTranslationalPropagator(
            orbital_model="CR3BP",
            integration_method=integration_method,
        )
    )

    # ------------------------------------------------------------------
    # Sensor
    # ------------------------------------------------------------------

    sensors = [
        AbsoluteSensor(
            1 / DT,
            verbose=False,
        )
    ]

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    nav_law = IdealNavigation()

    # ------------------------------------------------------------------
    # Mission phase
    # ------------------------------------------------------------------

    phase = MissionPhase(
        name="Phase 1",
        navigation=nav_law,
        dt_nav=DT,
        dt_propagation=DT,
        translational_model=translational_propagator,
    )

    # ------------------------------------------------------------------
    # Mission manager
    # ------------------------------------------------------------------

    mission_manager = MissionManager(
        initial_phase=phase
    )

    # ------------------------------------------------------------------
    # Spacecraft
    # ------------------------------------------------------------------

    sim.add_spacecraft(
        name="SC",
        initial_position=INITIAL_POSITION.copy(),
        initial_velocity=INITIAL_VELOCITY.copy(),
        sensors=sensors,
        mission_manager=mission_manager,
    )

    return sim


# %% [markdown]
# ## Generic Simulation Runner
#
# This avoids duplicating the same execution logic for Python, C++, and SciPy.


# %%
def run_simulation(
    name,
    integration_method,
    statistics_getter=None,
):
    """
    Execute one independent simulation.

    Parameters
    ----------
    name : str
        Human-readable solver name.

    integration_method : str
        ARGOS integration method identifier.

    statistics_getter : callable or None
        Function used to retrieve solver statistics.

    Returns
    -------
    dict
        Simulation result, history, final state, wall-clock time,
        and solver statistics.
    """

    print()
    print("=" * 80)
    print(f"RUNNING {name.upper()}")
    print("=" * 80)

    reset_solver_statistics()

    sim = create_simulation(
        integration_method
    )

    wall_start = time.perf_counter()

    result = sim.simulate()

    wall_time = (
        time.perf_counter()
        - wall_start
    )

    history = load_spacecraft_history(
        result,
        spacecraft_name="SC",
    )

    final_state = load_final_state(
        result,
        spacecraft_name="SC",
    )

    statistics = None

    if statistics_getter is not None:
        statistics = statistics_getter()

    print()
    print(
        f"Wall-clock simulation time: "
        f"{wall_time:.6f} s"
    )

    print(
        f"Final simulation time: "
        f"{final_state['t']:.12f} s"
    )

    print(
        "Final position:",
        final_state["position"],
    )

    print(
        "Final velocity:",
        final_state["velocity"],
    )

    print(
        "Final acceleration:",
        final_state["acceleration"],
    )

    return {
        "name": name,
        "integration_method": integration_method,
        "result": result,
        "history": history,
        "final_state": final_state,
        "wall_time": wall_time,
        "statistics": statistics,
    }


# %% [markdown]
# # Run Python RK45


# %%
python_run = run_simulation(
    name="Python RK45",
    integration_method="NATIVE_RK45",
    statistics_getter=(
        get_python_rk45_statistics
    ),
)


# %% [markdown]
# # Run C++ RK45


# %%
cpp_run = run_simulation(
    name="C++ RK45",
    integration_method="CPP_RK45",
    statistics_getter=(
        get_cpp_rk45_statistics
    ),
)


# %% [markdown]
# # Run SciPy RK45
#
# The SciPy implementation is selected through:
#
#     integration_method="SCIPY_RK45"
#
# If a dedicated SciPy statistics getter has not yet been implemented,
# `statistics` will remain `None`. Wall-clock time and trajectory results are
# still fully available for comparison.


# %%
scipy_run = run_simulation(
    name="SciPy RK45",
    integration_method="SCIPY_RK45",
    statistics_getter=(
        get_scipy_rk45_statistics
    ),
)


# %% [markdown]
# # Assign Results
#
# These aliases keep the rest of the analysis readable.


# %%
python_history = python_run["history"]
cpp_history = cpp_run["history"]
scipy_history = scipy_run["history"]

python_final_state = python_run["final_state"]
cpp_final_state = cpp_run["final_state"]
scipy_final_state = scipy_run["final_state"]

python_wall_time = python_run["wall_time"]
cpp_wall_time = cpp_run["wall_time"]
scipy_wall_time = scipy_run["wall_time"]

python_statistics = python_run["statistics"]
cpp_statistics = cpp_run["statistics"]
scipy_statistics = scipy_run["statistics"]


# %% [markdown]
# # Solver Statistics
#
# This function supports both the current Python/C++ statistics structure and
# the optional SciPy statistics structure.


# %%
def print_solver_statistics(
    name,
    statistics,
):
    print()
    print("=" * 80)
    print(f"{name.upper()} RK45 STATISTICS")
    print("=" * 80)

    if statistics is None:

        print(
            "No solver statistics available."
        )

        return

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

print_solver_statistics(
    "SciPy",
    scipy_statistics,
)


# %% [markdown]
# # Performance Comparison
#
# Wall-clock time is available for all three implementations.
#
# Detailed RK45 timing is shown when solver-level statistics are available.


# %%
def get_statistics_totals(
    statistics,
):
    """
    Extract aggregate solver statistics.

    Returns None if statistics are unavailable.
    """

    if statistics is None:
        return None

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

    return {
        "nfev": np.sum(nfev),
        "accepted": np.sum(accepted),
        "rejected": np.sum(rejected),
        "elapsed": np.sum(elapsed),
    }


# %%
def print_performance_comparison(
    runs,
):
    print()
    print("=" * 80)
    print("PERFORMANCE COMPARISON")
    print("=" * 80)

    print()
    print("Wall-clock simulation time")
    print("-" * 80)

    for run in runs:

        print(
            f"{run['name']:<20}: "
            f"{run['wall_time']:.6f} s"
        )

    print()
    print("Relative wall-clock time")
    print("-" * 80)

    reference_time = runs[0]["wall_time"]

    for run in runs:

        ratio = (
            run["wall_time"]
            / reference_time
        )

        speedup = (
            reference_time
            / run["wall_time"]
        )

        print(
            f"{run['name']:<20}: "
            f"{ratio:.3f}x relative to "
            f"{runs[0]['name']} "
            f"| speedup = {speedup:.3f}x"
        )

    print()
    print("Solver-level statistics")
    print("-" * 80)

    for run in runs:

        totals = get_statistics_totals(
            run["statistics"]
        )

        if totals is None:

            print(
                f"{run['name']:<20}: "
                f"not available"
            )

            continue

        print(
            f"{run['name']:<20}: "
            f"{int(totals['nfev']):,} "
            f"function evaluations, "
            f"{int(totals['accepted']):,} "
            f"accepted steps, "
            f"{int(totals['rejected']):,} "
            f"rejected steps, "
            f"{totals['elapsed']:.6f} s RK45 time"
        )


# %%
all_runs = [
    python_run,
    cpp_run,
    scipy_run,
]

print_performance_comparison(
    all_runs
)


# %% [markdown]
# # Final-State Comparison
#
# The final states of all three implementations are compared pairwise.


# %%
def compare_two_final_states(
    state_a,
    state_b,
):
    """
    Compute component-wise and maximum absolute errors between two states.
    """

    position_error = (
        state_b["position"]
        - state_a["position"]
    )

    velocity_error = (
        state_b["velocity"]
        - state_a["velocity"]
    )

    acceleration_error = (
        state_b["acceleration"]
        - state_a["acceleration"]
    )

    return {
        "position_error": position_error,
        "velocity_error": velocity_error,
        "acceleration_error": acceleration_error,
    }


# %%
def print_final_state_pair(
    name_a,
    state_a,
    name_b,
    state_b,
):
    errors = compare_two_final_states(
        state_a,
        state_b,
    )

    print()
    print(
        f"{name_b} - {name_a}"
    )
    print("-" * 80)

    print(
        f"Final time difference: "
        f"{state_b['t'] - state_a['t']:.12e} s"
    )

    print()
    print("Position [m]")

    print(
        f"{name_a}:",
        np.array2string(
            state_a["position"],
            precision=12,
        ),
    )

    print(
        f"{name_b}:",
        np.array2string(
            state_b["position"],
            precision=12,
        ),
    )

    print(
        "Error:",
        np.array2string(
            errors["position_error"],
            precision=12,
        ),
    )

    print(
        "Max abs error:",
        f"{np.max(np.abs(errors['position_error'])):.12e}",
        "m",
    )

    print()
    print("Velocity [m/s]")

    print(
        f"{name_a}:",
        np.array2string(
            state_a["velocity"],
            precision=12,
        ),
    )

    print(
        f"{name_b}:",
        np.array2string(
            state_b["velocity"],
            precision=12,
        ),
    )

    print(
        "Error:",
        np.array2string(
            errors["velocity_error"],
            precision=12,
        ),
    )

    print(
        "Max abs error:",
        f"{np.max(np.abs(errors['velocity_error'])):.12e}",
        "m/s",
    )

    print()
    print("Acceleration [m/s²]")

    print(
        f"{name_a}:",
        np.array2string(
            state_a["acceleration"],
            precision=12,
        ),
    )

    print(
        f"{name_b}:",
        np.array2string(
            state_b["acceleration"],
            precision=12,
        ),
    )

    print(
        "Error:",
        np.array2string(
            errors["acceleration_error"],
            precision=12,
        ),
    )

    print(
        "Max abs error:",
        f"{np.max(np.abs(errors['acceleration_error'])):.12e}",
        "m/s²",
    )

    return errors


# %% [markdown]
# ## Python vs C++


# %%
python_cpp_errors = print_final_state_pair(
    "Python RK45",
    python_final_state,
    "C++ RK45",
    cpp_final_state,
)


# %% [markdown]
# ## Python vs SciPy


# %%
python_scipy_errors = print_final_state_pair(
    "Python RK45",
    python_final_state,
    "SciPy RK45",
    scipy_final_state,
)


# %% [markdown]
# ## C++ vs SciPy


# %%
cpp_scipy_errors = print_final_state_pair(
    "C++ RK45",
    cpp_final_state,
    "SciPy RK45",
    scipy_final_state,
)


# %% [markdown]
# # Trajectory Comparison
#
# All three trajectories are plotted together.


# %%
def plot_trajectory_comparison(
    python_history,
    cpp_history,
    scipy_history,
):
    python_position = (
        python_history["position"]
    )

    cpp_position = (
        cpp_history["position"]
    )

    scipy_position = (
        scipy_history["position"]
    )

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

    ax.plot(
        scipy_position[:, 0],
        scipy_position[:, 1],
        scipy_position[:, 2],
        label="SciPy RK45",
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

    ax.scatter(
        scipy_position[-1, 0],
        scipy_position[-1, 1],
        scipy_position[-1, 2],
        marker="x",
        label="Final SciPy",
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
    scipy_history,
)


# %% [markdown]
# # Common Time Grid
#
# The three trajectories may contain different internal propagation
# timestamps. They are therefore interpolated onto a common time grid before
# computing trajectory errors.


# %%
def interpolate_vector_history(
    time_source,
    values_source,
    time_target,
):
    """
    Interpolate a vector-valued history component by component.
    """

    time_source = np.asarray(
        time_source,
        dtype=float,
    )

    values_source = np.asarray(
        values_source,
        dtype=float,
    )

    time_target = np.asarray(
        time_target,
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
common_time_start = max(
    python_history["t"][0],
    cpp_history["t"][0],
    scipy_history["t"][0],
)

common_time_end = min(
    python_history["t"][-1],
    cpp_history["t"][-1],
    scipy_history["t"][-1],
)

common_time = np.linspace(
    common_time_start,
    common_time_end,
    5000,
)


# %% [markdown]
# ## Position


# %%
python_position_interp = (
    interpolate_vector_history(
        python_history["t"],
        python_history["position"],
        common_time,
    )
)

cpp_position_interp = (
    interpolate_vector_history(
        cpp_history["t"],
        cpp_history["position"],
        common_time,
    )
)

scipy_position_interp = (
    interpolate_vector_history(
        scipy_history["t"],
        scipy_history["position"],
        common_time,
    )
)


# %% [markdown]
# ## Velocity


# %%
python_velocity_interp = (
    interpolate_vector_history(
        python_history["t"],
        python_history["velocity"],
        common_time,
    )
)

cpp_velocity_interp = (
    interpolate_vector_history(
        cpp_history["t"],
        cpp_history["velocity"],
        common_time,
    )
)

scipy_velocity_interp = (
    interpolate_vector_history(
        scipy_history["t"],
        scipy_history["velocity"],
        common_time,
    )
)


# %% [markdown]
# ## Acceleration


# %%
python_acceleration_interp = (
    interpolate_vector_history(
        python_history["t"],
        python_history["acceleration"],
        common_time,
    )
)

cpp_acceleration_interp = (
    interpolate_vector_history(
        cpp_history["t"],
        cpp_history["acceleration"],
        common_time,
    )
)

scipy_acceleration_interp = (
    interpolate_vector_history(
        scipy_history["t"],
        scipy_history["acceleration"],
        common_time,
    )
)


# %% [markdown]
# # Trajectory Errors
#
# Python is used as the reference trajectory for the primary error plots.
#
# Additional C++ vs SciPy errors are also computed.


# %%
python_cpp_position_error = (
    cpp_position_interp
    - python_position_interp
)

python_scipy_position_error = (
    scipy_position_interp
    - python_position_interp
)

cpp_scipy_position_error = (
    scipy_position_interp
    - cpp_position_interp
)


# %%
python_cpp_velocity_error = (
    cpp_velocity_interp
    - python_velocity_interp
)

python_scipy_velocity_error = (
    scipy_velocity_interp
    - python_velocity_interp
)

cpp_scipy_velocity_error = (
    scipy_velocity_interp
    - cpp_velocity_interp
)


# %%
python_cpp_acceleration_error = (
    cpp_acceleration_interp
    - python_acceleration_interp
)

python_scipy_acceleration_error = (
    scipy_acceleration_interp
    - python_acceleration_interp
)

cpp_scipy_acceleration_error = (
    scipy_acceleration_interp
    - cpp_acceleration_interp
)


# %% [markdown]
# ## Error Norms


# %%
python_cpp_position_error_norm = np.linalg.norm(
    python_cpp_position_error,
    axis=1,
)

python_scipy_position_error_norm = np.linalg.norm(
    python_scipy_position_error,
    axis=1,
)

cpp_scipy_position_error_norm = np.linalg.norm(
    cpp_scipy_position_error,
    axis=1,
)


# %%
python_cpp_velocity_error_norm = np.linalg.norm(
    python_cpp_velocity_error,
    axis=1,
)

python_scipy_velocity_error_norm = np.linalg.norm(
    python_scipy_velocity_error,
    axis=1,
)

cpp_scipy_velocity_error_norm = np.linalg.norm(
    cpp_scipy_velocity_error,
    axis=1,
)


# %%
python_cpp_acceleration_error_norm = np.linalg.norm(
    python_cpp_acceleration_error,
    axis=1,
)

python_scipy_acceleration_error_norm = np.linalg.norm(
    python_scipy_acceleration_error,
    axis=1,
)

cpp_scipy_acceleration_error_norm = np.linalg.norm(
    cpp_scipy_acceleration_error,
    axis=1,
)


# %% [markdown]
# # Position Error Along the Trajectory


# %%
plt.figure(
    figsize=(10, 5)
)

plt.plot(
    common_time,
    python_cpp_position_error_norm,
    label="C++ vs Python",
)

plt.plot(
    common_time,
    python_scipy_position_error_norm,
    label="SciPy vs Python",
)

plt.plot(
    common_time,
    cpp_scipy_position_error_norm,
    label="SciPy vs C++",
)

plt.xlabel("Time [s]")
plt.ylabel("Position error norm [m]")
plt.title(
    "RK45 position error comparison"
)

plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()


# %% [markdown]
# # Velocity Error Along the Trajectory


# %%
plt.figure(
    figsize=(10, 5)
)

plt.plot(
    common_time,
    python_cpp_velocity_error_norm,
    label="C++ vs Python",
)

plt.plot(
    common_time,
    python_scipy_velocity_error_norm,
    label="SciPy vs Python",
)

plt.plot(
    common_time,
    cpp_scipy_velocity_error_norm,
    label="SciPy vs C++",
)

plt.xlabel("Time [s]")
plt.ylabel("Velocity error norm [m/s]")
plt.title(
    "RK45 velocity error comparison"
)

plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()


# %% [markdown]
# # Acceleration Error Along the Trajectory


# %%
plt.figure(
    figsize=(10, 5)
)

plt.plot(
    common_time,
    python_cpp_acceleration_error_norm,
    label="C++ vs Python",
)

plt.plot(
    common_time,
    python_scipy_acceleration_error_norm,
    label="SciPy vs Python",
)

plt.plot(
    common_time,
    cpp_scipy_acceleration_error_norm,
    label="SciPy vs C++",
)

plt.xlabel("Time [s]")
plt.ylabel("Acceleration error norm [m/s²]")
plt.title(
    "RK45 acceleration error comparison"
)

plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()


# %% [markdown]
# # Solver Statistics Plots
#
# Detailed solver statistics are plotted only for implementations for which
# the ARGOS statistics interface is available.


# %%
def plot_statistics(
    runs,
):
    available_runs = [
        run
        for run in runs
        if run["statistics"] is not None
    ]

    if not available_runs:

        print(
            "No solver statistics available "
            "for plotting."
        )

        return

    labels = [
        run["name"]
        for run in available_runs
    ]

    statistics = [
        run["statistics"]
        for run in available_runs
    ]

    # ------------------------------------------------------------------
    # Function evaluations
    # ------------------------------------------------------------------

    nfev = [
        np.asarray(
            stat["nfev"],
            dtype=float,
        )
        for stat in statistics
    ]

    plt.figure(
        figsize=(10, 5)
    )

    plt.boxplot(
        nfev,
        tick_labels=labels,
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

    accepted = [
        np.asarray(
            stat["accepted_steps"],
            dtype=float,
        )
        for stat in statistics
    ]

    plt.figure(
        figsize=(10, 5)
    )

    plt.boxplot(
        accepted,
        tick_labels=labels,
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

    rejected = [
        np.asarray(
            stat["rejected_steps"],
            dtype=float,
        )
        for stat in statistics
    ]

    plt.figure(
        figsize=(10, 5)
    )

    plt.boxplot(
        rejected,
        tick_labels=labels,
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

    elapsed = [
        np.asarray(
            stat["elapsed_time"],
            dtype=float,
        )
        for stat in statistics
    ]

    plt.figure(
        figsize=(10, 5)
    )

    plt.boxplot(
        elapsed,
        tick_labels=labels,
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
    all_runs
)


# %% [markdown]
# # Summary
#
# The summary reports:
# - Wall-clock performance
# - Solver-level performance when available
# - Integration work
# - Final-state discrepancies
#
# No numerical equivalence is assumed solely from execution time.


# %%
def print_pair_summary(
    name,
    errors,
):
    print(
        f"{name}"
    )

    print(
        f"  Maximum position error     : "
        f"{np.max(np.abs(errors['position_error'])):.6e} m"
    )

    print(
        f"  Maximum velocity error     : "
        f"{np.max(np.abs(errors['velocity_error'])):.6e} m/s"
    )

    print(
        f"  Maximum acceleration error : "
        f"{np.max(np.abs(errors['acceleration_error'])):.6e} m/s²"
    )


# %%
print()
print("=" * 80)
print("SUMMARY")
print("=" * 80)

# ----------------------------------------------------------------------
# Performance
# ----------------------------------------------------------------------

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
    f"SciPy wall-clock time  : "
    f"{scipy_wall_time:.6f} s"
)

print()

print(
    f"C++ / Python time ratio   : "
    f"{cpp_wall_time / python_wall_time:.3f}x"
)

print(
    f"SciPy / Python time ratio : "
    f"{scipy_wall_time / python_wall_time:.3f}x"
)

print()

print(
    f"C++ speedup vs Python     : "
    f"{python_wall_time / cpp_wall_time:.3f}x"
)

print(
    f"SciPy speedup vs Python   : "
    f"{python_wall_time / scipy_wall_time:.3f}x"
)


# ----------------------------------------------------------------------
# Solver-level statistics
# ----------------------------------------------------------------------

print()
print("Integration work")
print("-" * 80)

for run in all_runs:

    totals = get_statistics_totals(
        run["statistics"]
    )

    if totals is None:

        print(
            f"{run['name']:<20}: "
            "statistics not available"
        )

        continue

    print(
        f"{run['name']:<20}: "
        f"{int(totals['nfev']):,} function evaluations"
    )

    print(
        f"{'':<20}  "
        f"{int(totals['accepted']):,} accepted steps"
    )

    print(
        f"{'':<20}  "
        f"{int(totals['rejected']):,} rejected steps"
    )

    print(
        f"{'':<20}  "
        f"{totals['elapsed']:.6f} s RK45 time"
    )


# ----------------------------------------------------------------------
# Final-state discrepancies
# ----------------------------------------------------------------------

print()
print("Final-state discrepancy")
print("-" * 80)

print_pair_summary(
    "C++ vs Python",
    python_cpp_errors,
)

print()

print_pair_summary(
    "SciPy vs Python",
    python_scipy_errors,
)

print()

print_pair_summary(
    "SciPy vs C++",
    cpp_scipy_errors,
)


# %% [markdown]
# # Interpretation
#
# The three implementations should be evaluated using both computational
# cost and numerical behavior.
#
# A lower execution time does not by itself establish numerical equivalence.
# Differences in adaptive step-size control, error estimation, tolerance
# handling, or integration intervals can produce different numbers of
# internal steps and function evaluations.
#
# In particular, the native Python RK45 implementation currently uses the
# Fehlberg RK45 formulation implemented in ARGOS, while SciPy's RK45 solver
# has its own implementation and adaptive-step controller.
#
# Therefore:
#
# - Wall-clock time measures practical execution cost.
# - Function evaluations indicate computational work when available.
# - Accepted/rejected steps characterize adaptive integration behavior.
# - Final-state errors quantify accumulated numerical differences.
# - Trajectory errors show how those differences evolve throughout the
#   simulation.
#
# The SciPy results should therefore be interpreted as an independent RK45
# reference implementation rather than assumed to be numerically identical
# to either native implementation.