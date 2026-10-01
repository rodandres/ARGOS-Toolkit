# %% [markdown]
# # Example 3 — CR3BP Trajectory Propagation Analysis
#
# This analysis compares three RK45 implementations for the same CR3BP
# simulation configuration:
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
# - Independent acceleration verification
# - 3D trajectory comparison
# - Position, velocity and acceleration errors
# - Statistical boxplots
#
# The simulations are executed independently to avoid interactions between
# solver statistics or simulation state.
#
# The independent acceleration verification evaluates the Python and C++
# dynamics models on exactly the same state. This separates differences
# caused by the dynamics implementation from differences caused by the
# numerical integrators.


# %% [markdown]
# ## Imports

# %%
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from argos.propagators.dynamic_models import (
    cr3bp,
    newton,
)

from argos import _cpp


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
    np.array(
        [1.02262383, 0.0, -0.18250869]
    )
    * LENGTH_FACTOR
)

INITIAL_VELOCITY = (
    np.array(
        [0.0, -0.10456466, 0.0]
    )
    * LENGTH_FACTOR
    / TIME_FACTOR_SEC
)


# %% [markdown]
# ## Utility Functions
#
# The simulation history is stored in NPZ chunks. These functions reconstruct
# the complete trajectory and extract the final state.
#
# The applied force is loaded because the independent dynamics evaluation must
# use the same force as the simulation.


# %%
def load_spacecraft_history(
    history_metadata,
    spacecraft_name="SC",
):
    """
    Load the complete spacecraft history from all NPZ chunks.
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
    force = []

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

            force.append(
                data["current_force_exerted"].copy()
            )

    return {
        "t": np.concatenate(t),
        "position": np.concatenate(position),
        "velocity": np.concatenate(velocity),
        "acceleration": np.concatenate(acceleration),
        "force": np.concatenate(force),
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
            "force": data["current_force_exerted"][-1].copy(),
        }


# %% [markdown]
# ## Solver Statistics

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

if not callable(
    get_scipy_rk45_statistics
):
    get_scipy_rk45_statistics = None


# %% [markdown]
# ## Simulation Factory

# %%
def create_simulation(
    integration_method,
):
    """
    Create a fresh Example 3 simulation.
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

# %%
def run_simulation(
    name,
    integration_method,
    statistics_getter=None,
):
    """
    Execute one independent simulation.
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

    print(
        "Final applied force:",
        final_state["force"],
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



# %% [markdown]
# # Run SciPy RK45

# %%
scipy_run = run_simulation(
    name="SciPy RK45",
    integration_method="SCIPY_RK45",
    statistics_getter=(
        get_scipy_rk45_statistics
    ),
)


cpp_run = run_simulation(
    name="C++ RK45",
    integration_method="CPP_RK45",
    statistics_getter=(
        get_cpp_rk45_statistics
    ),
)

# %% [markdown]
# # Assign Results

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

all_runs = [
    python_run,
    cpp_run,
    scipy_run,
]


# %% [markdown]
# # Applied Force Verification
#
# The Example 3 configuration does not include a commanded translational
# control force. We verify this directly from the recorded histories instead
# of assuming it.
#
# If the force is zero, the CR3BP gravitational acceleration is independent
# of spacecraft mass. Therefore the direct dynamics comparison can use any
# positive diagnostic mass.
#
# No spacecraft-internal API is accessed to obtain the mass.


# %%
def verify_zero_applied_force(
    runs,
    tolerance=1e-12,
):
    """
    Verify that the recorded applied force is zero.
    """

    print()
    print("=" * 80)
    print("APPLIED FORCE CHECK")
    print("=" * 80)

    maximum_force = 0.0

    for run in runs:

        force = np.asarray(
            run["history"]["force"],
            dtype=float,
        )

        max_force = float(
            np.max(
                np.abs(force)
            )
        )

        maximum_force = max(
            maximum_force,
            max_force,
        )

        print(
            f"{run['name']:<20}: "
            f"maximum |F| = "
            f"{max_force:.12e} N"
        )

    if maximum_force > tolerance:

        raise RuntimeError(
            "The recorded applied force is non-zero. "
            "The acceleration verification cannot use an arbitrary "
            "diagnostic mass. Use the actual spacecraft mass instead."
        )

    print()
    print(
        "Applied force is zero within the diagnostic tolerance."
    )

    return maximum_force


# %%
MAX_RECORDED_FORCE = verify_zero_applied_force(
    all_runs
)


# %% [markdown]
# ## Diagnostic Mass
#
# Because the verified applied force is zero, the acceleration returned by
# Newtonian CR3BP dynamics is independent of spacecraft mass.
#
# This mass is therefore not an assumed spacecraft property. It is only a
# positive numerical argument required by the `newton()` and C++ interfaces.


# %%
DIAGNOSTIC_MASS = 1.0


# %% [markdown]
# # Solver Statistics

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

# %%
def get_statistics_totals(
    statistics,
):
    """
    Extract aggregate solver statistics.
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
print_performance_comparison(
    all_runs
)


# %% [markdown]
# # Final-State Comparison

# %%
def compare_two_final_states(
    state_a,
    state_b,
):
    """
    Compute component-wise errors between two final states.
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


# %%
python_cpp_errors = print_final_state_pair(
    "Python RK45",
    python_final_state,
    "C++ RK45",
    cpp_final_state,
)


# %%
python_scipy_errors = print_final_state_pair(
    "Python RK45",
    python_final_state,
    "SciPy RK45",
    scipy_final_state,
)


# %%
cpp_scipy_errors = print_final_state_pair(
    "C++ RK45",
    cpp_final_state,
    "SciPy RK45",
    scipy_final_state,
)


# %% [markdown]
# # Independent Dynamics Verification
#
# This section evaluates the two dynamic-model implementations on exactly the
# same state.
#
# Python:
#
#     newton(..., cr3bp, ...)
#
# C++:
#
#     _cpp.evaluate_translational_dynamics(
#         ...,
#         "NEWTON",
#         "CR3BP",
#         ...
#     )
#
# This comparison is independent of the RK45 integration.
#
# It is performed at:
#
#     1. The common initial state
#     2. The Python final state
#     3. The C++ final state
#     4. The SciPy final state


# %%
def evaluate_python_acceleration(
    t,
    position,
    velocity,
    mass,
    applied_force,
):
    """
    Evaluate acceleration using the same Python dynamics path used by
    NativeTranslationalPropagator.
    """

    state = np.concatenate(
        (
            np.asarray(
                position,
                dtype=float,
            ),
            np.asarray(
                velocity,
                dtype=float,
            ),
        )
    )

    applied_force = np.asarray(
        applied_force,
        dtype=float,
    )

    disturbance_force = np.zeros(
        3,
        dtype=float,
    )

    derivative = newton(
        float(t),
        state,
        float(mass),
        applied_force,
        disturbance_force,
        cr3bp,
        (
            MU,
            LENGTH_FACTOR,
            TIME_FACTOR_SEC,
        ),
    )

    derivative = np.asarray(
        derivative,
        dtype=float,
    )

    if derivative.shape != (6,):
        raise ValueError(
            "Python dynamics returned an unexpected derivative shape: "
            f"{derivative.shape}"
        )

    return derivative[3:6].copy()


# %%
def evaluate_cpp_acceleration(
    t,
    position,
    velocity,
    mass,
    applied_force,
):
    """
    Evaluate acceleration using the same C++ dynamics path used by
    NativeTranslationalPropagator.
    """

    state = np.concatenate(
        (
            np.asarray(
                position,
                dtype=float,
            ),
            np.asarray(
                velocity,
                dtype=float,
            ),
        )
    )

    applied_force = np.asarray(
        applied_force,
        dtype=float,
    )

    disturbance_force = np.zeros(
        3,
        dtype=float,
    )

    derivative = (
        _cpp.evaluate_translational_dynamics(
            state,
            float(t),
            float(mass),
            applied_force,
            disturbance_force,
            "NEWTON",
            "CR3BP",
            float(MU),
            float(LENGTH_FACTOR),
            float(TIME_FACTOR_SEC),
        )
    )

    derivative = np.asarray(
        derivative,
        dtype=float,
    )

    if derivative.shape != (6,):
        raise ValueError(
            "C++ dynamics returned an unexpected derivative shape: "
            f"{derivative.shape}"
        )

    return derivative[3:6].copy()


# %% [markdown]
# ## Same-State Python vs C++ Dynamics

# %%
def compare_dynamics_at_state(
    label,
    t,
    position,
    velocity,
    applied_force,
    mass=DIAGNOSTIC_MASS,
):
    """
    Evaluate Python and C++ dynamics on exactly the same state.
    """

    python_acceleration = (
        evaluate_python_acceleration(
            t=t,
            position=position,
            velocity=velocity,
            mass=mass,
            applied_force=applied_force,
        )
    )

    cpp_acceleration = (
        evaluate_cpp_acceleration(
            t=t,
            position=position,
            velocity=velocity,
            mass=mass,
            applied_force=applied_force,
        )
    )

    difference = (
        cpp_acceleration
        - python_acceleration
    )

    print()
    print("=" * 80)
    print(
        f"DYNAMICS COMPARISON — {label}"
    )
    print("=" * 80)

    print()
    print(
        f"t = {float(t):.12e} s"
    )

    print()
    print("Position [m]")
    print(
        np.array2string(
            np.asarray(
                position,
                dtype=float,
            ),
            precision=15,
        )
    )

    print()
    print("Velocity [m/s]")
    print(
        np.array2string(
            np.asarray(
                velocity,
                dtype=float,
            ),
            precision=15,
        )
    )

    print()
    print("Applied force [N]")
    print(
        np.array2string(
            np.asarray(
                applied_force,
                dtype=float,
            ),
            precision=15,
        )
    )

    print()
    print("Python acceleration [m/s²]")
    print(
        np.array2string(
            python_acceleration,
            precision=15,
        )
    )

    print()
    print("C++ acceleration [m/s²]")
    print(
        np.array2string(
            cpp_acceleration,
            precision=15,
        )
    )

    print()
    print("C++ - Python [m/s²]")
    print(
        np.array2string(
            difference,
            precision=15,
        )
    )

    print()
    print(
        "Max abs difference: "
        f"{np.max(np.abs(difference)):.12e} m/s²"
    )

    print(
        "L2 difference:       "
        f"{np.linalg.norm(difference):.12e} m/s²"
    )

    return {
        "python": python_acceleration,
        "cpp": cpp_acceleration,
        "difference": difference,
        "max_abs_difference": float(
            np.max(
                np.abs(difference)
            )
        ),
        "l2_difference": float(
            np.linalg.norm(difference)
        ),
    }


# %% [markdown]
# ## Common Initial State
#
# The initial state comes directly from the recorded Python history.
# The Python and C++ dynamics are then evaluated on that exact same state.


# %%
initial_t = float(
    python_history["t"][0]
)

initial_position = (
    python_history["position"][0]
)

initial_velocity = (
    python_history["velocity"][0]
)

initial_force = (
    python_history["force"][0]
)

initial_dynamics_check = (
    compare_dynamics_at_state(
        label="COMMON INITIAL STATE",
        t=initial_t,
        position=initial_position,
        velocity=initial_velocity,
        applied_force=initial_force,
    )
)


# %% [markdown]
# ## Final-State Acceleration Verification

# %%
def verify_final_acceleration(
    name,
    final_state,
    mass=DIAGNOSTIC_MASS,
):
    """
    Compare stored acceleration against both dynamic-model implementations.
    """

    t = float(
        final_state["t"]
    )

    position = np.asarray(
        final_state["position"],
        dtype=float,
    )

    velocity = np.asarray(
        final_state["velocity"],
        dtype=float,
    )

    stored_acceleration = np.asarray(
        final_state["acceleration"],
        dtype=float,
    )

    applied_force = np.asarray(
        final_state["force"],
        dtype=float,
    )

    python_acceleration = (
        evaluate_python_acceleration(
            t=t,
            position=position,
            velocity=velocity,
            mass=mass,
            applied_force=applied_force,
        )
    )

    cpp_acceleration = (
        evaluate_cpp_acceleration(
            t=t,
            position=position,
            velocity=velocity,
            mass=mass,
            applied_force=applied_force,
        )
    )

    stored_vs_python = (
        stored_acceleration
        - python_acceleration
    )

    stored_vs_cpp = (
        stored_acceleration
        - cpp_acceleration
    )

    cpp_vs_python = (
        cpp_acceleration
        - python_acceleration
    )

    print()
    print("=" * 80)
    print(
        f"ACCELERATION VERIFICATION — {name.upper()}"
    )
    print("=" * 80)

    print()
    print(
        f"Final time: "
        f"{t:.12f} s"
    )

    print(
        f"Diagnostic mass: "
        f"{mass:.12e} kg"
    )

    print()
    print("Stored acceleration [m/s²]")
    print(
        np.array2string(
            stored_acceleration,
            precision=15,
        )
    )

    print()
    print("Python dynamics acceleration [m/s²]")
    print(
        np.array2string(
            python_acceleration,
            precision=15,
        )
    )

    print()
    print("C++ dynamics acceleration [m/s²]")
    print(
        np.array2string(
            cpp_acceleration,
            precision=15,
        )
    )

    print()
    print("Stored - Python [m/s²]")
    print(
        np.array2string(
            stored_vs_python,
            precision=15,
        )
    )

    print(
        "Max abs:",
        f"{np.max(np.abs(stored_vs_python)):.12e}",
        "m/s²",
    )

    print()
    print("Stored - C++ [m/s²]")
    print(
        np.array2string(
            stored_vs_cpp,
            precision=15,
        )
    )

    print(
        "Max abs:",
        f"{np.max(np.abs(stored_vs_cpp)):.12e}",
        "m/s²",
    )

    print()
    print("C++ - Python [m/s²]")
    print(
        np.array2string(
            cpp_vs_python,
            precision=15,
        )
    )

    print(
        "Max abs:",
        f"{np.max(np.abs(cpp_vs_python)):.12e}",
        "m/s²",
    )

    return {
        "stored": stored_acceleration,
        "python": python_acceleration,
        "cpp": cpp_acceleration,
        "stored_vs_python": stored_vs_python,
        "stored_vs_cpp": stored_vs_cpp,
        "cpp_vs_python": cpp_vs_python,
    }


# %%
python_acceleration_check = (
    verify_final_acceleration(
        name="Python RK45",
        final_state=python_final_state,
    )
)


# %%
cpp_acceleration_check = (
    verify_final_acceleration(
        name="C++ RK45",
        final_state=cpp_final_state,
    )
)


# %%
scipy_acceleration_check = (
    verify_final_acceleration(
        name="SciPy RK45",
        final_state=scipy_final_state,
    )
)


# %% [markdown]
# # Cross-Backend Acceleration Matrix
#
# Every row evaluates both dynamics implementations on exactly the same
# state. This isolates the dynamic-model implementation from the numerical
# integration.


# %%
def evaluate_both_models_on_state(
    state,
    mass=DIAGNOSTIC_MASS,
):
    python_acceleration = (
        evaluate_python_acceleration(
            t=state["t"],
            position=state["position"],
            velocity=state["velocity"],
            mass=mass,
            applied_force=state["force"],
        )
    )

    cpp_acceleration = (
        evaluate_cpp_acceleration(
            t=state["t"],
            position=state["position"],
            velocity=state["velocity"],
            mass=mass,
            applied_force=state["force"],
        )
    )

    return {
        "python": python_acceleration,
        "cpp": cpp_acceleration,
        "difference": (
            cpp_acceleration
            - python_acceleration
        ),
    }


# %%
cross_backend_results = {
    "Python final state": (
        evaluate_both_models_on_state(
            python_final_state
        )
    ),
    "C++ final state": (
        evaluate_both_models_on_state(
            cpp_final_state
        )
    ),
    "SciPy final state": (
        evaluate_both_models_on_state(
            scipy_final_state
        )
    ),
}


# %%
print()
print("=" * 80)
print("CROSS-BACKEND ACCELERATION MATRIX")
print("=" * 80)

for state_name, values in (
    cross_backend_results.items()
):

    print()
    print(state_name)
    print("-" * 80)

    print(
        "Python dynamics:",
        np.array2string(
            values["python"],
            precision=15,
        ),
    )

    print(
        "C++ dynamics   :",
        np.array2string(
            values["cpp"],
            precision=15,
        ),
    )

    print(
        "C++ - Python   :",
        np.array2string(
            values["difference"],
            precision=15,
        ),
    )

    print(
        "Max abs difference:",
        f"{np.max(np.abs(values['difference'])):.12e}",
        "m/s²",
    )


# %% [markdown]
# # Trajectory Comparison

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

    if values_source.ndim != 2:
        raise ValueError(
            "values_source must be a 2D array."
        )

    if len(time_source) != len(
        values_source
    ):
        raise ValueError(
            "time_source and values_source "
            "must have the same length."
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

if common_time_end <= common_time_start:
    raise RuntimeError(
        "The three histories do not contain a common time interval."
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
#
# These are the accelerations recorded by each simulation. They are distinct
# from the direct dynamics-model comparison above.


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
#
# These errors compare the acceleration histories stored by each simulation.
# They should be interpreted together with the independent dynamics
# verification above.


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
    "Stored acceleration error comparison"
)

plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()


# %% [markdown]
# # Solver Statistics Plots

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

# %%
def print_pair_summary(
    name,
    errors,
):
    print(
        name
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


# ----------------------------------------------------------------------
# Independent dynamics-model discrepancy
# ----------------------------------------------------------------------

print()
print("Independent dynamics-model discrepancy")
print("-" * 80)

print(
    "Common initial state:"
)

print(
    f"  Python vs C++ max difference = "
    f"{initial_dynamics_check['max_abs_difference']:.6e} m/s²"
)

for state_name, values in (
    cross_backend_results.items()
):

    print(
        f"{state_name:<25}: "
        f"{np.max(np.abs(values['difference'])):.6e} m/s²"
    )


# ----------------------------------------------------------------------
# Stored acceleration consistency
# ----------------------------------------------------------------------

print()
print("Stored acceleration consistency")
print("-" * 80)

print(
    "Python RK45:"
)

print(
    f"  Stored vs Python dynamics = "
    f"{np.max(np.abs(python_acceleration_check['stored_vs_python'])):.6e} m/s²"
)

print(
    f"  Stored vs C++ dynamics    = "
    f"{np.max(np.abs(python_acceleration_check['stored_vs_cpp'])):.6e} m/s²"
)

print()
print(
    "C++ RK45:"
)

print(
    f"  Stored vs Python dynamics = "
    f"{np.max(np.abs(cpp_acceleration_check['stored_vs_python'])):.6e} m/s²"
)

print(
    f"  Stored vs C++ dynamics    = "
    f"{np.max(np.abs(cpp_acceleration_check['stored_vs_cpp'])):.6e} m/s²"
)

print()
print(
    "SciPy RK45:"
)

print(
    f"  Stored vs Python dynamics = "
    f"{np.max(np.abs(scipy_acceleration_check['stored_vs_python'])):.6e} m/s²"
)

print(
    f"  Stored vs C++ dynamics    = "
    f"{np.max(np.abs(scipy_acceleration_check['stored_vs_cpp'])):.6e} m/s²"
)


# %% [markdown]
# # Interpretation
#
# The results should be interpreted in two separate layers.
#
# First, the trajectory comparison measures the behavior of the complete
# numerical propagation:
#
#     dynamics + RK45 implementation + adaptive step control
#
# Second, the direct dynamics comparison evaluates:
#
#     Python newton/cr3bp
#             vs
#     C++ NEWTON/CR3BP
#
# on exactly the same state.
#
# Therefore:
#
# - If Python and C++ dynamics differ on the common initial state, the
#   discrepancy exists independently of the integrators.
#
# - If Python and C++ dynamics agree on the common initial state but diverge
#   on a final state, the difference may depend on the state reached by each
#   trajectory or on how the state is represented/evaluated.
#
# - If the stored acceleration agrees with the backend dynamics evaluated on
#   the same state, the stored acceleration path is internally consistent.
#
# - If the stored acceleration does not agree with its own backend dynamics,
#   there is an issue in the acceleration calculation/storage path.
#
# - A difference between final accelerations alone does not prove that the
#   dynamic models differ, because the final positions and velocities are
#   also slightly different.
#
# Wall-clock time measures practical execution cost.
#
# Function evaluations indicate computational work when solver-level
# statistics are available.
#
# Accepted and rejected steps characterize adaptive integration behavior.
#
# Final-state errors quantify accumulated numerical differences.
#
# The stored acceleration trajectory comparison describes the acceleration
# histories produced by the simulations.
#
# The direct same-state dynamics comparison is the diagnostic used to
# determine whether the Python and C++ dynamics implementations themselves
# produce different accelerations.