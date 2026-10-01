import time

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from argos.solvers.solver_utilities import OdeResult
from argos.solvers.solver_statistics import SolverStatistics
from argos.solvers.RK45 import rk45
from argos import _cpp


# ============================================================================
# Solver statistics
# ============================================================================

CPP_RK45_STATS = SolverStatistics()
PYTHON_RK45_STATS = SolverStatistics()


# ============================================================================
# Native Python RK45
# ============================================================================

def native_rk45(f, t_span, y0, **kwargs):
    """
    Integrate an ordinary differential equation using the native
    Python RK45 implementation.
    """

    start = time.perf_counter()

    result = rk45(
        f,
        t_span,
        y0,
        **kwargs,
    )

    elapsed = time.perf_counter() - start

    PYTHON_RK45_STATS.add(
        nfev=int(result.nfev),
        accepted_steps=int(result.accepted_steps),
        rejected_steps=int(result.rejected_steps),
        elapsed_time=elapsed,
    )

    return result


# ============================================================================
# SciPy RK45
# ============================================================================

def scipy_rk45(f, t_span, y0, **kwargs):
    """
    Integrate an ordinary differential equation using SciPy's RK45 solver.

    Parameters
    ----------
    f : callable
        Function defining the ordinary differential equation.

    t_span : array-like, shape (2,)
        Integration interval ``[t0, tf]`` [s].

    y0 : array-like
        Initial state vector.

    **kwargs
        Additional keyword arguments passed to
        :func:`scipy.integrate.solve_ivp`.

    Returns
    -------
    OdeResult
        Integration result returned by SciPy.
    """

    return solve_ivp(
        f,
        t_span,
        y0,
        method="RK45",
        **kwargs,
    )


# ============================================================================
# Native C++ RK45
# ============================================================================

def cpp_rk45(f, t_span, y0, **kwargs):
    """
    Integrate translational dynamics using the native C++ RK45 solver.

    The callable ``f`` is used only to identify the native dynamics model.
    The C++ backend evaluates the dynamics internally and does not call
    the Python callable during integration.

    Required ``args``
    -----------------
    args : tuple
        Must contain:

        (
            mass,
            applied_force,
            disturbance_force,
            orbital_model,
            orbital_model_arguments,
        )

    Optional RK45 parameters
    ------------------------
    h0 : float
        Initial integration step.

    h_min : float
        Minimum integration step.

    h_max : float
        Maximum integration step.

    rtol : float
        Relative tolerance.

    atol : float
        Absolute tolerance.

    h_adaptative : bool
        Whether adaptive step size is enabled.

    max_iter : int
        Maximum number of integration iterations.

    verbose : bool
        Enable solver verbosity.
    """

    # ------------------------------------------------------------------------
    # Dynamics model
    # ------------------------------------------------------------------------

    if not isinstance(f, str):
        raise TypeError(
            "CPP_RK45 requires the dynamics model to be specified as a string."
        )

    dynamics_model = str(f).upper()

    # ------------------------------------------------------------------------
    # Solver arguments
    # ------------------------------------------------------------------------

    args = kwargs.pop("args", ())

    if len(args) != 5:
        raise ValueError(
            "CPP_RK45 expects args to contain: "
            "(mass, applied_force, disturbance_force, "
            "orbital_model, orbital_model_arguments)."
        )

    (
        mass,
        applied_force,
        disturbance_force,
        orbital_model,
        orbital_model_arguments,
    ) = args

    if orbital_model_arguments is None:
        orbital_model_arguments = ()

    orbital_model_arguments = tuple(orbital_model_arguments)

    # Normalize model name once.
    orbital_model = (
        None
        if orbital_model is None
        else str(orbital_model).upper()
    )

    # ------------------------------------------------------------------------
    # Orbital model parameters
    # ------------------------------------------------------------------------

    if orbital_model == "REL2BP":

        if len(orbital_model_arguments) != 1:
            raise ValueError(
                "REL2BP requires one orbital model argument: mu."
            )

        mu = orbital_model_arguments[0]
        length_factor = 1.0
        time_factor = 1.0

    elif orbital_model == "CR3BP":

        if len(orbital_model_arguments) != 3:
            raise ValueError(
                "CR3BP requires three orbital model arguments: "
                "(mu, length_factor, time_factor)."
            )

        (
            mu,
            length_factor,
            time_factor,
        ) = orbital_model_arguments

    elif orbital_model in (None, ""):

        mu = 0.0
        length_factor = 1.0
        time_factor = 1.0

    else:

        raise ValueError(
            f"Unsupported orbital model: '{orbital_model}'."
        )

    # ------------------------------------------------------------------------
    # RK45 parameters
    # ------------------------------------------------------------------------

    h0 = kwargs.pop("h0", 0.1)
    h_min = kwargs.pop("h_min", 1e-6)
    rtol = kwargs.pop("rtol", 1e-6)
    atol = kwargs.pop("atol", 1e-9)
    adaptive = kwargs.pop("h_adaptative", True)
    max_iter = kwargs.pop("max_iter", 100000)
    verbose = kwargs.pop("verbose", False)

    # ------------------------------------------------------------------------
    # Unsupported arguments
    # ------------------------------------------------------------------------

    if kwargs:
        unsupported = ", ".join(sorted(kwargs))

        raise TypeError(
            f"cpp_rk45() got unsupported keyword argument(s): "
            f"{unsupported}"
        )

    # ------------------------------------------------------------------------
    # Convert inputs
    # ------------------------------------------------------------------------

    y0 = np.asarray(y0, dtype=float)
    applied_force = np.asarray(applied_force, dtype=float)
    disturbance_force = np.asarray(disturbance_force, dtype=float)
    t_span = np.asarray(t_span, dtype=float)

    t_span = np.asarray(t_span, dtype=float)
    
    h_max = kwargs.pop(
        "h_max",
        abs(t_span[1] - t_span[0]),
    )

    # ------------------------------------------------------------------------
    # C++ propagation
    # ------------------------------------------------------------------------

    start = time.perf_counter()

    result = _cpp.propagate_translational(
        state=y0,
        t0=float(t_span[0]),
        tf=float(t_span[1]),
        mass=float(mass),
        applied_force=applied_force,
        disturbance_force=disturbance_force,
        dynamics_model=dynamics_model,
        orbital_model=orbital_model or "",
        mu=float(mu),
        length_factor=float(length_factor),
        time_factor=float(time_factor),
        h0=float(h0),
        h_min=float(h_min),
        h_max=float(h_max),
        rtol=float(rtol),
        atol=float(atol),
        adaptive=bool(adaptive),
        max_iter=int(max_iter),
        verbose=bool(verbose),
    )

    elapsed = time.perf_counter() - start

    # ------------------------------------------------------------------------
    # Record solver statistics
    # ------------------------------------------------------------------------

    CPP_RK45_STATS.add(
        nfev=int(result["nfev"]),
        accepted_steps=int(result["accepted_steps"]),
        rejected_steps=int(result["rejected_steps"]),
        elapsed_time=elapsed,
    )

    # ------------------------------------------------------------------------
    # Convert C++ result to OdeResult
    # ------------------------------------------------------------------------

    return OdeResult(
        t=np.asarray(result["t"]),
        y=np.asarray(result["y"]),
        sol=None,
        t_events=result["t_events"],
        y_events=result["y_events"],
        nfev=int(result["nfev"]),
        njev=int(result["njev"]),
        nlu=int(result["nlu"]),
        status=int(result["status"]),
        message=str(result["message"]),
        success=bool(result["success"]),
        accepted_steps=int(result["accepted_steps"]),
        rejected_steps=int(result["rejected_steps"]),
    )


# ============================================================================
# Supported solvers
# ============================================================================

SUPPORTED_SOLVERS = {
    "NATIVE_RK45": native_rk45,
    "SCIPY_RK45": scipy_rk45,
    "CPP_RK45": cpp_rk45,
}


# ============================================================================
# Generic solver interface
# ============================================================================

def solve(f, t_span, y0, method, **kwargs):
    """
    Integrate an ordinary differential equation using a supported solver.

    Parameters
    ----------
    f : callable
        Function defining the ordinary differential equation.

    t_span : array-like, shape (2,)
        Integration interval ``[t0, tf]``.

    y0 : array-like
        Initial state vector.

    method : str
        Name of the numerical integration method.

    **kwargs
        Additional keyword arguments passed to the selected solver.

    Returns
    -------
    OdeResult
        Result returned by the selected numerical solver.

    Raises
    ------
    ValueError
        If the requested integration method is not supported.
    """

    try:
        solver = SUPPORTED_SOLVERS[method.upper()]

    except KeyError:
        raise ValueError(
            f"Método '{method}' no soportado. "
            f"Disponibles: {', '.join(SUPPORTED_SOLVERS)}"
        )

    return solver(
        f,
        t_span,
        y0,
        **kwargs,
    )


# ============================================================================
# Statistics management
# ============================================================================

def reset_solver_statistics():
    """
    Clear all recorded Python and C++ RK45 statistics.
    """

    PYTHON_RK45_STATS.nfev.clear()
    PYTHON_RK45_STATS.accepted_steps.clear()
    PYTHON_RK45_STATS.rejected_steps.clear()
    PYTHON_RK45_STATS.elapsed_time.clear()

    CPP_RK45_STATS.nfev.clear()
    CPP_RK45_STATS.accepted_steps.clear()
    CPP_RK45_STATS.rejected_steps.clear()
    CPP_RK45_STATS.elapsed_time.clear()


def get_python_rk45_statistics():
    """
    Return a copy of the recorded Python RK45 statistics.
    """

    return {
        "nfev": np.asarray(
            PYTHON_RK45_STATS.nfev,
            dtype=float,
        ).copy(),

        "accepted_steps": np.asarray(
            PYTHON_RK45_STATS.accepted_steps,
            dtype=float,
        ).copy(),

        "rejected_steps": np.asarray(
            PYTHON_RK45_STATS.rejected_steps,
            dtype=float,
        ).copy(),

        "elapsed_time": np.asarray(
            PYTHON_RK45_STATS.elapsed_time,
            dtype=float,
        ).copy(),
    }


def get_cpp_rk45_statistics():
    """
    Return a copy of the recorded C++ RK45 statistics.
    """

    return {
        "nfev": np.asarray(
            CPP_RK45_STATS.nfev,
            dtype=float,
        ).copy(),

        "accepted_steps": np.asarray(
            CPP_RK45_STATS.accepted_steps,
            dtype=float,
        ).copy(),

        "rejected_steps": np.asarray(
            CPP_RK45_STATS.rejected_steps,
            dtype=float,
        ).copy(),

        "elapsed_time": np.asarray(
            CPP_RK45_STATS.elapsed_time,
            dtype=float,
        ).copy(),
    }


# ============================================================================
# Comparison
# ============================================================================

def print_solver_comparison(python_stats, cpp_stats):
    """
    Print a comparison between Python and C++ RK45 statistics.
    """

    metrics = [
        ("nfev", "Function evaluations"),
        ("accepted_steps", "Accepted steps"),
        ("rejected_steps", "Rejected steps"),
        ("elapsed_time", "Elapsed time [s]"),
    ]

    print()
    print("=" * 90)
    print("RK45 SOLVER COMPARISON")
    print("=" * 90)

    print(
        f"{'Metric':<25}"
        f"{'Python':>15}"
        f"{'C++':>15}"
        f"{'C++ / Python':>18}"
    )

    print("-" * 90)

    # ------------------------------------------------------------------------
    # Total
    # ------------------------------------------------------------------------

    for key, label in metrics:

        py_total = np.sum(python_stats[key])
        cpp_total = np.sum(cpp_stats[key])

        ratio = (
            cpp_total / py_total
            if py_total != 0
            else np.nan
        )

        print(
            f"{label:<25}"
            f"{py_total:>15.3f}"
            f"{cpp_total:>15.3f}"
            f"{ratio:>18.3f}x"
        )

    # ------------------------------------------------------------------------
    # Mean
    # ------------------------------------------------------------------------

    print("-" * 90)

    print(
        f"{'Mean per propagation':<25}"
        f"{'Python':>15}"
        f"{'C++':>15}"
        f"{'C++ / Python':>18}"
    )

    for key, label in metrics:

        py_mean = np.mean(python_stats[key])
        cpp_mean = np.mean(cpp_stats[key])

        ratio = (
            cpp_mean / py_mean
            if py_mean != 0
            else np.nan
        )

        print(
            f"{label:<25}"
            f"{py_mean:>15.6f}"
            f"{cpp_mean:>15.6f}"
            f"{ratio:>18.3f}x"
        )

    # ------------------------------------------------------------------------
    # Median
    # ------------------------------------------------------------------------

    print("-" * 90)

    print(
        f"{'Median per propagation':<25}"
        f"{'Python':>15}"
        f"{'C++':>15}"
        f"{'C++ / Python':>18}"
    )

    for key, label in metrics:

        py_median = np.median(python_stats[key])
        cpp_median = np.median(cpp_stats[key])

        ratio = (
            cpp_median / py_median
            if py_median != 0
            else np.nan
        )

        print(
            f"{label:<25}"
            f"{py_median:>15.6f}"
            f"{cpp_median:>15.6f}"
            f"{ratio:>18.3f}x"
        )

    print("=" * 90)


# ============================================================================
# Solver comparison plots
# ============================================================================

def plot_solver_comparison(python_stats, cpp_stats):
    """
    Plot distributions of the main RK45 performance metrics.
    """

    metrics = [
        (
            "nfev",
            "Function evaluations per propagation",
        ),
        (
            "accepted_steps",
            "Accepted steps per propagation",
        ),
        (
            "rejected_steps",
            "Rejected steps per propagation",
        ),
        (
            "elapsed_time",
            "Propagation time [s]",
        ),
    ]

    for key, title in metrics:

        plt.figure(figsize=(8, 5))

        plt.boxplot(
            [
                python_stats[key],
                cpp_stats[key],
            ],
            tick_labels=[
                "Python RK45",
                "C++ RK45",
            ],
        )

        plt.ylabel(title)
        plt.title(title)

        plt.grid(
            axis="y",
            alpha=0.3,
        )

        plt.tight_layout()
        plt.show()


# ============================================================================
# Rejection rate
# ============================================================================

def plot_rejection_rate(python_stats, cpp_stats):
    """
    Plot the percentage of rejected RK45 steps per propagation.
    """

    python_total_steps = (
        python_stats["accepted_steps"]
        + python_stats["rejected_steps"]
    )

    cpp_total_steps = (
        cpp_stats["accepted_steps"]
        + cpp_stats["rejected_steps"]
    )

    python_rate = (
        python_stats["rejected_steps"]
        / np.maximum(python_total_steps, 1)
        * 100.0
    )

    cpp_rate = (
        cpp_stats["rejected_steps"]
        / np.maximum(cpp_total_steps, 1)
        * 100.0
    )

    plt.figure(figsize=(8, 5))

    plt.boxplot(
        [
            python_rate,
            cpp_rate,
        ],
        tick_labels=[
            "Python RK45",
            "C++ RK45",
        ],
    )

    plt.ylabel("Rejected steps [%]")
    plt.title("RK45 step rejection rate")

    plt.grid(
        axis="y",
        alpha=0.3,
    )

    plt.tight_layout()
    plt.show()


# ============================================================================
# Evaluations per accepted step
# ============================================================================

def plot_evaluations_per_step(python_stats, cpp_stats):
    """
    Plot the number of function evaluations per accepted step.
    """

    python_ratio = (
        python_stats["nfev"]
        / np.maximum(
            python_stats["accepted_steps"],
            1,
        )
    )

    cpp_ratio = (
        cpp_stats["nfev"]
        / np.maximum(
            cpp_stats["accepted_steps"],
            1,
        )
    )

    plt.figure(figsize=(8, 5))

    plt.boxplot(
        [
            python_ratio,
            cpp_ratio,
        ],
        tick_labels=[
            "Python RK45",
            "C++ RK45",
        ],
    )

    plt.ylabel("Function evaluations / accepted step")
    plt.title("RK45 function evaluations per accepted step")

    plt.grid(
        axis="y",
        alpha=0.3,
    )

    plt.tight_layout()
    plt.show()