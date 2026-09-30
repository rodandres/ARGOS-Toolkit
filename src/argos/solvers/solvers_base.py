import time

import numpy as np
from scipy.integrate import solve_ivp

from argos.solvers.solver_utilities import OdeResult

from argos.solvers.solver_statistics import SolverStatistics
from argos.solvers.RK45 import rk45
from argos import _cpp

CPP_RK45_STATS = SolverStatistics()

def native_rk45(f, t_span, y0, **kwargs):
    """
    Integrate an ordinary differential equation using the native RK45 solver.

    Parameters
    ----------
    f : callable
        Function defining the ordinary differential equation. It must accept
        time, state, and optional additional arguments.
    t_span : array-like, shape (2,)
        Integration interval ``[t0, tf]`` [s].
    y0 : array-like
        Initial state vector.
    **kwargs
        Additional keyword arguments passed to :func:`rk45`.

    Returns
    -------
    OdeResult
        Integration result containing the time history, state history,
        solver status, and integration statistics.
    """
    return rk45(f, t_span, y0, **kwargs)


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
        Additional keyword arguments passed to :func:`scipy.integrate.solve_ivp`.

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
        **kwargs
    )

def cpp_rk45(f, t_span, y0, **kwargs):

    if not isinstance(f, str):
        raise TypeError(
            "CPP_RK45 requires the dynamics model to be specified as a string."
        )

    dynamics_model = str(f).upper()

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

    orbital_model = (
        None
        if orbital_model is None
        else str(orbital_model).upper()
    )

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

        mu, length_factor, time_factor = orbital_model_arguments

    elif orbital_model in (None, ""):
        mu = 0.0
        length_factor = 1.0
        time_factor = 1.0

    else:
        raise ValueError(
            f"Unsupported orbital model: '{orbital_model}'."
        )

    h0 = kwargs.pop("h0", 0.1)
    h_min = kwargs.pop("h_min", 1e-6)
    h_max = kwargs.pop("h_max", 1.0)
    rtol = kwargs.pop("rtol", 1e-6)
    atol = kwargs.pop("atol", 1e-9)
    adaptive = kwargs.pop("h_adaptative", True)
    max_iter = kwargs.pop("max_iter", 100000)
    verbose = kwargs.pop("verbose", False)

    if kwargs:
        unsupported = ", ".join(sorted(kwargs))
        raise TypeError(
            f"cpp_rk45() got unsupported keyword argument(s): {unsupported}"
        )

    y0 = np.asarray(y0, dtype=float)
    applied_force = np.asarray(applied_force, dtype=float)
    disturbance_force = np.asarray(disturbance_force, dtype=float)
    t_span = np.asarray(t_span, dtype=float)

    # ---------------------------------------------------------------
    # Timing
    # ---------------------------------------------------------------

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

    # ---------------------------------------------------------------
    # Statistics
    # ---------------------------------------------------------------

    CPP_RK45_STATS.add(
        nfev=int(result["nfev"]),
        accepted_steps=int(result["accepted_steps"]),
        rejected_steps=int(result["rejected_steps"]),
        elapsed_time=elapsed,
    )

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

SUPPORTED_SOLVERS = {    
    "NATIVE_RK45": native_rk45,
    "SCIPY_RK45": scipy_rk45,
    "CPP_RK45": cpp_rk45,
}


def solve(f, t_span, y0, method, **kwargs):
    """
    Integrate an ordinary differential equation using a supported solver.

    Parameters
    ----------
    f : callable
        Function defining the ordinary differential equation.
    t_span : array-like, shape (2,)
        Integration interval ``[t0, tf]`` [s].
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
    return solver(f, t_span, y0, **kwargs)