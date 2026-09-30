from scipy.integrate import solve_ivp

from argos.solvers.RK45 import rk45

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


SUPPORTED_SOLVERS = {    
    "NATIVE_RK45": native_rk45,
    "SCIPY_RK45": scipy_rk45
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