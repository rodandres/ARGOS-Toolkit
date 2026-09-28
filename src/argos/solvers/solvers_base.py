from scipy.integrate import solve_ivp

from py.modules.solvers.RK45 import rk45

def native_rk45(f, t_span, y0, **kwargs):
    return rk45(f, t_span, y0, **kwargs)


SUPPORTED_SOLVERS = {    
    "NATIVE_RK45": native_rk45,
}

def solve(f, t_span, y0, method, **kwargs):
    try:
        solver = SUPPORTED_SOLVERS[method.upper()]
    except KeyError:
        raise ValueError(
            f"Método '{method}' no soportado. "
            f"Disponibles: {', '.join(SUPPORTED_SOLVERS)}"
        )
    return solver(f, t_span, y0, **kwargs)