import numpy as np
from scipy.integrate import solve_ivp

from argos import _cpp


# ============================================================================
# Reference dynamics
# ============================================================================


def rel2bp(t, state, mu):
    r = state[:3]
    v = state[3:]

    norm_r = np.linalg.norm(r)

    acceleration = -mu * r / norm_r**3

    return np.concatenate(
        (v, acceleration)
    )


def newton(
    t,
    state,
    mass,
    applied_force,
    disturbance_force,
    orbital_model=None,
    *args,
):
    if orbital_model is not None:
        orbital_derivative = orbital_model(
            t,
            state,
            *args,
        )

        acceleration = orbital_derivative[3:6]

    else:
        acceleration = np.zeros(3)

    if mass <= 0:
        external_acceleration = np.zeros(3)
    else:
        external_acceleration = (
            applied_force + disturbance_force
        ) / mass

    total_acceleration = (
        acceleration + external_acceleration
    )

    velocity = state[3:6]

    return np.concatenate(
        (
            velocity,
            total_acceleration,
        )
    )


# ============================================================================
# Problem definition
# ============================================================================

mu = 6.67430e-11 * 5.972e24

state0 = np.array(
    [
        7.0e6,
        0.0,
        0.0,
        0.0,
        7.5e3,
        0.0,
    ],
    dtype=float,
)

mass = 1000.0

applied_force = np.zeros(3)
disturbance_force = np.zeros(3)

t0 = 0.0
tf = 10.0

rtol = 1e-9
atol = 1e-12


# ============================================================================
# C++ backend
# ============================================================================

result_cpp = _cpp.propagate_translational(
    state=state0,
    t0=t0,
    tf=tf,
    mass=mass,
    applied_force=applied_force,
    disturbance_force=disturbance_force,
    dynamics_model="newton",
    orbital_model="REL2BP",
    mu=mu,
    length_factor=1.0,
    time_factor=1.0,
    h0=0.1,
    h_min=1e-6,
    h_max=1.0,
    rtol=rtol,
    atol=atol,
    adaptive=True,
    max_iter=100000,
    verbose=False,
)


# ============================================================================
# Python reference
# ============================================================================

result_python = solve_ivp(
    fun=lambda t, state: newton(
        t,
        state,
        mass,
        applied_force,
        disturbance_force,
        rel2bp,
        mu,
    ),
    t_span=(t0, tf),
    y0=state0,
    method="RK45",
    rtol=rtol,
    atol=atol,
)


# ============================================================================
# Comparison
# ============================================================================

cpp_final = result_cpp["state"]
python_final = result_python.y[:, -1]

absolute_error = np.abs(
    cpp_final - python_final
)

relative_error = absolute_error / np.maximum(
    np.abs(python_final),
    np.finfo(float).eps,
)

max_absolute_error = np.max(
    absolute_error
)

max_relative_error = np.max(
    relative_error
)


# ============================================================================
# Output
# ============================================================================

print("=" * 72)
print("C++ RK45 vs Python solve_ivp")
print("=" * 72)

print("\nC++:")
print(f"  success:          {result_cpp['success']}")
print(f"  status:           {result_cpp['status']}")
print(f"  message:          {result_cpp['message']}")
print(f"  nfev:             {result_cpp['nfev']}")
print(f"  accepted steps:   {result_cpp['accepted_steps']}")
print(f"  rejected steps:   {result_cpp['rejected_steps']}")
print(f"  output points:    {len(result_cpp['t'])}")

print("\nPython solve_ivp:")
print(f"  success:          {result_python.success}")
print(f"  status:           {result_python.status}")
print(f"  message:          {result_python.message}")
print(f"  nfev:             {result_python.nfev}")
print(f"  output points:    {len(result_python.t)}")

print("\nFinal state:")

print("\nC++:")
print(cpp_final)

print("\nPython:")
print(python_final)

print("\nAbsolute error:")
print(absolute_error)

print(f"\nMaximum absolute error: {max_absolute_error:.6e}")
print(f"Maximum relative error: {max_relative_error:.6e}")

print("\nState-history shapes:")
print(f"  C++ t:    {result_cpp['t'].shape}")
print(f"  C++ y:    {result_cpp['y'].shape}")
print(f"  Python t: {result_python.t.shape}")
print(f"  Python y: {result_python.y.shape}")

print("=" * 72)