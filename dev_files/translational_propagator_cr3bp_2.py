import numpy as np
from scipy.integrate import solve_ivp

from argos import _cpp


# ============================================================================
# Python reference
# ============================================================================


def cr3bp(
    t,
    state,
    mu,
    length_factor,
    time_factor,
):
    x, y, z, x_dot, y_dot, z_dot = state

    # ------------------------------------------------------------------------
    # Dimensional -> normalized units
    # ------------------------------------------------------------------------

    x /= length_factor
    y /= length_factor
    z /= length_factor

    x_dot /= length_factor / time_factor
    y_dot /= length_factor / time_factor
    z_dot /= length_factor / time_factor

    # ------------------------------------------------------------------------
    # CR3BP equations
    # ------------------------------------------------------------------------

    r1 = np.sqrt(
        (x + mu) ** 2
        + y**2
        + z**2
    )

    r2 = np.sqrt(
        (x - (1.0 - mu)) ** 2
        + y**2
        + z**2
    )

    omega_x = (
        x
        - (1.0 - mu) * (x + mu) / r1**3
        - mu * (x - (1.0 - mu)) / r2**3
    )

    omega_y = (
        y
        - (1.0 - mu) * y / r1**3
        - mu * y / r2**3
    )

    omega_z = (
        -(1.0 - mu) * z / r1**3
        - mu * z / r2**3
    )

    x_ddot = 2.0 * y_dot + omega_x
    y_ddot = -2.0 * x_dot + omega_y
    z_ddot = omega_z

    # ------------------------------------------------------------------------
    # Normalized -> dimensional units
    # ------------------------------------------------------------------------

    x_dot *= length_factor / time_factor
    y_dot *= length_factor / time_factor
    z_dot *= length_factor / time_factor

    x_ddot *= length_factor / time_factor**2
    y_ddot *= length_factor / time_factor**2
    z_ddot *= length_factor / time_factor**2

    return np.array(
        [
            x_dot,
            y_dot,
            z_dot,
            x_ddot,
            y_ddot,
            z_ddot,
        ]
    )


# ============================================================================
# CR3BP parameters
# ============================================================================

mu = 1.215e-2

length_factor = 384400.0e3

time_factor = (
    27.321661
    / (2.0 * np.pi)
    * 24.0
    * 3600.0
)


# ============================================================================
# Initial state
# ============================================================================

state0 = np.array(
    [
        3.8e8,
        0.0,
        0.0,
        0.0,
        1.0e3,
        0.0,
    ],
    dtype=float,
)


# ============================================================================
# Integration parameters
# ============================================================================

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
    mass=1000.0,
    applied_force=np.zeros(3),
    disturbance_force=np.zeros(3),
    dynamics_model="cr3bp",
    orbital_model="",
    mu=mu,
    length_factor=length_factor,
    time_factor=time_factor,
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
    fun=lambda t, state: cr3bp(
        t,
        state,
        mu,
        length_factor,
        time_factor,
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
print("C++ CR3BP vs Python solve_ivp")
print("=" * 72)

print("\nCR3BP parameters:")
print(f"  mu:             {mu}")
print(f"  length_factor:  {length_factor}")
print(f"  time_factor:    {time_factor}")

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