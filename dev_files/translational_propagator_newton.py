import numpy as np
from scipy.integrate import solve_ivp

from argos import _cpp


# ============================================================================
# Python reference
# ============================================================================


def newton(
    t,
    state,
    mass,
    applied_force,
    disturbance_force,
):
    if mass <= 0:
        external_acceleration = np.zeros(3)
    else:
        external_acceleration = (
            applied_force + disturbance_force
        ) / mass

    velocity = state[3:6]

    return np.concatenate(
        (
            velocity,
            external_acceleration,
        )
    )


# ============================================================================
# Problem definition
# ============================================================================

state0 = np.array(
    [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    dtype=float,
)

mass = 1000.0

applied_force = np.array(
    [
        10.0,
        20.0,
        30.0,
    ],
    dtype=float,
)

disturbance_force = np.array(
    [
        5.0,
        -10.0,
        15.0,
    ],
    dtype=float,
)

t0 = 0.0
tf = 10.0

rtol = 1e-9
atol = 1e-12


# ============================================================================
# Expected analytical result
# ============================================================================

total_force = (
    applied_force
    + disturbance_force
)

expected_acceleration = (
    total_force / mass
)

expected_velocity = (
    expected_acceleration * (tf - t0)
)

expected_position = (
    0.5
    * expected_acceleration
    * (tf - t0) ** 2
)

expected_final = np.concatenate(
    (
        expected_position,
        expected_velocity,
    )
)


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
    orbital_model="",
    mu=0.0,
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


error_cpp_analytical = np.abs(
    cpp_final - expected_final
)

error_python_analytical = np.abs(
    python_final - expected_final
)

error_cpp_python = np.abs(
    cpp_final - python_final
)


max_error_cpp_analytical = np.max(
    error_cpp_analytical
)

max_error_python_analytical = np.max(
    error_python_analytical
)

max_error_cpp_python = np.max(
    error_cpp_python
)


# ============================================================================
# Output
# ============================================================================

print("=" * 72)
print("C++ NEWTON + external forces")
print("=" * 72)

print("\nMass:")
print(f"  {mass} kg")

print("\nApplied force:")
print(applied_force)

print("\nDisturbance force:")
print(disturbance_force)

print("\nTotal force:")
print(total_force)

print("\nExpected acceleration:")
print(expected_acceleration)

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

print("\nExpected:")
print(expected_final)

print("\nC++:")
print(cpp_final)

print("\nPython:")
print(python_final)

print("\nC++ absolute error vs analytical:")
print(error_cpp_analytical)

print("\nPython absolute error vs analytical:")
print(error_python_analytical)

print("\nC++ vs Python absolute error:")
print(error_cpp_python)

print(
    "\nMaximum C++ error vs analytical: "
    f"{max_error_cpp_analytical:.6e}"
)

print(
    "Maximum Python error vs analytical: "
    f"{max_error_python_analytical:.6e}"
)

print(
    "Maximum C++ vs Python error: "
    f"{max_error_cpp_python:.6e}"
)

print("\nState-history shapes:")
print(f"  C++ t:    {result_cpp['t'].shape}")
print(f"  C++ y:    {result_cpp['y'].shape}")
print(f"  Python t: {result_python.t.shape}")
print(f"  Python y: {result_python.y.shape}")

print("=" * 72)