import numpy as np
import pytest

from argos.solvers.RK45 import rk45
from argos.solvers.solver_utilities import OdeResult
from argos.solvers.solvers_base import (
    SUPPORTED_SOLVERS,
    native_rk45,
    solve,
)


# ============================================================================
# RK45 - Basic integration
# ============================================================================


def test_rk45_integrates_constant_derivative():
    def dynamics(t, y):
        return np.array([2.0])

    result = rk45(
        dynamics,
        [0.0, 1.0],
        [0.0],
        h_adaptative=False,
        h0=0.1,
    )

    assert result.success
    assert result.status == 0
    assert result.t[-1] == pytest.approx(1.0)
    assert result.y.shape[0] == 1
    assert result.y.shape[1] == len(result.t)
    assert result.y[0, -1] == pytest.approx(2.0)


def test_rk45_integrates_linear_dynamics():
    # dy/dt = t
    # y(0) = 0
    # analytical solution: y(t) = t² / 2
    def dynamics(t, y):
        return np.array([t])

    result = rk45(
        dynamics,
        [0.0, 2.0],
        [0.0],
        h_adaptative=False,
        h0=0.1,
    )

    assert result.success
    assert result.y[0, -1] == pytest.approx(2.0)


def test_rk45_integrates_exponential_dynamics():
    # dy/dt = y
    # y(0) = 1
    # analytical solution: y(t) = exp(t)
    def dynamics(t, y):
        return y

    result = rk45(
        dynamics,
        [0.0, 1.0],
        [1.0],
        h_adaptative=True,
        h0=0.1,
        rtol=1e-8,
        atol=1e-10,
    )

    assert result.success
    assert result.y[0, -1] == pytest.approx(
        np.exp(1.0),
        rel=1e-7,
    )


# ============================================================================
# RK45 - Vector states
# ============================================================================


def test_rk45_integrates_vector_state():
    def dynamics(t, y):
        return np.array(
            [
                y[1],
                -y[0],
            ]
        )

    result = rk45(
        dynamics,
        [0.0, 1.0],
        [1.0, 0.0],
        h_adaptative=True,
        h0=0.01,
    )

    assert result.success
    assert result.y.shape[0] == 2
    assert result.y.shape[1] == len(result.t)

    # Expected analytical solution:
    # x(t) = cos(t)
    # v(t) = -sin(t)
    assert result.y[0, -1] == pytest.approx(np.cos(1.0), rel=1e-5)
    assert result.y[1, -1] == pytest.approx(-np.sin(1.0), rel=1e-5)


# ============================================================================
# RK45 - Initial conditions
# ============================================================================


def test_rk45_preserves_initial_condition():
    def dynamics(t, y):
        return np.array([1.0, -2.0, 3.0])

    initial_state = np.array([10.0, 20.0, 30.0])

    result = rk45(
        dynamics,
        [0.0, 1.0],
        initial_state,
        h_adaptative=False,
        h0=0.1,
    )

    assert np.allclose(result.y[:, 0], initial_state)
    assert result.t[0] == pytest.approx(0.0)


# ============================================================================
# RK45 - Final time and step handling
# ============================================================================


def test_rk45_does_not_integrate_past_final_time():
    def dynamics(t, y):
        return np.array([1.0])

    result = rk45(
        dynamics,
        [0.0, 1.0],
        [0.0],
        h_adaptative=False,
        h0=0.3,
    )

    assert result.t[-1] == pytest.approx(1.0)
    assert np.all(result.t <= 1.0)


def test_rk45_fixed_step_uses_requested_step_size():
    def dynamics(t, y):
        return np.array([1.0])

    result = rk45(
        dynamics,
        [0.0, 1.0],
        [0.0],
        h_adaptative=False,
        h0=0.25,
    )

    expected_times = np.array(
        [
            0.0,
            0.25,
            0.50,
            0.75,
            1.00,
        ]
    )

    assert np.allclose(result.t, expected_times)


# ============================================================================
# RK45 - Adaptive integration
# ============================================================================


def test_rk45_adaptive_integration_changes_step_size():
    def dynamics(t, y):
        return np.array([y[0]])

    result = rk45(
        dynamics,
        [0.0, 5.0],
        [1.0],
        h_adaptative=True,
        h0=0.01,
    )

    assert result.success
    assert len(result.t) > 1

    step_sizes = np.diff(result.t)

    assert np.all(step_sizes > 0.0)
    assert not np.allclose(
        step_sizes,
        step_sizes[0],
    )


def test_rk45_respects_maximum_step_size():
    def dynamics(t, y):
        return np.array([1.0])

    result = rk45(
        dynamics,
        [0.0, 2.0],
        [0.0],
        h_adaptative=True,
        h0=0.1,
        h_max=0.2,
    )

    step_sizes = np.diff(result.t)

    assert np.all(step_sizes <= 0.2 + 1e-12)


# ============================================================================
# RK45 - Function arguments
# ============================================================================


def test_rk45_passes_additional_arguments():
    def dynamics(t, y, coefficient):
        return np.array([coefficient])

    result = rk45(
        dynamics,
        [0.0, 1.0],
        [0.0],
        args=(3.0,),
        h_adaptative=False,
        h0=0.1,
    )

    assert result.success
    assert result.y[0, -1] == pytest.approx(3.0)


# ============================================================================
# RK45 - Function evaluation count
# ============================================================================


def test_rk45_function_evaluation_count():
    def dynamics(t, y):
        return np.array([1.0])

    result = rk45(
        dynamics,
        [0.0, 1.0],
        [0.0],
        h_adaptative=False,
        h0=0.1,
    )

    assert result.nfev % 6 == 0
    assert result.nfev == 6 * (len(result.t) - 1)


# ============================================================================
# RK45 - Events
# ============================================================================


def test_rk45_detects_non_terminal_event():
    def dynamics(t, y):
        return np.array([1.0])

    def event(t, y):
        return y[0] - 0.5

    result = rk45(
        dynamics,
        [0.0, 1.0],
        [0.0],
        events=event,
        h_adaptative=False,
        h0=0.1,
    )

    assert result.success
    assert len(result.t_events) == 1
    assert len(result.y_events) == 1
    assert len(result.t_events[0]) == 1

    assert result.t_events[0][0] == pytest.approx(0.5)
    assert result.y_events[0][0][0] == pytest.approx(0.5)


def test_rk45_terminates_on_terminal_event():
    def dynamics(t, y):
        return np.array([1.0])

    def event(t, y):
        return y[0] - 0.5

    event.terminal = True

    result = rk45(
        dynamics,
        [0.0, 1.0],
        [0.0],
        events=event,
        h_adaptative=False,
        h0=0.1,
    )

    assert result.success
    assert result.status == 1
    assert "Terminated by event" in result.message

    assert result.t[-1] == pytest.approx(0.5)


def test_rk45_event_direction_positive():
    def dynamics(t, y):
        return np.array([1.0])

    def event(t, y):
        return y[0]

    event.direction = 1

    result = rk45(
        dynamics,
        [-1.0, 1.0],
        [-1.0],
        events=event,
        h_adaptative=False,
        h0=0.1,
    )

    assert len(result.t_events[0]) == 1
    assert result.t_events[0][0] == pytest.approx(0.0)


# ============================================================================
# RK45 - Maximum iterations
# ============================================================================


def test_rk45_reports_maximum_iteration_failure():
    def dynamics(t, y):
        return np.array([1.0])

    result = rk45(
        dynamics,
        [0.0, 10.0],
        [0.0],
        h_adaptative=False,
        h0=0.1,
        max_iter=5,
    )

    assert not result.success
    assert result.status == -1
    assert "Maximum number of iterations exceeded" in result.message


# ============================================================================
# Solver utilities
# ============================================================================


def test_native_rk45_matches_direct_rk45():
    def dynamics(t, y):
        return np.array([y[0]])

    kwargs = {
        "h_adaptative": False,
        "h0": 0.1,
    }

    direct_result = rk45(
        dynamics,
        [0.0, 1.0],
        [1.0],
        **kwargs,
    )

    native_result = native_rk45(
        dynamics,
        [0.0, 1.0],
        [1.0],
        **kwargs,
    )

    assert np.allclose(
        native_result.t,
        direct_result.t,
    )

    assert np.allclose(
        native_result.y,
        direct_result.y,
    )


def test_supported_solvers_contains_native_rk45():
    assert "NATIVE_RK45" in SUPPORTED_SOLVERS
    assert callable(SUPPORTED_SOLVERS["NATIVE_RK45"])


def test_solve_selects_native_rk45():
    def dynamics(t, y):
        return np.array([1.0])

    result = solve(
        dynamics,
        [0.0, 1.0],
        [0.0],
        "NATIVE_RK45",
        h_adaptative=False,
        h0=0.1,
    )

    assert result.success
    assert result.y[0, -1] == pytest.approx(1.0)


def test_solve_method_is_case_insensitive():
    def dynamics(t, y):
        return np.array([1.0])

    result = solve(
        dynamics,
        [0.0, 1.0],
        [0.0],
        "native_rk45",
        h_adaptative=False,
        h0=0.1,
    )

    assert result.success
    assert result.y[0, -1] == pytest.approx(1.0)


def test_solve_rejects_unsupported_method():
    def dynamics(t, y):
        return np.array([1.0])

    with pytest.raises(ValueError):
        solve(
            dynamics,
            [0.0, 1.0],
            [0.0],
            "INVALID",
        )


def test_rk45_does_not_detect_same_event_twice():
    def dynamics(t, y):
        return np.array([1.0])

    def event(t, y):
        return y[0] - 0.5

    result = rk45(
        dynamics,
        [0.0, 1.0],
        [0.0],
        events=event,
        h_adaptative=False,
        h0=0.1,
    )

    assert len(result.t_events[0]) == 1
    assert result.t_events[0][0] == pytest.approx(0.5)        