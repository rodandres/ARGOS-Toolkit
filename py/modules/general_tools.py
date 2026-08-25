
import numpy as np

def _as_3d_array(value: float | np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)

    if arr.ndim == 0:
        return np.full(3, arr)

    if arr.shape == (3,):
        return arr.copy()

    raise ValueError(
        f"{name} Shall be a scalar or a 3D numpy vector, "
        f"but received {arr.shape}"
    )

import copy
import numpy as np

# NOTE: The quaternion interpolation shall be change to linear interpolation (LERP) for simplicity.

def get_state_at(history, spacecraft_name: str, t: float):
    """
    Returns the true state of a spacecraft at a requested time.

    If an exact time is available, the corresponding state is returned.
    If the requested time lies between two recorded states, linear
    interpolation is performed.
    If the requested time lies outside the recorded range, linear
    extrapolation using the nearest two states is performed.

    Args:
        history: SimulationHistory containing spacecraft histories.
        spacecraft_name: Name of the spacecraft.
        t: Requested time [s].

    Returns:
        StateVariables: Spacecraft state at time t.

    Raises:
        ValueError: If the spacecraft does not exist, the history is empty,
                    or fewer than two samples are available.
    """

    # Get spacecraft history
    spacecraft_history = history.spacecrafts_history.get(
        spacecraft_name
    )

    if spacecraft_history is None:
        raise ValueError(
            f"Spacecraft '{spacecraft_name}' not found in history."
        )

    times = np.asarray(
        spacecraft_history.t,
        dtype=float
    )

    states = spacecraft_history.true_state

    if len(times) == 0:
        raise ValueError(
            f"No history available for spacecraft '{spacecraft_name}'."
        )

    if len(times) != len(states):
        raise ValueError(
            f"Inconsistent history for spacecraft '{spacecraft_name}': "
            f"{len(times)} times but {len(states)} states."
        )

    # ---------------------------------------------------------
    # Exact match
    # ---------------------------------------------------------

    exact_indices = np.where(
        np.isclose(times, t)
    )[0]

    if len(exact_indices) > 0:
        return copy.deepcopy(
            states[exact_indices[0]]
        )

    # ---------------------------------------------------------
    # Need at least two samples for interpolation/extrapolation
    # ---------------------------------------------------------

    if len(times) < 2:
        raise ValueError(
            f"At least two states are required to interpolate or "
            f"extrapolate spacecraft '{spacecraft_name}'."
        )

    # ---------------------------------------------------------
    # Select the two states
    # ---------------------------------------------------------

    if t < times[0]:

        # Extrapolation before first sample
        i0 = 0
        i1 = 1

    elif t > times[-1]:

        # Extrapolation after last sample
        i0 = len(times) - 2
        i1 = len(times) - 1

    else:

        # Interpolation
        i1 = np.searchsorted(times, t)
        i0 = i1 - 1

    t0 = times[i0]
    t1 = times[i1]

    state0 = states[i0]
    state1 = states[i1]

    # Interpolation/extrapolation coefficient
    alpha = (t - t0) / (t1 - t0)

    # ---------------------------------------------------------
    # Interpolate StateVariables
    # ---------------------------------------------------------

    result = copy.deepcopy(state0)

    for field_name in state0.__dataclass_fields__:

        value0 = getattr(state0, field_name)
        value1 = getattr(state1, field_name)

        if value0 is None or value1 is None:
            setattr(result, field_name, None)
            continue

        value = value0 + alpha * (
            value1 - value0
        )

        setattr(
            result,
            field_name,
            value
        )

    return result