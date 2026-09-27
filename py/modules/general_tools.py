
import os

import numpy as np

from py.general.dataclasses import StateVariables

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from py.general.data_save import SimulationHistory


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

def get_state_at(history: SimulationHistory, spacecraft_name: str, t: float):
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
    if spacecraft_name not in history.spacecraft_names:
        raise ValueError(
            f"Spacecraft '{spacecraft_name}' not found in history."
        )

    spacecraft_history = history.spacecrafts_history[spacecraft_name]

    # Get current events from the spacecraft history
    current_events = spacecraft_history.events
    event_at_time_t = None

    for event in current_events:
        if event.time <= t:
            event_at_time_t = event
        else:
            break

    if event_at_time_t is None:
        raise ValueError(
            f"No events found in history for spacecraft "
            f"'{spacecraft_name}' at or before time {t}."
        )

    # Analyze tick from that phase time to actual t
    time_since_event = t - event_at_time_t.time
    dt_master = event_at_time_t.dt_master

    ticks_since_event = time_since_event / dt_master

    tick_base = event_at_time_t.tick
    tick_at_t = tick_base + round(ticks_since_event)

    # Find tick in chunk
    current_chunk_index = spacecraft_history.chunk_index
    current_index = spacecraft_history.index
    chunk_size = spacecraft_history.chunk_size

    total_samples = (
        current_chunk_index * chunk_size
        + current_index
    )

    if total_samples == 0:
        raise ValueError(
            f"No history available for spacecraft '{spacecraft_name}'."
        )

    if total_samples < 2:
        raise ValueError(
            f"At least two states are required to interpolate or "
            f"extrapolate spacecraft '{spacecraft_name}'."
        )

    # ---------------------------------------------------------
    # Determine the two ticks needed
    # ---------------------------------------------------------

    if tick_at_t <= 0:

        # Extrapolation before first sample
        tick_need_0 = 0
        tick_need_1 = 1

    elif tick_at_t >= total_samples - 1:

        # Extrapolation after last sample
        tick_need_0 = total_samples - 2
        tick_need_1 = total_samples - 1

    else:

        # Interpolation between neighboring samples
        tick_need_0 = tick_at_t - 1
        tick_need_1 = tick_at_t

    # ---------------------------------------------------------
    # Find chunk and index for tick 0
    # ---------------------------------------------------------

    chunk_index_0 = tick_need_0 // chunk_size
    index_0 = tick_need_0 % chunk_size

    # ---------------------------------------------------------
    # Find chunk and index for tick 1
    # ---------------------------------------------------------

    chunk_index_1 = tick_need_1 // chunk_size
    index_1 = tick_need_1 % chunk_size

    # ---------------------------------------------------------
    # Load state 0
    # ---------------------------------------------------------

    if chunk_index_0 == current_chunk_index:

        # State is in current chunk
        t0 = spacecraft_history.t[index_0]

        state0 = StateVariables(
            position=spacecraft_history.true_position[index_0],
            velocity=spacecraft_history.true_velocity[index_0],
            acceleration=spacecraft_history.true_acceleration[index_0],
            attitude=spacecraft_history.true_attitude[index_0],
            angular_velocity=spacecraft_history.true_angular_velocity[
                index_0
            ],
            angular_acceleration=(
                spacecraft_history.true_angular_acceleration[index_0]
            ),
        )

    else:

        # State is in a previous chunk
        chunk_file_0 = os.path.join(
            history.data_file_path,
            f"{spacecraft_name}_history_chunk_{chunk_index_0}.npz"
        )

        with np.load(chunk_file_0) as chunk_data_0:

            t0 = chunk_data_0["t"][index_0]

            state0 = StateVariables(
                position=chunk_data_0["true_position"][index_0],
                velocity=chunk_data_0["true_velocity"][index_0],
                acceleration=chunk_data_0["true_acceleration"][index_0],
                attitude=chunk_data_0["true_attitude"][index_0],
                angular_velocity=chunk_data_0[
                    "true_angular_velocity"
                ][index_0],
                angular_acceleration=chunk_data_0[
                    "true_angular_acceleration"
                ][index_0],
            )

    # ---------------------------------------------------------
    # Load state 1
    # ---------------------------------------------------------

    if chunk_index_1 == current_chunk_index:

        # State is in current chunk
        t1 = spacecraft_history.t[index_1]

        state1 = StateVariables(
            position=spacecraft_history.true_position[index_1],
            velocity=spacecraft_history.true_velocity[index_1],
            acceleration=spacecraft_history.true_acceleration[index_1],
            attitude=spacecraft_history.true_attitude[index_1],
            angular_velocity=spacecraft_history.true_angular_velocity[
                index_1
            ],
            angular_acceleration=(
                spacecraft_history.true_angular_acceleration[index_1]
            ),
        )

    else:

        # State is in a previous chunk
        chunk_file_1 = os.path.join(
            history.data_file_path,
            f"{spacecraft_name}_history_chunk_{chunk_index_1}.npz"
        )

        with np.load(chunk_file_1) as chunk_data_1:

            t1 = chunk_data_1["t"][index_1]

            state1 = StateVariables(
                position=chunk_data_1["true_position"][index_1],
                velocity=chunk_data_1["true_velocity"][index_1],
                acceleration=chunk_data_1["true_acceleration"][index_1],
                attitude=chunk_data_1["true_attitude"][index_1],
                angular_velocity=chunk_data_1[
                    "true_angular_velocity"
                ][index_1],
                angular_acceleration=chunk_data_1[
                    "true_angular_acceleration"
                ][index_1],
            )

    # ---------------------------------------------------------
    # Exact match
    # ---------------------------------------------------------

    if np.isclose(t0, t):
        return copy.deepcopy(state0)

    if np.isclose(t1, t):
        return copy.deepcopy(state1)

    # ---------------------------------------------------------
    # Interpolation / extrapolation coefficient
    # ---------------------------------------------------------

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