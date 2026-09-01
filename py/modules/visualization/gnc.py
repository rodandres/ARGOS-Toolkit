from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from py.modules.visualization.style import *

def plot_control_result(
    spacecraft_name: str,
    result,
    save: bool = False,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    filename: str = "control_result.png",
    dpi: int = 300,
):
    """
    Plot commanded versus applied force and torque for a spacecraft.

    Parameters
    ----------
    spacecraft_name : str
        Name of the spacecraft to plot.
    result : SimulationHistory
        General simulation result containing spacecraft histories.
    save : bool, optional
        Save the generated figure to disk. Default is False.
    output_dir : Path | str, optional
        Directory where the figure will be saved.
    filename : str, optional
        Output filename.
    dpi : int, optional
        Figure resolution in dots per inch.

    Returns
    -------
    matplotlib.figure.Figure
        Generated figure.
    """

    # ------------------------------------------------------------------
    # Validate spacecraft
    # ------------------------------------------------------------------

    if spacecraft_name not in result.spacecrafts_history:
        raise KeyError(
            f"Spacecraft '{spacecraft_name}' not found in simulation history."
        )

    history = result.spacecrafts_history[spacecraft_name]

    # ------------------------------------------------------------------
    # Retrieve data
    # ------------------------------------------------------------------

    t = np.asarray(history.t)

    commanded_force = np.asarray(
        history.control_output_data["force"]
    )

    applied_force = np.asarray(
        history.current_force_exerted
    )

    commanded_torque = np.asarray(
        history.control_output_data["torque"]
    )

    applied_torque = np.asarray(
        history.current_torque_exerted
    )

    labels = ("X", "Y", "Z")

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------

    fig, axes = plt.subplots(
        3,
        2,
        figsize=(13, 9),
        sharex="col",
    )

    style_figure(fig)

    # ------------------------------------------------------------------
    # Force / Torque plots
    # ------------------------------------------------------------------

    for i, label in enumerate(labels):

        # ==============================================================
        # Force
        # ==============================================================

        ax_force = axes[i, 0]

        ax_force.plot(
            t,
            commanded_force[:, i],
            linestyle="--",
            linewidth=1.5,
            color=ACCENT_COLOR,
            label="Commanded",
        )

        ax_force.plot(
            t,
            applied_force[:, i],
            linestyle="-",
            linewidth=1.5,
            color=SECONDARY_COLOR,
            label="Applied",
        )

        style_axes(
            ax_force,
            title=f"Force {label}",
            ylabel=r"$F_{" + label.lower() + r"}$ [N]",
        )

        ax_force.legend(
            fontsize=7,
            labelcolor=TEXT_COLOR,
            edgecolor="none",
            facecolor=AXES_BG,
        )

        # ==============================================================
        # Torque
        # ==============================================================

        ax_torque = axes[i, 1]

        ax_torque.plot(
            t,
            commanded_torque[:, i],
            linestyle="--",
            linewidth=1.5,
            color=ACCENT_COLOR,
            label="Commanded",
        )

        ax_torque.plot(
            t,
            applied_torque[:, i],
            linestyle="-",
            linewidth=1.5,
            color=SECONDARY_COLOR,
            label="Applied",
        )

        style_axes(
            ax_torque,
            title=f"Torque {label}",
            ylabel=r"$\tau_{" + label.lower() + r"}$ [N·m]",
        )

        ax_torque.legend(
            fontsize=7,
            labelcolor=TEXT_COLOR,
            edgecolor="none",
            facecolor=AXES_BG,
        )

    # ------------------------------------------------------------------
    # X-axis labels
    # ------------------------------------------------------------------

    axes[-1, 0].set_xlabel("Time [s]")
    axes[-1, 1].set_xlabel("Time [s]")

    # ------------------------------------------------------------------
    # Figure title
    # ------------------------------------------------------------------

    fig.suptitle(
        f'Control Result - "{spacecraft_name}"',
        fontsize=13,
        y=0.99,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    # ------------------------------------------------------------------
    # Finalize
    # ------------------------------------------------------------------

    return finalize_figure(
        fig,
        show=True,
        save=save,
        output_dir=output_dir,
        filename=filename,
        dpi=dpi,
    )

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================================
# Helpers
# ============================================================================

def _normalize_spacecraft_names(
    spacecraft_names: str | list[str],
) -> list[str]:
    """Normalize a spacecraft name or list of names."""

    if isinstance(spacecraft_names, str):
        spacecraft_names = [spacecraft_names]

    if not spacecraft_names:
        raise ValueError(
            "At least one spacecraft name must be provided."
        )

    return spacecraft_names


def _get_target_name(
    history,
    spacecraft_name: str,
) -> str | None:
    """
    Retrieve the latest valid target spacecraft name.

    Returns
    -------
    str | None
        Target spacecraft name, or None if no target is defined.
    """

    if not hasattr(history, "target_name"):
        return None

    target_names = np.asarray(
        history.target_name,
        dtype=object,
    )

    if target_names.size == 0:
        return None

    for target_name in reversed(target_names):

        if target_name is not None:
            return str(target_name)

    return None


def _interpolate_state(
    source_t: np.ndarray,
    source_state: np.ndarray,
    target_t: np.ndarray,
) -> np.ndarray:
    """
    Interpolate a state history onto another time vector.
    """

    source_t = np.asarray(source_t)
    source_state = np.asarray(source_state)
    target_t = np.asarray(target_t)

    if source_state.ndim == 1:
        source_state = source_state[:, None]

    if len(source_t) != len(source_state):
        raise ValueError(
            "Time vector and state history have different lengths: "
            f"{len(source_t)} vs {len(source_state)}."
        )

    if len(source_t) == len(target_t) and np.allclose(
        source_t,
        target_t,
    ):
        return source_state.copy()

    interpolated = np.empty(
        (
            len(target_t),
            source_state.shape[1],
        )
    )

    for i in range(source_state.shape[1]):

        interpolated[:, i] = np.interp(
            target_t,
            source_t,
            source_state[:, i],
        )

    return interpolated


def _plot_state_comparison(
    ax,
    t,
    true_data,
    estimated_data,
    labels,
    ylabel,
    title,
    line_colors,
    estimated_label="Navigation",
):
    """
    Plot true and estimated state components on a single axis.

    True state:
        Solid line

    Estimated state:
        Dashed line
    """

    t = np.asarray(t)
    true_data = np.asarray(true_data)
    estimated_data = np.asarray(estimated_data)

    if true_data.ndim == 1:
        true_data = true_data[:, None]

    if estimated_data.ndim == 1:
        estimated_data = estimated_data[:, None]

    if len(t) != len(true_data):
        raise ValueError(
            f"True state length ({len(true_data)}) does not match "
            f"time length ({len(t)})."
        )

    if len(t) != len(estimated_data):
        raise ValueError(
            f"Estimated state length ({len(estimated_data)}) does not "
            f"match time length ({len(t)})."
        )

    for i, label in enumerate(labels):

        ax.plot(
            t,
            true_data[:, i],
            color=line_colors[i],
            linewidth=1.4,
            linestyle="-",
            label=f"True {label}",
        )

        ax.plot(
            t,
            estimated_data[:, i],
            color=line_colors[i],
            linewidth=1.2,
            linestyle="--",
            alpha=0.8,
            label=f"{estimated_label} {label}",
        )

    style_axes(
        ax,
        title=title,
        ylabel=ylabel,
    )

    ax.legend(
        fontsize=7,
        ncol=2,
        labelcolor=TEXT_COLOR,
        edgecolor="none",
        facecolor=AXES_BG,
    )


def _disable_axis(
    ax,
    title="No target defined",
):
    """
    Disable an axis when target data is unavailable.
    """

    ax.set_title(
        title,
        color=TEXT_COLOR,
        fontsize=9,
    )

    ax.text(
        0.5,
        0.5,
        "No target defined",
        transform=ax.transAxes,
        ha="center",
        va="center",
        color=TEXT_COLOR,
        fontsize=9,
    )

    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_visible(False)


# ============================================================================
# State comparison
# ============================================================================

def plot_state_comparison(
    spacecraft_names: str | list[str],
    result,
    save: bool = False,
    show: bool = True,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    filename: str = "state_comparison.png",
    dpi: int = 300,
):
    """
    Plot spacecraft and target state comparisons.

    Each spacecraft is displayed in its own column.

    Rows
    ----
    1. Own Position
        True vs Navigation

    2. Target Position
        True Target vs Estimated Target

    3. Own Velocity
        True vs Navigation

    4. Target Velocity
        True Target vs Estimated Target

    5. Own Attitude
        True vs Navigation

    6. Target Attitude
        True Target vs Estimated Target

    7. Own Angular Velocity
        True vs Navigation

    8. Target Angular Velocity
        True Target vs Estimated Target

    Notes
    -----
    If a spacecraft has no target defined, target rows are disabled
    while its own-state rows remain available.
    """

    # =========================================================================
    # Normalize input
    # =========================================================================

    spacecraft_names = _normalize_spacecraft_names(
        spacecraft_names
    )

    # =========================================================================
    # Validate spacecrafts
    # =========================================================================

    for name in spacecraft_names:

        if name not in result.spacecrafts_history:
            raise KeyError(
                f"Spacecraft '{name}' not found in simulation history."
            )

    # =========================================================================
    # Figure
    # =========================================================================

    n_spacecrafts = len(spacecraft_names)

    fig, axes = plt.subplots(
        8,
        n_spacecrafts,
        figsize=(
            7 * n_spacecrafts,
            24,
        ),
        sharex="col",
        squeeze=False,
    )

    style_figure(fig)

    # =========================================================================
    # Labels
    # =========================================================================

    vector_labels = (
        "X",
        "Y",
        "Z",
    )

    quaternion_labels = (
        "qx",
        "qy",
        "qz",
        "qw",
    )

    angular_velocity_labels = (
        "X",
        "Y",
        "Z",
    )

    # =========================================================================
    # Colors
    # =========================================================================

    vector_colors = (
        LINE_COLOR,
        SECONDARY_COLOR,
        ACCENT_COLOR,
    )

    quaternion_colors = (
        LINE_COLOR,
        SECONDARY_COLOR,
        ACCENT_COLOR,
        TEXT_COLOR,
    )

    angular_velocity_colors = (
        LINE_COLOR,
        SECONDARY_COLOR,
        ACCENT_COLOR,
    )

    # =========================================================================
    # Process spacecrafts
    # =========================================================================

    for column, spacecraft_name in enumerate(
        spacecraft_names
    ):

        history = result.spacecrafts_history[
            spacecraft_name
        ]

        # =====================================================================
        # Time
        # =====================================================================

        t = np.asarray(
            history.t
        )

        # =====================================================================
        # Own spacecraft state
        # =====================================================================

        true_position = np.asarray(
            history.true_state["position"]
        )

        navigation_position = np.asarray(
            history.estimated_data[
                "spacecraft_state"
            ]["position"]
        )

        true_velocity = np.asarray(
            history.true_state["velocity"]
        )

        navigation_velocity = np.asarray(
            history.estimated_data[
                "spacecraft_state"
            ]["velocity"]
        )

        true_attitude = np.asarray(
            history.true_state["attitude"]
        )

        navigation_attitude = np.asarray(
            history.estimated_data[
                "spacecraft_state"
            ]["attitude"]
        )

        true_angular_velocity = np.asarray(
            history.true_state["angular_velocity"]
        )

        navigation_angular_velocity = np.asarray(
            history.estimated_data[
                "spacecraft_state"
            ]["angular_velocity"]
        )

        # =====================================================================
        # Own position
        # =====================================================================

        _plot_state_comparison(
            axes[0, column],
            t,
            true_position,
            navigation_position,
            vector_labels,
            ylabel="Position",
            title=f"Position - {spacecraft_name}",
            line_colors=vector_colors,
            estimated_label="Navigation",
        )

        # =====================================================================
        # Own velocity
        # =====================================================================

        _plot_state_comparison(
            axes[2, column],
            t,
            true_velocity,
            navigation_velocity,
            vector_labels,
            ylabel="Velocity",
            title=f"Velocity - {spacecraft_name}",
            line_colors=vector_colors,
            estimated_label="Navigation",
        )

        # =====================================================================
        # Own attitude
        # =====================================================================

        _plot_state_comparison(
            axes[4, column],
            t,
            true_attitude,
            navigation_attitude,
            quaternion_labels,
            ylabel="Quaternion",
            title=f"Attitude - {spacecraft_name}",
            line_colors=quaternion_colors,
            estimated_label="Navigation",
        )

        # =====================================================================
        # Own angular velocity
        # =====================================================================

        _plot_state_comparison(
            axes[6, column],
            t,
            true_angular_velocity,
            navigation_angular_velocity,
            angular_velocity_labels,
            ylabel="$\\omega$",
            title=f"Angular Velocity - {spacecraft_name}",
            line_colors=angular_velocity_colors,
            estimated_label="Navigation",
        )

        # =====================================================================
        # Target
        # =====================================================================

        target_name = _get_target_name(
            history,
            spacecraft_name,
        )

        # ---------------------------------------------------------------------
        # No target
        # ---------------------------------------------------------------------

        if target_name is None:

            _disable_axis(
                axes[1, column],
                title="Target Position",
            )

            _disable_axis(
                axes[3, column],
                title="Target Velocity",
            )

            _disable_axis(
                axes[5, column],
                title="Target Attitude",
            )

            _disable_axis(
                axes[7, column],
                title="Target Angular Velocity",
            )

            axes[-1, column].set_xlabel(
                "Time [s]"
            )

            continue

        # ---------------------------------------------------------------------
        # Validate target
        # ---------------------------------------------------------------------

        if target_name not in result.spacecrafts_history:

            raise KeyError(
                f"Target spacecraft '{target_name}' referenced by "
                f"'{spacecraft_name}' was not found in simulation history."
            )

        target_history = result.spacecrafts_history[
            target_name
        ]

        target_t = np.asarray(
            target_history.t
        )

        # =====================================================================
        # Target true state
        # =====================================================================

        target_true_position = np.asarray(
            target_history.true_state["position"]
        )

        target_true_velocity = np.asarray(
            target_history.true_state["velocity"]
        )

        target_true_attitude = np.asarray(
            target_history.true_state["attitude"]
        )

        target_true_angular_velocity = np.asarray(
            target_history.true_state["angular_velocity"]
        )

        # =====================================================================
        # Target estimated state
        # =====================================================================

        target_estimated_position = np.asarray(
            history.estimated_data[
                "reference_state"
            ]["position"]
        )

        target_estimated_velocity = np.asarray(
            history.estimated_data[
                "reference_state"
            ]["velocity"]
        )

        target_estimated_attitude = np.asarray(
            history.estimated_data[
                "reference_state"
            ]["attitude"]
        )

        target_estimated_angular_velocity = np.asarray(
            history.estimated_data[
                "reference_state"
            ]["angular_velocity"]
        )

        # =====================================================================
        # Interpolate target truth onto observer time
        # =====================================================================

        target_true_position = _interpolate_state(
            target_t,
            target_true_position,
            t,
        )

        target_true_velocity = _interpolate_state(
            target_t,
            target_true_velocity,
            t,
        )

        target_true_attitude = _interpolate_state(
            target_t,
            target_true_attitude,
            t,
        )

        target_true_angular_velocity = _interpolate_state(
            target_t,
            target_true_angular_velocity,
            t,
        )

        # =====================================================================
        # Target position
        # =====================================================================

        _plot_state_comparison(
            axes[1, column],
            t,
            target_true_position,
            target_estimated_position,
            vector_labels,
            ylabel="Position",
            title=f"Target Position - {target_name}",
            line_colors=vector_colors,
            estimated_label="Estimated",
        )

        # =====================================================================
        # Target velocity
        # =====================================================================

        _plot_state_comparison(
            axes[3, column],
            t,
            target_true_velocity,
            target_estimated_velocity,
            vector_labels,
            ylabel="Velocity",
            title=f"Target Velocity - {target_name}",
            line_colors=vector_colors,
            estimated_label="Estimated",
        )

        # =====================================================================
        # Target attitude
        # =====================================================================

        _plot_state_comparison(
            axes[5, column],
            t,
            target_true_attitude,
            target_estimated_attitude,
            quaternion_labels,
            ylabel="Quaternion",
            title=f"Target Attitude - {target_name}",
            line_colors=quaternion_colors,
            estimated_label="Estimated",
        )

        # =====================================================================
        # Target angular velocity
        # =====================================================================

        _plot_state_comparison(
            axes[7, column],
            t,
            target_true_angular_velocity,
            target_estimated_angular_velocity,
            angular_velocity_labels,
            ylabel="$\\omega$",
            title=f"Target Angular Velocity - {target_name}",
            line_colors=angular_velocity_colors,
            estimated_label="Estimated",
        )

        # =====================================================================
        # X axis
        # =====================================================================

        axes[-1, column].set_xlabel(
            "Time [s]"
        )

    # =========================================================================
    # Figure title
    # =========================================================================

    if len(spacecraft_names) == 1:

        title = (
            f"State Comparison - "
            f"{spacecraft_names[0]}"
        )

    else:

        title = "Spacecraft State Comparison"

    fig.suptitle(
        title,
        fontsize=13,
        y=0.995,
    )

    fig.tight_layout(
        rect=[
            0,
            0,
            1,
            0.985,
        ]
    )

    # =========================================================================
    # Finalize
    # =========================================================================

    return finalize_figure(
        fig,
        show=show,
        save=save,
        output_dir=output_dir,
        filename=filename,
        dpi=dpi,
    )