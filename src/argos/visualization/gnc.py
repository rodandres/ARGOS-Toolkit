from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json

from py.modules.visualization.style import *

def plot_control_result(
    spacecraft_name: str,
    metadata,
    save: bool = False,
    show: bool = True,
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

    metadata : SimulationHistoryMetadata | str | Path
        Simulation metadata. Can be either a ``SimulationHistoryMetadata``
        instance or a path to ``simulation_history_metadata.json``.

    save : bool, optional
        Save the generated figure to disk. Default is False.

    show : bool, optional
        Display the generated figure. Default is True.

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
    # Load metadata
    # ------------------------------------------------------------------

    if isinstance(metadata, (str, Path)):

        metadata_path = Path(metadata)

        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata file not found: '{metadata_path}'"
            )

        with open(
            metadata_path,
            "r",
            encoding="utf-8",
        ) as file:
            metadata = json.load(file)

        unique_id = metadata["unique_id"]
        available_spacecrafts = metadata["spacecraft_names"]
        data_file_path = metadata["data_file_path"]
        csv_folder_path = metadata.get(
            "csv_folder_path",
            None,
        )

    else:

        unique_id = metadata.unique_id
        available_spacecrafts = metadata.spacecraft_names
        data_file_path = metadata.data_file_path
        csv_folder_path = metadata.csv_folder_path

    # ------------------------------------------------------------------
    # Validate spacecraft
    # ------------------------------------------------------------------

    if spacecraft_name not in available_spacecrafts:
        raise KeyError(
            f"Spacecraft '{spacecraft_name}' not found in simulation "
            f"metadata. Available spacecrafts: "
            f"{available_spacecrafts}"
        )

    # ------------------------------------------------------------------
    # Determine CSV directory
    # ------------------------------------------------------------------

    if csv_folder_path:

        csv_directory = (
            Path(csv_folder_path)
            / f"simulation_{unique_id}"
        )

    else:

        csv_directory = Path(data_file_path)

    csv_file = (
        csv_directory
        / f"{spacecraft_name}.csv"
    )

    if not csv_file.exists():
        raise FileNotFoundError(
            f"CSV file for spacecraft '{spacecraft_name}' "
            f"not found: '{csv_file}'"
        )

    # ------------------------------------------------------------------
    # Load CSV
    # ------------------------------------------------------------------

    data = pd.read_csv(csv_file)

    required_columns = [
        "t",

        "control_force_0",
        "control_force_1",
        "control_force_2",

        "current_force_exerted_0",
        "current_force_exerted_1",
        "current_force_exerted_2",

        "control_torque_0",
        "control_torque_1",
        "control_torque_2",

        "current_torque_exerted_0",
        "current_torque_exerted_1",
        "current_torque_exerted_2",
    ]

    missing_columns = [
        column_name
        for column_name in required_columns
        if column_name not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"CSV file '{csv_file}' is missing required "
            f"columns: {missing_columns}"
        )

    # ------------------------------------------------------------------
    # Retrieve data
    # ------------------------------------------------------------------

    t = data["t"].to_numpy()

    commanded_force = data[
        [
            "control_force_0",
            "control_force_1",
            "control_force_2",
        ]
    ].to_numpy()

    applied_force = data[
        [
            "current_force_exerted_0",
            "current_force_exerted_1",
            "current_force_exerted_2",
        ]
    ].to_numpy()

    commanded_torque = data[
        [
            "control_torque_0",
            "control_torque_1",
            "control_torque_2",
        ]
    ].to_numpy()

    applied_torque = data[
        [
            "current_torque_exerted_0",
            "current_torque_exerted_1",
            "current_torque_exerted_2",
        ]
    ].to_numpy()

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
            ylabel=rf"$F_{{{label.lower()}}}$ [N]",
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
            ylabel=rf"$\tau_{{{label.lower()}}}$ [N·m]",
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

    fig.tight_layout(
        rect=[0, 0, 1, 0.96]
    )

    # ------------------------------------------------------------------
    # Finalize
    # ------------------------------------------------------------------

    return finalize_figure(
        fig,
        show=show,
        save=save,
        output_dir=output_dir,
        filename=filename,
        dpi=dpi,
    )

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
    metadata,
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
    # Load metadata
    # =========================================================================

    if isinstance(metadata, (str, Path)):

        metadata_path = Path(metadata)

        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata file not found: '{metadata_path}'"
            )

        with open(
            metadata_path,
            "r",
            encoding="utf-8",
        ) as file:
            metadata = json.load(file)

        unique_id = metadata["unique_id"]
        available_spacecrafts = metadata["spacecraft_names"]
        data_file_path = metadata["data_file_path"]
        csv_folder_path = metadata.get(
            "csv_folder_path",
            None,
        )
        targets_info = metadata.get(
            "targets_info",
            {},
        )

    else:

        unique_id = metadata.unique_id
        available_spacecrafts = metadata.spacecraft_names
        data_file_path = metadata.data_file_path
        csv_folder_path = metadata.csv_folder_path
        targets_info = getattr(
            metadata,
            "targets_info",
            {},
        )

    # =========================================================================
    # Determine CSV directory
    # =========================================================================

    if csv_folder_path:

        csv_directory = (
            Path(csv_folder_path)
            / f"simulation_{unique_id}"
        )

    else:

        csv_directory = Path(data_file_path)

    # =========================================================================
    # Validate spacecrafts
    # =========================================================================

    for name in spacecraft_names:

        if name not in available_spacecrafts:
            raise KeyError(
                f"Spacecraft '{name}' not found in simulation metadata. "
                f"Available spacecrafts: {available_spacecrafts}"
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

        # =====================================================================
        # Load spacecraft CSV
        # =====================================================================

        csv_file = (
            csv_directory
            / f"{spacecraft_name}.csv"
        )

        if not csv_file.exists():
            raise FileNotFoundError(
                f"CSV file for spacecraft '{spacecraft_name}' "
                f"not found: '{csv_file}'"
            )

        data = pd.read_csv(
            csv_file
        )

        # =====================================================================
        # Required own-state columns
        # =====================================================================

        required_columns = [
            "t",

            "true_position_0",
            "true_position_1",
            "true_position_2",

            "estimated_sc_position_0",
            "estimated_sc_position_1",
            "estimated_sc_position_2",

            "true_velocity_0",
            "true_velocity_1",
            "true_velocity_2",

            "estimated_sc_velocity_0",
            "estimated_sc_velocity_1",
            "estimated_sc_velocity_2",

            "true_attitude_0",
            "true_attitude_1",
            "true_attitude_2",
            "true_attitude_3",

            "estimated_sc_attitude_0",
            "estimated_sc_attitude_1",
            "estimated_sc_attitude_2",
            "estimated_sc_attitude_3",

            "true_angular_velocity_0",
            "true_angular_velocity_1",
            "true_angular_velocity_2",

            "estimated_sc_angular_velocity_0",
            "estimated_sc_angular_velocity_1",
            "estimated_sc_angular_velocity_2",
        ]

        missing_columns = [
            column_name
            for column_name in required_columns
            if column_name not in data.columns
        ]

        if missing_columns:
            raise ValueError(
                f"CSV file '{csv_file}' is missing required "
                f"columns: {missing_columns}"
            )

        # =====================================================================
        # Time
        # =====================================================================

        t = data[
            "t"
        ].to_numpy()

        # =====================================================================
        # Own spacecraft state
        # =====================================================================

        true_position = np.column_stack(
            [
                data["true_position_0"].to_numpy(),
                data["true_position_1"].to_numpy(),
                data["true_position_2"].to_numpy(),
            ]
        )

        navigation_position = np.column_stack(
            [
                data["estimated_sc_position_0"].to_numpy(),
                data["estimated_sc_position_1"].to_numpy(),
                data["estimated_sc_position_2"].to_numpy(),
            ]
        )

        true_velocity = np.column_stack(
            [
                data["true_velocity_0"].to_numpy(),
                data["true_velocity_1"].to_numpy(),
                data["true_velocity_2"].to_numpy(),
            ]
        )

        navigation_velocity = np.column_stack(
            [
                data["estimated_sc_velocity_0"].to_numpy(),
                data["estimated_sc_velocity_1"].to_numpy(),
                data["estimated_sc_velocity_2"].to_numpy(),
            ]
        )

        true_attitude = np.column_stack(
            [
                data["true_attitude_0"].to_numpy(),
                data["true_attitude_1"].to_numpy(),
                data["true_attitude_2"].to_numpy(),
                data["true_attitude_3"].to_numpy(),
            ]
        )

        navigation_attitude = np.column_stack(
            [
                data["estimated_sc_attitude_0"].to_numpy(),
                data["estimated_sc_attitude_1"].to_numpy(),
                data["estimated_sc_attitude_2"].to_numpy(),
                data["estimated_sc_attitude_3"].to_numpy(),
            ]
        )

        true_angular_velocity = np.column_stack(
            [
                data["true_angular_velocity_0"].to_numpy(),
                data["true_angular_velocity_1"].to_numpy(),
                data["true_angular_velocity_2"].to_numpy(),
            ]
        )

        navigation_angular_velocity = np.column_stack(
            [
                data["estimated_sc_angular_velocity_0"].to_numpy(),
                data["estimated_sc_angular_velocity_1"].to_numpy(),
                data["estimated_sc_angular_velocity_2"].to_numpy(),
            ]
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

        spacecraft_targets = targets_info.get(
            spacecraft_name,
            [],
        )

        # ---------------------------------------------------------------------
        # No target
        # ---------------------------------------------------------------------

        if not spacecraft_targets:

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
        # Target
        # ---------------------------------------------------------------------

        target_name = spacecraft_targets[0]

        # ---------------------------------------------------------------------
        # Validate target
        # ---------------------------------------------------------------------

        if target_name not in available_spacecrafts:

            raise KeyError(
                f"Target spacecraft '{target_name}' referenced by "
                f"'{spacecraft_name}' was not found in simulation metadata."
            )

        target_csv_file = (
            csv_directory
            / f"{target_name}.csv"
        )

        if not target_csv_file.exists():
            raise FileNotFoundError(
                f"CSV file for target spacecraft '{target_name}' "
                f"not found: '{target_csv_file}'"
            )

        target_data = pd.read_csv(
            target_csv_file
        )

        # =====================================================================
        # Required target truth columns
        # =====================================================================

        target_required_columns = [
            "t",

            "true_position_0",
            "true_position_1",
            "true_position_2",

            "true_velocity_0",
            "true_velocity_1",
            "true_velocity_2",

            "true_attitude_0",
            "true_attitude_1",
            "true_attitude_2",
            "true_attitude_3",

            "true_angular_velocity_0",
            "true_angular_velocity_1",
            "true_angular_velocity_2",
        ]

        missing_target_columns = [
            column_name
            for column_name in target_required_columns
            if column_name not in target_data.columns
        ]

        if missing_target_columns:
            raise ValueError(
                f"CSV file '{target_csv_file}' is missing required "
                f"columns: {missing_target_columns}"
            )

        # =====================================================================
        # Target time
        # =====================================================================

        target_t = target_data[
            "t"
        ].to_numpy()

        # =====================================================================
        # Target true state
        # =====================================================================

        target_true_position = np.column_stack(
            [
                target_data["true_position_0"].to_numpy(),
                target_data["true_position_1"].to_numpy(),
                target_data["true_position_2"].to_numpy(),
            ]
        )

        target_true_velocity = np.column_stack(
            [
                target_data["true_velocity_0"].to_numpy(),
                target_data["true_velocity_1"].to_numpy(),
                target_data["true_velocity_2"].to_numpy(),
            ]
        )

        target_true_attitude = np.column_stack(
            [
                target_data["true_attitude_0"].to_numpy(),
                target_data["true_attitude_1"].to_numpy(),
                target_data["true_attitude_2"].to_numpy(),
                target_data["true_attitude_3"].to_numpy(),
            ]
        )

        target_true_angular_velocity = np.column_stack(
            [
                target_data["true_angular_velocity_0"].to_numpy(),
                target_data["true_angular_velocity_1"].to_numpy(),
                target_data["true_angular_velocity_2"].to_numpy(),
            ]
        )

        # =====================================================================
        # Target estimated state
        # =====================================================================

        target_estimated_columns = [
            "estimated_target_position_0",
            "estimated_target_position_1",
            "estimated_target_position_2",

            "estimated_target_velocity_0",
            "estimated_target_velocity_1",
            "estimated_target_velocity_2",

            "estimated_target_attitude_0",
            "estimated_target_attitude_1",
            "estimated_target_attitude_2",
            "estimated_target_attitude_3",

            "estimated_target_angular_velocity_0",
            "estimated_target_angular_velocity_1",
            "estimated_target_angular_velocity_2",
        ]

        missing_estimated_columns = [
            column_name
            for column_name in target_estimated_columns
            if column_name not in data.columns
        ]

        if missing_estimated_columns:
            raise ValueError(
                f"CSV file '{csv_file}' is missing required target "
                f"estimation columns: {missing_estimated_columns}"
            )

        target_estimated_position = np.column_stack(
            [
                data["estimated_target_position_0"].to_numpy(),
                data["estimated_target_position_1"].to_numpy(),
                data["estimated_target_position_2"].to_numpy(),
            ]
        )

        target_estimated_velocity = np.column_stack(
            [
                data["estimated_target_velocity_0"].to_numpy(),
                data["estimated_target_velocity_1"].to_numpy(),
                data["estimated_target_velocity_2"].to_numpy(),
            ]
        )

        target_estimated_attitude = np.column_stack(
            [
                data["estimated_target_attitude_0"].to_numpy(),
                data["estimated_target_attitude_1"].to_numpy(),
                data["estimated_target_attitude_2"].to_numpy(),
                data["estimated_target_attitude_3"].to_numpy(),
            ]
        )

        target_estimated_angular_velocity = np.column_stack(
            [
                data["estimated_target_angular_velocity_0"].to_numpy(),
                data["estimated_target_angular_velocity_1"].to_numpy(),
                data["estimated_target_angular_velocity_2"].to_numpy(),
            ]
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