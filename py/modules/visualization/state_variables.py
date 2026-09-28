from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from py.modules.visualization.style import *


def plot_attitude_quaternions(
    spacecraft_names: str | list[str],
    metadata,
    save: bool = False,
    show: bool = True,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    filename: str = "attitude_quaternions.png",
    dpi: int = 300,
):
    """
    Plot true, navigation, and guidance attitude quaternions
    directly from the simulation CSV files.

    Parameters
    ----------
    spacecraft_names : str | list[str]
        Spacecraft name or list of spacecraft names to plot.

    metadata : SimulationHistoryMetadata | str | Path
        Simulation metadata. Can be either:

        - A SimulationHistoryMetadata instance.
        - A path to ``simulation_history_metadata.json``.

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
    # Normalize spacecraft input
    # ------------------------------------------------------------------

    if isinstance(spacecraft_names, str):
        spacecraft_names = [spacecraft_names]

    if not spacecraft_names:
        raise ValueError(
            "At least one spacecraft name must be provided."
        )

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

        data_file_path = metadata["data_file_path"]
        available_spacecrafts = metadata["spacecraft_names"]

        # If the simulation was converted to CSV, use the metadata
        # information when available.
        csv_folder_path = metadata.get(
            "csv_folder_path",
            None,
        )

        unique_id = metadata["unique_id"]

    else:

        # SimulationHistoryMetadata instance
        data_file_path = metadata.data_file_path
        available_spacecrafts = metadata.spacecraft_names
        csv_folder_path = metadata.csv_folder_path
        unique_id = metadata.unique_id

    # ------------------------------------------------------------------
    # Determine CSV directory
    # ------------------------------------------------------------------

    if csv_folder_path:

        csv_directory = (
            Path(csv_folder_path)
            / f"simulation_{unique_id}"
        )

    else:

        # Fallback:
        # data_file_path is something like:
        #
        # sim_data/simulation_1790547517069
        #
        # Therefore use it directly if no CSV directory was specified.
        csv_directory = Path(data_file_path)

    # ------------------------------------------------------------------
    # Validate spacecrafts
    # ------------------------------------------------------------------

    for spacecraft_name in spacecraft_names:

        if spacecraft_name not in available_spacecrafts:
            raise KeyError(
                f"Spacecraft '{spacecraft_name}' not found in "
                "simulation metadata. "
                f"Available spacecrafts: {available_spacecrafts}"
            )

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------

    n_spacecrafts = len(spacecraft_names)

    fig, axes = plt.subplots(
        2,
        n_spacecrafts,
        figsize=(6 * n_spacecrafts, 8),
        sharex="col",
        squeeze=False,
    )

    style_figure(fig)

    quaternion_labels = (
        "qx",
        "qy",
        "qz",
        "qw",
    )

    true_colors = [
        LINE_COLOR,
        SECONDARY_COLOR,
        ACCENT_COLOR,
        TEXT_COLOR,
    ]

    navigation_colors = true_colors

    # ------------------------------------------------------------------
    # Plot each spacecraft
    # ------------------------------------------------------------------

    for column, spacecraft_name in enumerate(spacecraft_names):

        # --------------------------------------------------------------
        # Load CSV
        # --------------------------------------------------------------

        csv_file = (
            csv_directory
            / f"{spacecraft_name}.csv"
        )

        if not csv_file.exists():
            raise FileNotFoundError(
                f"CSV file for spacecraft '{spacecraft_name}' "
                f"not found: '{csv_file}'"
            )

        data = pd.read_csv(csv_file)

        # --------------------------------------------------------------
        # Required columns
        # --------------------------------------------------------------

        required_columns = [
            "t",

            "true_attitude_0",
            "true_attitude_1",
            "true_attitude_2",
            "true_attitude_3",

            "estimated_sc_attitude_0",
            "estimated_sc_attitude_1",
            "estimated_sc_attitude_2",
            "estimated_sc_attitude_3",

            "guidance_attitude_0",
            "guidance_attitude_1",
            "guidance_attitude_2",
            "guidance_attitude_3",
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

        # --------------------------------------------------------------
        # Extract data
        # --------------------------------------------------------------

        t = data["t"].to_numpy()

        true_q = data[
            [
                "true_attitude_0",
                "true_attitude_1",
                "true_attitude_2",
                "true_attitude_3",
            ]
        ].to_numpy()

        navigation_q = data[
            [
                "estimated_sc_attitude_0",
                "estimated_sc_attitude_1",
                "estimated_sc_attitude_2",
                "estimated_sc_attitude_3",
            ]
        ].to_numpy()

        guidance_q = data[
            [
                "guidance_attitude_0",
                "guidance_attitude_1",
                "guidance_attitude_2",
                "guidance_attitude_3",
            ]
        ].to_numpy()

        # --------------------------------------------------------------
        # Axes
        # --------------------------------------------------------------

        ax_true = axes[0, column]
        ax_navigation = axes[1, column]

        # --------------------------------------------------------------
        # True attitude
        # --------------------------------------------------------------

        for i, label in enumerate(quaternion_labels):

            ax_true.plot(
                t,
                true_q[:, i],
                color=true_colors[i],
                linewidth=1.4,
                label=f"True {label}",
            )

            ax_true.plot(
                t,
                guidance_q[:, i],
                color=true_colors[i],
                linewidth=1.0,
                linestyle="--",
                alpha=0.75,
                label=f"Guidance {label}",
            )

        style_axes(
            ax_true,
            title=f"Attitude Quaternion - {spacecraft_name}",
            ylabel="Quaternion",
        )

        ax_true.set_ylim(
            -1.05,
            1.05,
        )

        ax_true.legend(
            fontsize=7,
            ncol=2,
            labelcolor=TEXT_COLOR,
            edgecolor="none",
            facecolor=AXES_BG,
        )

        # --------------------------------------------------------------
        # Navigation attitude
        # --------------------------------------------------------------

        for i, label in enumerate(quaternion_labels):

            ax_navigation.plot(
                t,
                navigation_q[:, i],
                color=navigation_colors[i],
                linewidth=1.4,
                label=f"Navigation {label}",
            )

            ax_navigation.plot(
                t,
                guidance_q[:, i],
                color=navigation_colors[i],
                linewidth=1.0,
                linestyle="--",
                alpha=0.75,
                label=f"Guidance {label}",
            )

        style_axes(
            ax_navigation,
            title=f"Attitude Quaternion - {spacecraft_name}",
            xlabel="Time [s]",
            ylabel="Quaternion",
        )

        ax_navigation.set_ylim(
            -1.05,
            1.05,
        )

        ax_navigation.legend(
            fontsize=7,
            ncol=2,
            labelcolor=TEXT_COLOR,
            edgecolor="none",
            facecolor=AXES_BG,
        )

    # ------------------------------------------------------------------
    # Figure title
    # ------------------------------------------------------------------

    fig.suptitle(
        "Attitude Quaternion",
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

def plot_position(
    spacecraft_names: str | list[str],
    metadata,
    save: bool = False,
    show: bool = True,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    filename: str = "position.png",
    dpi: int = 300,
):
    """
    Plot true, navigation, and guidance position directly from
    simulation CSV files.

    Parameters
    ----------
    spacecraft_names : str | list[str]
        Spacecraft name or list of spacecraft names to plot.
        Each spacecraft is displayed in its own column.

    metadata : SimulationHistoryMetadata | str | Path
        Simulation metadata. Can be either:

        - A ``SimulationHistoryMetadata`` instance.
        - A path to ``simulation_history_metadata.json``.

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
    # Normalize spacecraft input
    # ------------------------------------------------------------------

    if isinstance(spacecraft_names, str):
        spacecraft_names = [spacecraft_names]

    if not spacecraft_names:
        raise ValueError(
            "At least one spacecraft name must be provided."
        )

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
    # Determine CSV directory
    # ------------------------------------------------------------------

    if csv_folder_path:

        csv_directory = (
            Path(csv_folder_path)
            / f"simulation_{unique_id}"
        )

    else:

        csv_directory = Path(data_file_path)

    # ------------------------------------------------------------------
    # Validate spacecrafts
    # ------------------------------------------------------------------

    for name in spacecraft_names:

        if name not in available_spacecrafts:
            raise KeyError(
                f"Spacecraft '{name}' not found in simulation "
                f"metadata. Available spacecrafts: "
                f"{available_spacecrafts}"
            )

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------

    n_spacecrafts = len(spacecraft_names)

    fig, axes = plt.subplots(
        3,
        n_spacecrafts,
        figsize=(6 * n_spacecrafts, 10),
        sharex="col",
        squeeze=False,
    )

    style_figure(fig)

    labels = (
        "X",
        "Y",
        "Z",
    )

    line_colors = (
        LINE_COLOR,
        SECONDARY_COLOR,
        ACCENT_COLOR,
    )

    # ------------------------------------------------------------------
    # Plot each spacecraft
    # ------------------------------------------------------------------

    for column, spacecraft_name in enumerate(spacecraft_names):

        # --------------------------------------------------------------
        # Load CSV
        # --------------------------------------------------------------

        csv_file = (
            csv_directory
            / f"{spacecraft_name}.csv"
        )

        if not csv_file.exists():
            raise FileNotFoundError(
                f"CSV file for spacecraft '{spacecraft_name}' "
                f"not found: '{csv_file}'"
            )

        data = pd.read_csv(csv_file)

        # --------------------------------------------------------------
        # Required columns
        # --------------------------------------------------------------

        required_columns = [
            "t",

            "true_position_0",
            "true_position_1",
            "true_position_2",

            "estimated_sc_position_0",
            "estimated_sc_position_1",
            "estimated_sc_position_2",

            "guidance_position_0",
            "guidance_position_1",
            "guidance_position_2",
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

        # --------------------------------------------------------------
        # Extract data
        # --------------------------------------------------------------

        t = data["t"].to_numpy()

        true_position = data[
            [
                "true_position_0",
                "true_position_1",
                "true_position_2",
            ]
        ].to_numpy()

        navigation_position = data[
            [
                "estimated_sc_position_0",
                "estimated_sc_position_1",
                "estimated_sc_position_2",
            ]
        ].to_numpy()

        guidance_position = data[
            [
                "guidance_position_0",
                "guidance_position_1",
                "guidance_position_2",
            ]
        ].to_numpy()

        # --------------------------------------------------------------
        # Plot components
        # --------------------------------------------------------------

        for i, label in enumerate(labels):

            ax = axes[i, column]

            ax.plot(
                t,
                true_position[:, i],
                color=line_colors[i],
                linewidth=1.4,
                label=f"True {label}",
            )

            ax.plot(
                t,
                navigation_position[:, i],
                color=TEXT_COLOR,
                linewidth=1.2,
                label=f"Navigation {label}",
            )

            ax.plot(
                t,
                guidance_position[:, i],
                color=ACCENT_COLOR,
                linewidth=1.0,
                linestyle="--",
                alpha=0.8,
                label=f"Guidance {label}",
            )

            style_axes(
                ax,
                title=f"Position {label}",
                ylabel=f"$r_{label.lower()}$",
            )

            ax.legend(
                fontsize=7,
                ncol=3,
                labelcolor=TEXT_COLOR,
                edgecolor="none",
                facecolor=AXES_BG,
            )

        axes[-1, column].set_xlabel(
            "Time [s]"
        )

    # ------------------------------------------------------------------
    # Figure title
    # ------------------------------------------------------------------

    fig.suptitle(
        "Position",
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

def plot_velocity(
    spacecraft_names: str | list[str],
    metadata,
    save: bool = False,
    show: bool = True,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    filename: str = "velocity.png",
    dpi: int = 300,
):
    """
    Plot true, navigation, and guidance velocity directly from
    simulation CSV files.

    Parameters
    ----------
    spacecraft_names : str | list[str]
        Spacecraft name or list of spacecraft names to plot.
        Each spacecraft is displayed in its own column.

    metadata : SimulationHistoryMetadata | str | Path
        Simulation metadata. Can be either:

        - A ``SimulationHistoryMetadata`` instance.
        - A path to ``simulation_history_metadata.json``.

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
    # Normalize spacecraft input
    # ------------------------------------------------------------------

    if isinstance(spacecraft_names, str):
        spacecraft_names = [spacecraft_names]

    if not spacecraft_names:
        raise ValueError(
            "At least one spacecraft name must be provided."
        )

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
    # Determine CSV directory
    # ------------------------------------------------------------------

    if csv_folder_path:

        csv_directory = (
            Path(csv_folder_path)
            / f"simulation_{unique_id}"
        )

    else:

        csv_directory = Path(data_file_path)

    # ------------------------------------------------------------------
    # Validate spacecrafts
    # ------------------------------------------------------------------

    for name in spacecraft_names:

        if name not in available_spacecrafts:
            raise KeyError(
                f"Spacecraft '{name}' not found in simulation "
                f"metadata. Available spacecrafts: "
                f"{available_spacecrafts}"
            )

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------

    n_spacecrafts = len(spacecraft_names)

    fig, axes = plt.subplots(
        3,
        n_spacecrafts,
        figsize=(6 * n_spacecrafts, 10),
        sharex="col",
        squeeze=False,
    )

    style_figure(fig)

    labels = (
        "X",
        "Y",
        "Z",
    )

    line_colors = (
        LINE_COLOR,
        SECONDARY_COLOR,
        ACCENT_COLOR,
    )

    # ------------------------------------------------------------------
    # Plot each spacecraft
    # ------------------------------------------------------------------

    for column, spacecraft_name in enumerate(spacecraft_names):

        # --------------------------------------------------------------
        # Load CSV
        # --------------------------------------------------------------

        csv_file = (
            csv_directory
            / f"{spacecraft_name}.csv"
        )

        if not csv_file.exists():
            raise FileNotFoundError(
                f"CSV file for spacecraft '{spacecraft_name}' "
                f"not found: '{csv_file}'"
            )

        data = pd.read_csv(csv_file)

        # --------------------------------------------------------------
        # Required columns
        # --------------------------------------------------------------

        required_columns = [
            "t",

            "true_velocity_0",
            "true_velocity_1",
            "true_velocity_2",

            "estimated_sc_velocity_0",
            "estimated_sc_velocity_1",
            "estimated_sc_velocity_2",

            "guidance_velocity_0",
            "guidance_velocity_1",
            "guidance_velocity_2",
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

        # --------------------------------------------------------------
        # Extract data
        # --------------------------------------------------------------

        t = data["t"].to_numpy()

        true_velocity = data[
            [
                "true_velocity_0",
                "true_velocity_1",
                "true_velocity_2",
            ]
        ].to_numpy()

        navigation_velocity = data[
            [
                "estimated_sc_velocity_0",
                "estimated_sc_velocity_1",
                "estimated_sc_velocity_2",
            ]
        ].to_numpy()

        guidance_velocity = data[
            [
                "guidance_velocity_0",
                "guidance_velocity_1",
                "guidance_velocity_2",
            ]
        ].to_numpy()

        # --------------------------------------------------------------
        # Plot components
        # --------------------------------------------------------------

        for i, label in enumerate(labels):

            ax = axes[i, column]

            ax.plot(
                t,
                true_velocity[:, i],
                color=line_colors[i],
                linewidth=1.4,
                label=f"True {label}",
            )

            ax.plot(
                t,
                navigation_velocity[:, i],
                color=TEXT_COLOR,
                linewidth=1.2,
                label=f"Navigation {label}",
            )

            ax.plot(
                t,
                guidance_velocity[:, i],
                color=ACCENT_COLOR,
                linewidth=1.0,
                linestyle="--",
                alpha=0.8,
                label=f"Guidance {label}",
            )

            style_axes(
                ax,
                title=f"Velocity {label}",
                ylabel=f"$v_{label.lower()}$",
            )

            ax.legend(
                fontsize=7,
                ncol=3,
                labelcolor=TEXT_COLOR,
                edgecolor="none",
                facecolor=AXES_BG,
            )

        axes[-1, column].set_xlabel(
            "Time [s]"
        )

    # ------------------------------------------------------------------
    # Figure title
    # ------------------------------------------------------------------

    fig.suptitle(
        "Velocity",
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

def plot_angular_velocity(
    spacecraft_names: str | list[str],
    metadata,
    save: bool = False,
    show: bool = True,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    filename: str = "angular_velocity.png",
    dpi: int = 300,
):
    """
    Plot true, navigation, and guidance angular velocity directly
    from simulation CSV files.

    Parameters
    ----------
    spacecraft_names : str | list[str]
        Spacecraft name or list of spacecraft names to plot.
        Each spacecraft is displayed in its own column.

    metadata : SimulationHistoryMetadata | str | Path
        Simulation metadata. Can be either:

        - A ``SimulationHistoryMetadata`` instance.
        - A path to ``simulation_history_metadata.json``.

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
    # Normalize spacecraft input
    # ------------------------------------------------------------------

    if isinstance(spacecraft_names, str):
        spacecraft_names = [spacecraft_names]

    if not spacecraft_names:
        raise ValueError(
            "At least one spacecraft name must be provided."
        )

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
    # Determine CSV directory
    # ------------------------------------------------------------------

    if csv_folder_path:

        csv_directory = (
            Path(csv_folder_path)
            / f"simulation_{unique_id}"
        )

    else:

        csv_directory = Path(data_file_path)

    # ------------------------------------------------------------------
    # Validate spacecrafts
    # ------------------------------------------------------------------

    for name in spacecraft_names:

        if name not in available_spacecrafts:
            raise KeyError(
                f"Spacecraft '{name}' not found in simulation "
                f"metadata. Available spacecrafts: "
                f"{available_spacecrafts}"
            )

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------

    n_spacecrafts = len(spacecraft_names)

    fig, axes = plt.subplots(
        3,
        n_spacecrafts,
        figsize=(6 * n_spacecrafts, 10),
        sharex="col",
        squeeze=False,
    )

    style_figure(fig)

    labels = (
        "X",
        "Y",
        "Z",
    )

    line_colors = (
        LINE_COLOR,
        SECONDARY_COLOR,
        ACCENT_COLOR,
    )

    # ------------------------------------------------------------------
    # Plot each spacecraft
    # ------------------------------------------------------------------

    for column, spacecraft_name in enumerate(spacecraft_names):

        # --------------------------------------------------------------
        # Load CSV
        # --------------------------------------------------------------

        csv_file = (
            csv_directory
            / f"{spacecraft_name}.csv"
        )

        if not csv_file.exists():
            raise FileNotFoundError(
                f"CSV file for spacecraft '{spacecraft_name}' "
                f"not found: '{csv_file}'"
            )

        data = pd.read_csv(csv_file)

        # --------------------------------------------------------------
        # Required columns
        # --------------------------------------------------------------

        required_columns = [
            "t",

            "true_angular_velocity_0",
            "true_angular_velocity_1",
            "true_angular_velocity_2",

            "estimated_sc_angular_velocity_0",
            "estimated_sc_angular_velocity_1",
            "estimated_sc_angular_velocity_2",

            "guidance_angular_velocity_0",
            "guidance_angular_velocity_1",
            "guidance_angular_velocity_2",
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

        # --------------------------------------------------------------
        # Extract data
        # --------------------------------------------------------------

        t = data["t"].to_numpy()

        true_angular_velocity = data[
            [
                "true_angular_velocity_0",
                "true_angular_velocity_1",
                "true_angular_velocity_2",
            ]
        ].to_numpy()

        navigation_angular_velocity = data[
            [
                "estimated_sc_angular_velocity_0",
                "estimated_sc_angular_velocity_1",
                "estimated_sc_angular_velocity_2",
            ]
        ].to_numpy()

        guidance_angular_velocity = data[
            [
                "guidance_angular_velocity_0",
                "guidance_angular_velocity_1",
                "guidance_angular_velocity_2",
            ]
        ].to_numpy()

        # --------------------------------------------------------------
        # Plot components
        # --------------------------------------------------------------

        for i, label in enumerate(labels):

            ax = axes[i, column]

            ax.plot(
                t,
                true_angular_velocity[:, i],
                color=line_colors[i],
                linewidth=1.4,
                label=f"True {label}",
            )

            ax.plot(
                t,
                navigation_angular_velocity[:, i],
                color=TEXT_COLOR,
                linewidth=1.2,
                label=f"Navigation {label}",
            )

            ax.plot(
                t,
                guidance_angular_velocity[:, i],
                color=ACCENT_COLOR,
                linewidth=1.0,
                linestyle="--",
                alpha=0.8,
                label=f"Guidance {label}",
            )

            style_axes(
                ax,
                title=f"Angular Velocity {label}",
                ylabel=f"$\\omega_{label.lower()}$",
            )

            ax.legend(
                fontsize=7,
                ncol=3,
                labelcolor=TEXT_COLOR,
                edgecolor="none",
                facecolor=AXES_BG,
            )

        axes[-1, column].set_xlabel(
            "Time [s]"
        )

    # ------------------------------------------------------------------
    # Figure title
    # ------------------------------------------------------------------

    fig.suptitle(
        "Angular Velocity",
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