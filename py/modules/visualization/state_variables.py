from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from py.modules.visualization.style import *

def plot_attitude_quaternions(
    spacecraft_names: str | list[str],
    result,
    save: bool = False,
    show: bool = True,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    filename: str = "attitude_quaternions.png",
    dpi: int = 300,
):
    """
    Plot true, navigation, and guidance attitude quaternions.

    Parameters
    ----------
    spacecraft_names : str | list[str]
        Spacecraft name or list of spacecraft names to plot.
        Each spacecraft is displayed in its own column.
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
    # Normalize spacecraft input
    # ------------------------------------------------------------------

    if isinstance(spacecraft_names, str):
        spacecraft_names = [spacecraft_names]

    if not spacecraft_names:
        raise ValueError("At least one spacecraft name must be provided.")

    # ------------------------------------------------------------------
    # Validate spacecrafts
    # ------------------------------------------------------------------

    for name in spacecraft_names:
        if name not in result.spacecrafts_history:
            raise KeyError(
                f"Spacecraft '{name}' not found in simulation history."
            )

    n_spacecrafts = len(spacecraft_names)

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------

    fig, axes = plt.subplots(
        2,
        n_spacecrafts,
        figsize=(6 * n_spacecrafts, 8),
        sharex="col",
        squeeze=False,
    )

    style_figure(fig)

    quaternion_labels = ("qx", "qy", "qz", "qw")

    # Use the shared palette consistently
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

        history = result.spacecrafts_history[spacecraft_name]

        t = np.asarray(history.t)

        true_q = np.asarray(
            history.true_state["attitude"]
        )

        navigation_q = np.asarray(
            history.estimated_data["spacecraft_state"]["attitude"]
        )

        guidance_q = np.asarray(
            history.reference_data["state"]["attitude"]
        )

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

        ax_true.set_ylim(-1.05, 1.05)

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

        ax_navigation.set_ylim(-1.05, 1.05)

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

    fig.tight_layout(rect=[0, 0, 1, 0.96])

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
    result,
    save: bool = False,
    show: bool = True,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    filename: str = "position.png",
    dpi: int = 300,
):
    """
    Plot true, navigation, and guidance position.

    Parameters
    ----------
    spacecraft_names : str | list[str]
        Spacecraft name or list of spacecraft names to plot.
        Each spacecraft is displayed in its own column.
    result : SimulationHistory
        General simulation result containing spacecraft histories.
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

    if isinstance(spacecraft_names, str):
        spacecraft_names = [spacecraft_names]

    if not spacecraft_names:
        raise ValueError("At least one spacecraft name must be provided.")

    for name in spacecraft_names:
        if name not in result.spacecrafts_history:
            raise KeyError(
                f"Spacecraft '{name}' not found in simulation history."
            )

    n_spacecrafts = len(spacecraft_names)

    fig, axes = plt.subplots(
        3,
        n_spacecrafts,
        figsize=(6 * n_spacecrafts, 10),
        sharex="col",
        squeeze=False,
    )

    style_figure(fig)

    labels = ("X", "Y", "Z")
    line_colors = (
        LINE_COLOR,
        SECONDARY_COLOR,
        ACCENT_COLOR,
    )

    for column, spacecraft_name in enumerate(spacecraft_names):

        history = result.spacecrafts_history[spacecraft_name]

        t = np.asarray(history.t)

        true_position = np.asarray(
            history.true_state["position"]
        )

        navigation_position = np.asarray(
            history.estimated_data["spacecraft_state"]["position"]
        )

        guidance_position = np.asarray(
            history.reference_data["state"]["position"]
        )

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

        axes[-1, column].set_xlabel("Time [s]")

    fig.suptitle(
        "Position",
        fontsize=13,
        y=0.99,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.96])

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
    result,
    save: bool = False,
    show: bool = True,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    filename: str = "velocity.png",
    dpi: int = 300,
):
    """
    Plot true, navigation, and guidance velocity.

    Parameters
    ----------
    spacecraft_names : str | list[str]
        Spacecraft name or list of spacecraft names to plot.
        Each spacecraft is displayed in its own column.
    result : SimulationHistory
        General simulation result containing spacecraft histories.
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

    if isinstance(spacecraft_names, str):
        spacecraft_names = [spacecraft_names]

    if not spacecraft_names:
        raise ValueError("At least one spacecraft name must be provided.")

    for name in spacecraft_names:
        if name not in result.spacecrafts_history:
            raise KeyError(
                f"Spacecraft '{name}' not found in simulation history."
            )

    n_spacecrafts = len(spacecraft_names)

    fig, axes = plt.subplots(
        3,
        n_spacecrafts,
        figsize=(6 * n_spacecrafts, 10),
        sharex="col",
        squeeze=False,
    )

    style_figure(fig)

    labels = ("X", "Y", "Z")
    line_colors = (
        LINE_COLOR,
        SECONDARY_COLOR,
        ACCENT_COLOR,
    )

    for column, spacecraft_name in enumerate(spacecraft_names):

        history = result.spacecrafts_history[spacecraft_name]

        t = np.asarray(history.t)

        true_velocity = np.asarray(
            history.true_state["velocity"]
        )

        navigation_velocity = np.asarray(
            history.estimated_data["spacecraft_state"]["velocity"]
        )

        guidance_velocity = np.asarray(
            history.reference_data["state"]["velocity"]
        )

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

        axes[-1, column].set_xlabel("Time [s]")

    fig.suptitle(
        "Velocity",
        fontsize=13,
        y=0.99,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.96])

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
    result,
    save: bool = False,
    show: bool = True,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    filename: str = "angular_velocity.png",
    dpi: int = 300,
):
    """
    Plot true, navigation, and guidance angular velocity.

    Parameters
    ----------
    spacecraft_names : str | list[str]
        Spacecraft name or list of spacecraft names to plot.
        Each spacecraft is displayed in its own column.
    result : SimulationHistory
        General simulation result containing spacecraft histories.
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

    if isinstance(spacecraft_names, str):
        spacecraft_names = [spacecraft_names]

    if not spacecraft_names:
        raise ValueError("At least one spacecraft name must be provided.")

    for name in spacecraft_names:
        if name not in result.spacecrafts_history:
            raise KeyError(
                f"Spacecraft '{name}' not found in simulation history."
            )

    n_spacecrafts = len(spacecraft_names)

    fig, axes = plt.subplots(
        3,
        n_spacecrafts,
        figsize=(6 * n_spacecrafts, 10),
        sharex="col",
        squeeze=False,
    )

    style_figure(fig)

    labels = ("X", "Y", "Z")
    line_colors = (
        LINE_COLOR,
        SECONDARY_COLOR,
        ACCENT_COLOR,
    )

    for column, spacecraft_name in enumerate(spacecraft_names):

        history = result.spacecrafts_history[spacecraft_name]

        t = np.asarray(history.t)

        true_angular_velocity = np.asarray(
            history.true_state["angular_velocity"]
        )

        navigation_angular_velocity = np.asarray(
            history.estimated_data["spacecraft_state"]["angular_velocity"]
        )

        guidance_angular_velocity = np.asarray(
            history.reference_data["state"]["angular_velocity"]
        )

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
                ylabel=rf"$\omega_{label.lower()}$",
            )

            ax.legend(
                fontsize=7,
                ncol=3,
                labelcolor=TEXT_COLOR,
                edgecolor="none",
                facecolor=AXES_BG,
            )

        axes[-1, column].set_xlabel("Time [s]")

    fig.suptitle(
        "Angular Velocity",
        fontsize=13,
        y=0.99,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    return finalize_figure(
        fig,
        show=show,
        save=save,
        output_dir=output_dir,
        filename=filename,
        dpi=dpi,
    )