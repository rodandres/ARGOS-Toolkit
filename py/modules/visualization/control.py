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