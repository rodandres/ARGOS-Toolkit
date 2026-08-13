from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from py.modules.visualization.style import *


# ============================================================================
# Internal helpers
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


def _get_spacecraft_position(
    result,
    spacecraft_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Retrieve XYZ position history for a spacecraft."""

    if spacecraft_name not in result.spacecrafts_history:
        raise KeyError(
            f"Spacecraft '{spacecraft_name}' not found in simulation history."
        )

    history = result.spacecrafts_history[spacecraft_name]

    t = np.asarray(history.t)

    position = np.asarray(
        history.true_state["position"]
    )

    return (
        t,
        position[:, 0],
        position[:, 1],
        position[:, 2],
    )


def _trajectory_colors(
    n_spacecrafts: int,
):
    """Return a consistent set of trajectory colors."""

    palette = [
        LINE_COLOR,
        SECONDARY_COLOR,
        ACCENT_COLOR,
        TEXT_COLOR,
    ]

    if n_spacecrafts <= len(palette):
        return palette[:n_spacecrafts]

    cmap = plt.colormaps["tab10"].resampled(
        n_spacecrafts
    )

    return [
        cmap(i / max(n_spacecrafts - 1, 1))
        for i in range(n_spacecrafts)
    ]


def _add_trajectory_legend(ax):
    """Apply the shared legend styling."""

    legend = ax.legend(
        fontsize=8,
        labelcolor=TEXT_COLOR,
        edgecolor="none",
        facecolor=AXES_BG,
    )

    return legend


# ============================================================================
# 3D scaling
# ============================================================================

def _set_equal_3d_limits(
    ax,
    positions: list[np.ndarray],
    padding: float = 0.05,
):
    """
    Set equal physical scaling for X, Y and Z axes.

    This is required because Matplotlib does not automatically preserve
    equal physical scaling in 3D.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        3D Matplotlib axis.

    positions : list[np.ndarray]
        List of position arrays with shape (N, 3).

    padding : float, optional
        Fractional padding added around the total range.
    """

    if not positions:
        return

    all_positions = np.vstack(
        positions
    )

    x = all_positions[:, 0]
    y = all_positions[:, 1]
    z = all_positions[:, 2]

    x_min = np.min(x)
    x_max = np.max(x)

    y_min = np.min(y)
    y_max = np.max(y)

    z_min = np.min(z)
    z_max = np.max(z)

    # --------------------------------------------------------------
    # Center of the complete visualization volume
    # --------------------------------------------------------------

    center_x = 0.5 * (
        x_min + x_max
    )

    center_y = 0.5 * (
        y_min + y_max
    )

    center_z = 0.5 * (
        z_min + z_max
    )

    # --------------------------------------------------------------
    # Find the largest physical range
    # --------------------------------------------------------------

    range_x = x_max - x_min
    range_y = y_max - y_min
    range_z = z_max - z_min

    max_range = max(
        range_x,
        range_y,
        range_z,
    )

    # Prevent degenerate limits
    if max_range <= 0:
        max_range = 1.0

    # Add padding
    max_range *= (
        1.0 + padding
    )

    half_range = (
        max_range / 2.0
    )

    # --------------------------------------------------------------
    # Equal physical limits
    # --------------------------------------------------------------

    ax.set_xlim(
        center_x - half_range,
        center_x + half_range,
    )

    ax.set_ylim(
        center_y - half_range,
        center_y + half_range,
    )

    ax.set_zlim(
        center_z - half_range,
        center_z + half_range,
    )

    # --------------------------------------------------------------
    # Equal box aspect
    # --------------------------------------------------------------

    ax.set_box_aspect(
        (1, 1, 1)
    )


# ============================================================================
# Celestial bodies
# ============================================================================

BODY_DATA = {

    "Earth": {
        "radius": 6378.1363e3,
        "color": "green"
    },

    "Moon": {
        "radius": 1737.4e3,
        "color": "white"
    },

}


def _get_body_data(
    body: str,
    position: np.ndarray | None = None,
    radius: float | None = None,
    scale: float = 1.0,
):
    """
    Retrieve visualization data for a celestial body.

    Parameters
    ----------
    body : str
        Celestial body name.

    position : np.ndarray, optional
        Body position [x, y, z].

    radius : float, optional
        Custom body radius.

    scale : float, optional
        Visual scaling factor.

    Returns
    -------
    dict
        Body visualization data.
    """

    # --------------------------------------------------------------
    # Validate body
    # --------------------------------------------------------------

    body_key = body.capitalize()

    if body_key not in BODY_DATA:

        raise ValueError(
            f"Unsupported celestial body '{body}'. "
            f"Supported bodies: {list(BODY_DATA.keys())}"
        )

    # --------------------------------------------------------------
    # Default position
    # --------------------------------------------------------------

    if position is None:

        position = np.zeros(
            3
        )

    position = np.asarray(
        position,
        dtype=float,
    )

    if position.shape != (3,):

        raise ValueError(
            "Body position must be a 3-element vector [x, y, z]."
        )

    # --------------------------------------------------------------
    # Default radius
    # --------------------------------------------------------------

    if radius is None:

        radius = BODY_DATA[
            body_key
        ]["radius"]

    radius = float(
        radius
    )

    # --------------------------------------------------------------
    # Output
    # --------------------------------------------------------------

    return {

        "name": body_key,

        "position": position,

        "radius": radius * scale,

        "color": BODY_DATA[body_key]["color"],

    }


# ============================================================================
# 2D celestial body
# ============================================================================

def _plot_body_2d(
    ax,
    body_data: dict,
    plane: str,
):
    """
    Plot a celestial body as a circle in a 2D projection.
    """

    from matplotlib.patches import Circle

    position = body_data[
        "position"
    ]

    radius = body_data[
        "radius"
    ]

    # --------------------------------------------------------------
    # Select projection
    # --------------------------------------------------------------

    if plane == "xy":

        center = (
            position[0],
            position[1],
        )

    elif plane == "xz":

        center = (
            position[0],
            position[2],
        )

    elif plane == "yz":

        center = (
            position[1],
            position[2],
        )

    else:

        raise ValueError(
            f"Unsupported plane '{plane}'."
        )

    # --------------------------------------------------------------
    # Create circle
    # --------------------------------------------------------------

    circle = Circle(

        center,

        radius,

        color=body_data["color"],

        alpha=0.35,

        linewidth=1.0,

        label=body_data["name"],

    )

    ax.add_patch(
        circle
    )


# ============================================================================
# 3D celestial body
# ============================================================================

def _plot_body_3d(
    ax,
    body_data: dict,
):
    """
    Plot a celestial body as a sphere in 3D.
    """

    position = body_data[
        "position"
    ]

    radius = body_data[
        "radius"
    ]

    # --------------------------------------------------------------
    # Sphere parameterization
    # --------------------------------------------------------------

    u = np.linspace(
        0,
        2 * np.pi,
        50,
    )

    v = np.linspace(
        0,
        np.pi,
        30,
    )

    x = (
        radius
        * np.outer(
            np.cos(u),
            np.sin(v),
        )
        + position[0]
    )

    y = (
        radius
        * np.outer(
            np.sin(u),
            np.sin(v),
        )
        + position[1]
    )

    z = (
        radius
        * np.outer(
            np.ones_like(u),
            np.cos(v),
        )
        + position[2]
    )

    # --------------------------------------------------------------
    # Plot sphere
    # --------------------------------------------------------------

    ax.plot_surface(

        x,
        y,
        z,

        alpha=0.25,

        color=body_data["color"],

        linewidth=0,

        antialiased=True,

    )


# ============================================================================
# XY Plane
# ============================================================================

def plot_trajectory_xy(
    spacecraft_names: str | list[str],
    result,
    body: str | None = None,
    body_position: np.ndarray | None = None,
    body_radius: float | None = None,
    body_scale: float = 1.0,
    save: bool = False,
    show: bool = True,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    filename: str = "trajectory_xy.png",
    dpi: int = 300,
):

    spacecraft_names = _normalize_spacecraft_names(
        spacecraft_names
    )

    colors = _trajectory_colors(
        len(spacecraft_names)
    )

    fig, ax = plt.subplots(
        figsize=(8, 7)
    )

    style_figure(
        fig
    )

    # --------------------------------------------------------------
    # Body
    # --------------------------------------------------------------

    if body is not None:

        body_data = _get_body_data(
            body,
            position=body_position,
            radius=body_radius,
            scale=body_scale,
        )

        _plot_body_2d(
            ax,
            body_data,
            plane="xy",
        )

    # --------------------------------------------------------------
    # Spacecraft
    # --------------------------------------------------------------

    for spacecraft_name, color in zip(
        spacecraft_names,
        colors,
    ):

        _, x, y, _ = _get_spacecraft_position(
            result,
            spacecraft_name,
        )

        ax.plot(
            x,
            y,
            color=color,
            linewidth=1.4,
            label=spacecraft_name,
        )

    # --------------------------------------------------------------
    # Style
    # --------------------------------------------------------------

    style_axes(
        ax,
        title="Spacecraft Trajectory - XY Plane",
        xlabel="X",
        ylabel="Y",
        equal_aspect=True,
    )

    _add_trajectory_legend(
        ax
    )

    fig.tight_layout()

    return finalize_figure(
        fig,
        show=show,
        save=save,
        output_dir=output_dir,
        filename=filename,
        dpi=dpi,
    )


# ============================================================================
# XZ Plane
# ============================================================================

def plot_trajectory_xz(
    spacecraft_names: str | list[str],
    result,
    body: str | None = None,
    body_position: np.ndarray | None = None,
    body_radius: float | None = None,
    body_scale: float = 1.0,
    save: bool = False,
    show: bool = True,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    filename: str = "trajectory_xz.png",
    dpi: int = 300,
):

    spacecraft_names = _normalize_spacecraft_names(
        spacecraft_names
    )

    colors = _trajectory_colors(
        len(spacecraft_names)
    )

    fig, ax = plt.subplots(
        figsize=(8, 7)
    )

    style_figure(
        fig
    )

    # --------------------------------------------------------------
    # Body
    # --------------------------------------------------------------

    if body is not None:

        body_data = _get_body_data(
            body,
            position=body_position,
            radius=body_radius,
            scale=body_scale,
        )

        _plot_body_2d(
            ax,
            body_data,
            plane="xz",
        )

    # --------------------------------------------------------------
    # Spacecraft
    # --------------------------------------------------------------

    for spacecraft_name, color in zip(
        spacecraft_names,
        colors,
    ):

        _, x, _, z = _get_spacecraft_position(
            result,
            spacecraft_name,
        )

        ax.plot(
            x,
            z,
            color=color,
            linewidth=1.4,
            label=spacecraft_name,
        )

    # --------------------------------------------------------------
    # Style
    # --------------------------------------------------------------

    style_axes(
        ax,
        title="Spacecraft Trajectory - XZ Plane",
        xlabel="X",
        ylabel="Z",
        equal_aspect=True,
    )

    _add_trajectory_legend(
        ax
    )

    fig.tight_layout()

    return finalize_figure(
        fig,
        show=show,
        save=save,
        output_dir=output_dir,
        filename=filename,
        dpi=dpi,
    )


# ============================================================================
# YZ Plane
# ============================================================================

def plot_trajectory_yz(
    spacecraft_names: str | list[str],
    result,
    body: str | None = None,
    body_position: np.ndarray | None = None,
    body_radius: float | None = None,
    body_scale: float = 1.0,
    save: bool = False,
    show: bool = True,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    filename: str = "trajectory_yz.png",
    dpi: int = 300,
):

    spacecraft_names = _normalize_spacecraft_names(
        spacecraft_names
    )

    colors = _trajectory_colors(
        len(spacecraft_names)
    )

    fig, ax = plt.subplots(
        figsize=(8, 7)
    )

    style_figure(
        fig
    )

    # --------------------------------------------------------------
    # Body
    # --------------------------------------------------------------

    if body is not None:

        body_data = _get_body_data(
            body,
            position=body_position,
            radius=body_radius,
            scale=body_scale,
        )

        _plot_body_2d(
            ax,
            body_data,
            plane="yz",
        )

    # --------------------------------------------------------------
    # Spacecraft
    # --------------------------------------------------------------

    for spacecraft_name, color in zip(
        spacecraft_names,
        colors,
    ):

        _, _, y, z = _get_spacecraft_position(
            result,
            spacecraft_name,
        )

        ax.plot(
            y,
            z,
            color=color,
            linewidth=1.4,
            label=spacecraft_name,
        )

    # --------------------------------------------------------------
    # Style
    # --------------------------------------------------------------

    style_axes(
        ax,
        title="Spacecraft Trajectory - YZ Plane",
        xlabel="Y",
        ylabel="Z",
        equal_aspect=True,
    )

    _add_trajectory_legend(
        ax
    )

    fig.tight_layout()

    return finalize_figure(
        fig,
        show=show,
        save=save,
        output_dir=output_dir,
        filename=filename,
        dpi=dpi,
    )


# ============================================================================
# 3D XYZ
# ============================================================================

def plot_trajectory_3d(
    spacecraft_names: str | list[str],
    result,
    body: str | None = None,
    body_position: np.ndarray | None = None,
    body_radius: float | None = None,
    body_scale: float = 1.0,
    save: bool = False,
    show: bool = True,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    filename: str = "trajectory_3d.png",
    dpi: int = 300,
):

    spacecraft_names = _normalize_spacecraft_names(
        spacecraft_names
    )

    colors = _trajectory_colors(
        len(spacecraft_names)
    )

    fig = plt.figure(
        figsize=(9, 8)
    )

    style_figure(
        fig
    )

    ax = fig.add_subplot(
        111,
        projection="3d",
    )

    # --------------------------------------------------------------
    # Storage for equal scaling
    # --------------------------------------------------------------

    positions = []

    # --------------------------------------------------------------
    # Body
    # --------------------------------------------------------------

    body_data = None

    if body is not None:

        body_data = _get_body_data(
            body,
            position=body_position,
            radius=body_radius,
            scale=body_scale,
        )

        _plot_body_3d(
            ax,
            body_data,
        )

        # Include entire body in scaling
        body_position_array = body_data[
            "position"
        ]

        body_radius_value = body_data[
            "radius"
        ]

        positions.append(
            np.array(
                [
                    body_position_array
                    - body_radius_value,

                    body_position_array
                    + body_radius_value,
                ]
            )
        )

    # --------------------------------------------------------------
    # Spacecraft trajectories
    # --------------------------------------------------------------

    for spacecraft_name, color in zip(
        spacecraft_names,
        colors,
    ):

        _, x, y, z = _get_spacecraft_position(
            result,
            spacecraft_name,
        )

        position = np.column_stack(
            (
                x,
                y,
                z,
            )
        )

        positions.append(
            position
        )

        ax.plot(
            x,
            y,
            z,
            color=color,
            linewidth=1.4,
            label=spacecraft_name,
        )

    # --------------------------------------------------------------
    # Equal 3D scaling
    # --------------------------------------------------------------

    _set_equal_3d_limits(
        ax,
        positions,
    )

    # --------------------------------------------------------------
    # Style
    # --------------------------------------------------------------

    style_3d_axes(
        ax,
        title="Spacecraft Trajectory - 3D",
        xlabel="X",
        ylabel="Y",
        zlabel="Z",
    )

    _add_trajectory_legend(
        ax
    )

    fig.tight_layout()

    return finalize_figure(
        fig,
        show=show,
        save=save,
        output_dir=output_dir,
        filename=filename,
        dpi=dpi,
    )


# ============================================================================
# Combined trajectory figure
# ============================================================================

def plot_trajectory(
    spacecraft_names: str | list[str],
    result,
    body: str | None = None,
    body_position: np.ndarray | None = None,
    body_radius: float | None = None,
    body_scale: float = 1.0,
    save: bool = False,
    show: bool = True,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    filename: str = "trajectory.png",
    dpi: int = 300,
):
    """
    Plot spacecraft trajectories in a combined 3D and planar view.

    The left side contains the XY, XZ and YZ projections.
    The right side contains the 3D trajectory.

    Parameters
    ----------
    spacecraft_names : str | list[str]
        Spacecraft name or list of spacecraft names.

    result : SimulationHistory
        General simulation result containing spacecraft histories.

    body : str, optional
        Celestial body to display.
        Supported: "Earth", "Moon".

    body_position : np.ndarray, optional
        Body position [x, y, z].

    body_radius : float, optional
        Custom celestial body radius.

    body_scale : float, optional
        Visual scaling factor applied to the body radius.

    save : bool, optional
        Save the generated figure.

    show : bool, optional
        Display the generated figure.

    output_dir : Path | str, optional
        Output directory.

    filename : str, optional
        Output filename.

    dpi : int, optional
        Figure resolution.

    Returns
    -------
    matplotlib.figure.Figure
        Generated figure.
    """

    spacecraft_names = _normalize_spacecraft_names(
        spacecraft_names
    )

    colors = _trajectory_colors(
        len(spacecraft_names)
    )

    # --------------------------------------------------------------
    # Figure
    # --------------------------------------------------------------

    fig = plt.figure(
        figsize=(14, 10)
    )

    style_figure(
        fig
    )

    grid = fig.add_gridspec(
        3,
        2,
        width_ratios=[
            1,
            1.15,
        ],
        hspace=0.25,
        wspace=0.12,
    )

    ax_xy = fig.add_subplot(
        grid[0, 0]
    )

    ax_xz = fig.add_subplot(
        grid[1, 0]
    )

    ax_yz = fig.add_subplot(
        grid[2, 0]
    )

    ax_3d = fig.add_subplot(
        grid[:, 1],
        projection="3d",
    )

    # --------------------------------------------------------------
    # Storage for 3D equal scaling
    # --------------------------------------------------------------

    positions_3d = []

    # --------------------------------------------------------------
    # Body
    # --------------------------------------------------------------

    body_data = None

    if body is not None:

        body_data = _get_body_data(
            body,
            position=body_position,
            radius=body_radius,
            scale=body_scale,
        )

        # 2D
        _plot_body_2d(
            ax_xy,
            body_data,
            plane="xy",
        )

        _plot_body_2d(
            ax_xz,
            body_data,
            plane="xz",
        )

        _plot_body_2d(
            ax_yz,
            body_data,
            plane="yz",
        )

        # 3D
        _plot_body_3d(
            ax_3d,
            body_data,
        )

        # Include entire body in 3D limits
        body_position_array = body_data[
            "position"
        ]

        body_radius_value = body_data[
            "radius"
        ]

        positions_3d.append(
            np.array(
                [
                    body_position_array
                    - body_radius_value,

                    body_position_array
                    + body_radius_value,
                ]
            )
        )

    # --------------------------------------------------------------
    # Spacecraft trajectories
    # --------------------------------------------------------------

    for spacecraft_name, color in zip(
        spacecraft_names,
        colors,
    ):

        _, x, y, z = _get_spacecraft_position(
            result,
            spacecraft_name,
        )

        # ----------------------------------------------------------
        # Save 3D position data
        # ----------------------------------------------------------

        position = np.column_stack(
            (
                x,
                y,
                z,
            )
        )

        positions_3d.append(
            position
        )

        # ----------------------------------------------------------
        # XY
        # ----------------------------------------------------------

        ax_xy.plot(
            x,
            y,
            color=color,
            linewidth=1.3,
            label=spacecraft_name,
        )

        # ----------------------------------------------------------
        # XZ
        # ----------------------------------------------------------

        ax_xz.plot(
            x,
            z,
            color=color,
            linewidth=1.3,
        )

        # ----------------------------------------------------------
        # YZ
        # ----------------------------------------------------------

        ax_yz.plot(
            y,
            z,
            color=color,
            linewidth=1.3,
        )

        # ----------------------------------------------------------
        # 3D
        # ----------------------------------------------------------

        ax_3d.plot(
            x,
            y,
            z,
            color=color,
            linewidth=1.3,
            label=spacecraft_name,
        )

    # --------------------------------------------------------------
    # Planar styles
    # --------------------------------------------------------------

    style_axes(
        ax_xy,
        title="XY Plane",
        xlabel="X",
        ylabel="Y",
        equal_aspect=True,
    )

    style_axes(
        ax_xz,
        title="XZ Plane",
        xlabel="X",
        ylabel="Z",
        equal_aspect=True,
    )

    style_axes(
        ax_yz,
        title="YZ Plane",
        xlabel="Y",
        ylabel="Z",
        equal_aspect=True,
    )

    # --------------------------------------------------------------
    # 3D equal scaling
    # --------------------------------------------------------------

    _set_equal_3d_limits(
        ax_3d,
        positions_3d,
    )

    # --------------------------------------------------------------
    # 3D style
    # --------------------------------------------------------------

    style_3d_axes(
        ax_3d,
        title="3D Trajectory",
        xlabel="X",
        ylabel="Y",
        zlabel="Z",
    )

    # --------------------------------------------------------------
    # Legend
    # --------------------------------------------------------------

    _add_trajectory_legend(
        ax_xy
    )

    # --------------------------------------------------------------
    # Figure title
    # --------------------------------------------------------------

    if len(spacecraft_names) == 1:

        title = (
            f"Spacecraft Trajectory - "
            f"{spacecraft_names[0]}"
        )

    else:

        title = (
            "Spacecraft Trajectories"
        )

    fig.suptitle(
        title,
        fontsize=13,
        y=0.99,
    )

    fig.tight_layout(
        rect=[
            0,
            0,
            1,
            0.96,
        ]
    )

    # --------------------------------------------------------------
    # Finalize
    # --------------------------------------------------------------

    return finalize_figure(
        fig,
        show=show,
        save=save,
        output_dir=output_dir,
        filename=filename,
        dpi=dpi,
    )