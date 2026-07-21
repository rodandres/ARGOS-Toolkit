import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.figure import Figure
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_OUTPUT_DIR = Path(
    "/run/media/afra/F4C64AC9C64A8C34/repos/RPOD_NRHO/outputs"
)

FIGURE_BG = "#0b1020"
AXES_BG = "#111a33"
SPINE_COLOR = "#2c3558"
TEXT_COLOR = "#e4e8f3"
MUTED_TEXT_COLOR = "#a5b0cf"
GRID_COLOR = "#26304d"
ACCENT_COLOR = "#ffb347"
SECONDARY_COLOR = "#7ecfff"
LINE_COLOR = "#9ad5ff"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "figure.facecolor": FIGURE_BG,
        "axes.facecolor": AXES_BG,
        "axes.edgecolor": SPINE_COLOR,
        "axes.labelcolor": TEXT_COLOR,
        "axes.titlecolor": TEXT_COLOR,
        "xtick.color": MUTED_TEXT_COLOR,
        "ytick.color": MUTED_TEXT_COLOR,
        "text.color": TEXT_COLOR,
        "grid.color": GRID_COLOR,
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,
        "legend.facecolor": AXES_BG,
        "legend.edgecolor": SPINE_COLOR,
        "savefig.facecolor": FIGURE_BG,
        "savefig.edgecolor": FIGURE_BG,
    }
)

from py.modules.cislunar_astrodynamics.lyapunov_halo_engine import (
    effective_potential,
    eigenvalues_on_L_points,
    find_L_points,
)


def _style_figure(fig: Figure) -> None:
    """
    Apply the shared figure background style.

    Parameters
    ----------
    fig : Figure
        Figure to style.
    """
    fig.patch.set_facecolor(FIGURE_BG)


def _style_axes(
    ax: Any,
    *,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    equal_aspect: bool = False,
) -> None:
    """
    Apply the shared 2D axis style.

    Parameters
    ----------
    ax : Any
        Axis to style.
    title : str | None, optional
        Axis title.
    xlabel : str | None, optional
        X-axis label.
    ylabel : str | None, optional
        Y-axis label.
    equal_aspect : bool, optional
        Force equal data aspect.
    """
    ax.set_facecolor(AXES_BG)

    for spine in ax.spines.values():
        spine.set_edgecolor(SPINE_COLOR)

    ax.tick_params(colors=MUTED_TEXT_COLOR, labelsize=8)
    ax.grid(True, color=GRID_COLOR, linewidth=0.6, linestyle="--", alpha=0.8)

    if title is not None:
        ax.set_title(title, fontsize=12, color=TEXT_COLOR, pad=10)

    if xlabel is not None:
        ax.set_xlabel(xlabel, color=TEXT_COLOR, fontsize=10)

    if ylabel is not None:
        ax.set_ylabel(ylabel, color=TEXT_COLOR, fontsize=10)

    if equal_aspect:
        ax.set_aspect("equal")


def _style_3d_axes(
    ax: Any,
    *,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    zlabel: str | None = None,
) -> None:
    """
    Apply the shared 3D axis style.

    Parameters
    ----------
    ax : Any
        3D axis to style.
    title : str | None, optional
        Axis title.
    xlabel : str | None, optional
        X-axis label.
    ylabel : str | None, optional
        Y-axis label.
    zlabel : str | None, optional
        Z-axis label.
    """
    ax.set_facecolor(AXES_BG)

    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = True
        pane.set_facecolor(AXES_BG)
        pane.set_edgecolor(SPINE_COLOR)

    ax.tick_params(colors=MUTED_TEXT_COLOR, labelsize=8)

    if title is not None:
        ax.set_title(title, fontsize=12, color=TEXT_COLOR, pad=10)

    if xlabel is not None:
        ax.set_xlabel(xlabel, color=TEXT_COLOR, fontsize=10)

    if ylabel is not None:
        ax.set_ylabel(ylabel, color=TEXT_COLOR, fontsize=10)

    if zlabel is not None:
        ax.set_zlabel(zlabel, color=TEXT_COLOR, fontsize=10)


def _finalize_figure(
    fig: Figure,
    *,
    show: bool,
    save: bool,
    output_dir: Path | str,
    filename: str,
    dpi: int,
) -> Figure:
    """
    Save or display a figure using the shared output pattern.

    Parameters
    ----------
    fig : Figure
        Figure to finalize.
    show : bool
        Display the figure.
    save : bool
        Save the figure to disk.
    output_dir : Path | str
        Output directory.
    filename : str
        Output filename.
    dpi : int
        Figure resolution in dots per inch.

    Returns
    -------
    Figure
        Finalized figure.
    """
    if save:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        fig.savefig(
            output_dir / filename,
            dpi=dpi,
            bbox_inches="tight",
            facecolor=fig.get_facecolor(),
        )

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def _style_colorbar(colorbar: Any, label: str) -> None:
    """
    Apply the shared colorbar style.

    Parameters
    ----------
    colorbar : Any
        Colorbar to style.
    label : str
        Colorbar label.
    """
    colorbar.set_label(label, color=TEXT_COLOR, fontsize=8)
    colorbar.ax.yaxis.set_tick_params(color=MUTED_TEXT_COLOR)
    plt.setp(colorbar.ax.yaxis.get_ticklabels(), color=MUTED_TEXT_COLOR, fontsize=7)


def _set_orbit_limits(
    ax: Any,
    family: list[dict[str, Any]],
    mu: float,
    L2: float,
    *,
    x_padding_factor: float = 0.06,
    y_padding_factor: float = 0.18,
    min_x_padding: float = 0.004,
    min_y_padding: float = 0.01,
) -> None:
    """
    Set balanced limits for orbit family plots.

    Parameters
    ----------
    ax : Any
        Axis to update.
    family : list[dict[str, Any]]
        Orbit family data.
    mu : float
        Mass parameter of the CR3BP.
    L2 : float
        Position of the L2 point.
    x_padding_factor : float, optional
        Padding applied to the x span.
    y_padding_factor : float, optional
        Padding applied to the y span.
    min_x_padding : float, optional
        Minimum padding on x.
    min_y_padding : float, optional
        Minimum padding on y.
    """
    x_values = np.concatenate([orbit["trajectory"][0] for orbit in family])
    y_values = np.concatenate([orbit["trajectory"][1] for orbit in family])

    x_min = min(np.min(x_values), 1 - mu, L2)
    x_max = max(np.max(x_values), 1 - mu, L2)
    y_min = np.min(y_values)
    y_max = np.max(y_values)

    x_span = max(x_max - x_min, 1e-6)
    y_span = max(y_max - y_min, 1e-6)

    x_padding = max(x_span * x_padding_factor, min_x_padding)
    y_padding = max(y_span * y_padding_factor, min_y_padding)

    ax.set_xlim(x_min - x_padding, x_max + x_padding)
    ax.set_ylim(y_min - y_padding, y_max + y_padding)


def graph_effective_potential_3d(
    mu: float,
    *,
    show: bool = True,
    save: bool = False,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    filename: str = "effective_potential_3d.png",
    dpi: int = 300,
):
    """
    Plot the CR3BP effective potential in 3D.

    Parameters
    ----------
    mu : float
        Mass parameter of the CR3BP.
    show : bool, optional
        Display the figure.
    save : bool, optional
        Save the figure to disk.
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
    x = np.linspace(-0.5, 1.5, 500)
    y = np.linspace(-0.8, 0.8, 500)
    x_grid, y_grid = np.meshgrid(x, y)

    omega = effective_potential(x_grid, y_grid, mu)
    omega = np.where(np.isfinite(omega), omega, np.nan)

    fig = plt.figure(figsize=(10, 7))
    _style_figure(fig)
    ax = fig.add_subplot(111, projection="3d")

    norm = colors.LogNorm(
        vmin=np.nanmin(omega[np.isfinite(omega)]),
        vmax=np.nanmax(omega),
    )

    surface = ax.plot_surface(
        x_grid,
        y_grid,
        omega,
        cmap="viridis",
        norm=norm,
        linewidth=0,
        antialiased=True,
    )

    _style_3d_axes(
        ax,
        title="CR3BP Effective Potential (log scale)",
        xlabel="x",
        ylabel="y",
        zlabel="Ω",
    )
    ax.set_box_aspect([1, 1, 0.6])
    ax.view_init(elev=35, azim=45)

    fig.colorbar(surface, shrink=0.5, aspect=10)
    return _finalize_figure(
        fig,
        show=show,
        save=save,
        output_dir=output_dir,
        filename=filename,
        dpi=dpi,
    )


def graph_effective_potential_2d(
    mu: float,
    *,
    show: bool = True,
    save: bool = False,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    filename: str = "effective_potential_2d.png",
    dpi: int = 300,
):
    """
    Plot the CR3BP effective potential as 2D contours.

    Parameters
    ----------
    mu : float
        Mass parameter of the CR3BP.
    show : bool, optional
        Display the figure.
    save : bool, optional
        Save the figure to disk.
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
    x = np.linspace(-0.5, 1.5, 600)
    y = np.linspace(-0.8, 0.8, 600)
    x_grid, y_grid = np.meshgrid(x, y)

    omega = effective_potential(x_grid, y_grid, mu)
    omega_log = np.log(omega)

    fig, ax = plt.subplots(figsize=(8, 6))
    _style_figure(fig)

    contour_set = ax.contourf(
        x_grid,
        y_grid,
        omega_log,
        levels=80,
        cmap="viridis",
    )

    colorbar = fig.colorbar(contour_set, ax=ax)
    _style_colorbar(colorbar, "log(Ω)")
    ax.scatter(
        [-mu, 1 - mu],
        [0, 0],
        color=ACCENT_COLOR,
        label="Primaries",
        s=35,
    )
    _style_axes(
        ax,
        title="CR3BP - Effective Potential Contours",
        xlabel="x",
        ylabel="y",
        equal_aspect=True,
    )
    ax.legend()
    return _finalize_figure(
        fig,
        show=show,
        save=save,
        output_dir=output_dir,
        filename=filename,
        dpi=dpi,
    )


def zvc_plot(
    mu: float,
    C: float,
    *,
    show: bool = True,
    save: bool = False,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    filename: str = "zvc_plot.png",
    dpi: int = 300,
):
    """
    Plot the zero-velocity curves for a given Jacobi constant.

    Parameters
    ----------
    mu : float
        Mass parameter of the CR3BP.
    C : float
        Jacobi constant.
    show : bool, optional
        Display the figure.
    save : bool, optional
        Save the figure to disk.
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
    x = np.linspace(-1.5, 1.5, 800)
    y = np.linspace(-1.2, 1.2, 800)
    x_grid, y_grid = np.meshgrid(x, y)

    omega = effective_potential(x_grid, y_grid, mu)
    zvc = 2 * omega - C

    fig, ax = plt.subplots(figsize=(8, 6))
    _style_figure(fig)

    ax.contourf(
        x_grid,
        y_grid,
        zvc,
        levels=[-1e9, 0, 1e9],
        colors=[FIGURE_BG, AXES_BG],
        alpha=1,
    )
    ax.contour(x_grid, y_grid, zvc, levels=[0], colors=TEXT_COLOR, linewidths=1.5)
    ax.scatter([-mu, 1 - mu], [0, 0], color=ACCENT_COLOR, s=50)
    _style_axes(
        ax,
        title=f"Zero Velocity Curves (C = {C:.3f})",
        xlabel="x",
        ylabel="y",
        equal_aspect=True,
    )
    return _finalize_figure(
        fig,
        show=show,
        save=save,
        output_dir=output_dir,
        filename=filename,
        dpi=dpi,
    )


def plot_L_points(
    mu: float,
    *,
    show: bool = True,
    save: bool = False,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    filename: str = "lagrange_points.png",
    dpi: int = 300,
):
    """
    Plot the collinear Lagrange points on the x-axis.

    Parameters
    ----------
    mu : float
        Mass parameter of the CR3BP.
    show : bool, optional
        Display the figure.
    save : bool, optional
        Save the figure to disk.
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
    l1, l2, l3 = find_L_points(mu)
    print(f"L1: {l1}, L2: {l2}, L3: {l3}")

    primary_1_x = -mu
    primary_2_x = 1 - mu

    fig, ax = plt.subplots(figsize=(8, 6))
    _style_figure(fig)

    x = np.linspace(-1.5, 1.5, 1000)
    y = np.zeros_like(x)

    ax.plot(x, y, color=GRID_COLOR, linestyle="--", linewidth=1.0, alpha=0.7)
    ax.scatter(primary_1_x, 0, label="Primary 1", s=100, color=SECONDARY_COLOR)
    ax.scatter(primary_2_x, 0, label="Primary 2", s=100, color=ACCENT_COLOR)
    ax.scatter([l1], [0], marker="x", s=100, label=f"L1: {l1:.4f}")
    ax.scatter([l2], [0], marker="^", s=100, label=f"L2: {l2:.4f}")
    ax.scatter([l3], [0], marker="*", s=100, label=f"L3: {l3:.4f}")
    _style_axes(
        ax,
        title="CR3BP - Lagrange Points",
        xlabel="x",
        ylabel="y",
        equal_aspect=True,
    )
    ax.legend()
    return _finalize_figure(
        fig,
        show=show,
        save=save,
        output_dir=output_dir,
        filename=filename,
        dpi=dpi,
    )


def plot_eigenvalues(
    eigvals: np.ndarray,
    title: str = "Eigenvalues in the Complex Plane",
    *,
    show: bool = True,
    save: bool = False,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    filename: str = "eigenvalues.png",
    dpi: int = 300,
):
    """
    Plot eigenvalues in the complex plane.

    Parameters
    ----------
    eigvals : np.ndarray
        Array of eigenvalues.
    title : str, optional
        Figure title.
    show : bool, optional
        Display the figure.
    save : bool, optional
        Save the figure to disk.
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
    real_part = np.real(eigvals)
    imaginary_part = np.imag(eigvals)

    fig, ax = plt.subplots(figsize=(6, 6))
    _style_figure(fig)
    ax.axhline(0, color=GRID_COLOR, linewidth=1.0)
    ax.axvline(0, color=GRID_COLOR, linewidth=1.0)
    ax.scatter(real_part, imaginary_part, color=SECONDARY_COLOR)

    for index, _value in enumerate(eigvals):
        ax.text(real_part[index], imaginary_part[index], f"{index}", fontsize=8)

    _style_axes(
        ax,
        title=title,
        xlabel="Real",
        ylabel="Imaginary",
        equal_aspect=True,
    )
    return _finalize_figure(
        fig,
        show=show,
        save=save,
        output_dir=output_dir,
        filename=filename,
        dpi=dpi,
    )


def plot_trajectory(
    sol: Any,
    mu: float,
    L_point: float | None = None,
    *,
    show: bool = True,
    save: bool = False,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    filename: str = "trajectory.png",
    dpi: int = 300,
    show_primaries: bool = False,
):
    """
    Plot a trajectory in 3D and its orthogonal projections.

    Parameters
    ----------
    sol : Any
        Numerical solution containing the trajectory states.
    mu : float
        Mass parameter of the CR3BP.
    L_point : float | None, optional
        Reference Lagrange point position on the x-axis.
    show : bool, optional
        Display the figure.
    save : bool, optional
        Save the figure to disk.
    output_dir : Path | str, optional
        Directory where the figure will be saved.
    filename : str, optional
        Output filename.
    dpi : int, optional
        Figure resolution in dots per inch.
    show_primaries : bool, optional
        Display the primaries on each subplot.

    Returns
    -------
    matplotlib.figure.Figure
        Generated figure.
    """
    x = sol.y[0]
    y = sol.y[1]
    z = sol.y[2]

    primary_1_x = -mu
    primary_2_x = 1 - mu

    fig = plt.figure(figsize=(18, 10))
    _style_figure(fig)

    # ------------------------------------------------------------------
    # 3D view
    # ------------------------------------------------------------------

    ax_3d = fig.add_subplot(221, projection="3d")
    ax_3d.plot(x, y, z)

    if L_point is not None:
        ax_3d.scatter([L_point], [0], [0], marker="x", s=80)

    _style_3d_axes(
        ax_3d,
        title="3D Trajectory",
        xlabel="x",
        ylabel="y",
        zlabel="z",
    )

    # ------------------------------------------------------------------
    # XY projection
    # ------------------------------------------------------------------

    ax_xy = fig.add_subplot(222)
    ax_xy.plot(x, y)

    if L_point is not None:
        ax_xy.scatter([L_point], [0], marker="x", s=80)

    _style_axes(ax_xy, title="XY Plane", xlabel="x", ylabel="y")

    # ------------------------------------------------------------------
    # XZ projection
    # ------------------------------------------------------------------

    ax_xz = fig.add_subplot(223)
    ax_xz.plot(x, z)

    if L_point is not None:
        ax_xz.scatter([L_point], [0], marker="x", s=80)

    _style_axes(ax_xz, title="XZ Plane", xlabel="x", ylabel="z")

    # ------------------------------------------------------------------
    # YZ projection
    # ------------------------------------------------------------------

    ax_yz = fig.add_subplot(224)
    ax_yz.plot(y, z)

    if L_point is not None:
        ax_yz.scatter([0], [L_point], marker="x", s=80)

    _style_axes(ax_yz, title="YZ Plane", xlabel="y", ylabel="z")

    if show_primaries:
        for axis in [ax_3d, ax_xy, ax_xz, ax_yz]:
            axis.scatter([primary_1_x, primary_2_x], [0, 0], s=50, color="red")

    fig.tight_layout()
    return _finalize_figure(
        fig,
        show=show,
        save=save,
        output_dir=output_dir,
        filename=filename,
        dpi=dpi,
    )


def plot_lyapunov_family(
    family: list[dict[str, Any]],
    mu: float,
    L2: float,
    *,
    show: bool = True,
    save: bool = False,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    filename: str = "lyapunov_family.png",
    dpi: int = 300,
):
    """
    Plot a family of Lyapunov orbits around L2.

    Parameters
    ----------
    family : list[dict[str, Any]]
        Orbit family data.
    mu : float
        Mass parameter of the CR3BP.
    L2 : float
        Position of the L2 point.
    show : bool, optional
        Display the figure.
    save : bool, optional
        Save the figure to disk.
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
    n_orbits = len(family)
    cmap = plt.colormaps["plasma"].resampled(n_orbits)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    _style_figure(fig)
    for axis in axes:
        axis.set_facecolor(AXES_BG)
        for spine in axis.spines.values():
            spine.set_edgecolor(SPINE_COLOR)

    # ------------------------------------------------------------------
    # Left panel: XY orbits
    # ------------------------------------------------------------------

    ax_xy = axes[0]

    for orbit in family:
        orbit_color = cmap(orbit["index"] / max(n_orbits - 1, 1))
        x = orbit["trajectory"][0]
        y = orbit["trajectory"][1]
        ax_xy.plot(x, y, color=orbit_color, linewidth=0.8, alpha=0.85)

    moon = plt.Circle((1 - mu, 0), 0.0173, color="#c8c8d8", zorder=5)
    ax_xy.add_patch(moon)
    ax_xy.text(
        1 - mu,
        -0.032,
        "Moon",
        color=TEXT_COLOR,
        fontsize=8,
        ha="center",
        va="top",
    )

    ax_xy.scatter(
        [L2],
        [0],
        marker="x",
        s=80,
        color="#ff9944",
        linewidths=1.5,
        zorder=6,
    )
    ax_xy.text(
        L2,
        -0.032,
        "L2",
        color=ACCENT_COLOR,
        fontsize=8,
        ha="center",
        va="top",
    )

    ax_xy.set_xlabel("x  [DU]", color=TEXT_COLOR, fontsize=9)
    ax_xy.set_ylabel("y  [DU]", color=TEXT_COLOR, fontsize=9)
    ax_xy.set_title(
        "Lyapunov Family — XY Plane",
        fontsize=11,
        pad=10,
    )
    ax_xy.tick_params(colors=MUTED_TEXT_COLOR)
    ax_xy.grid(True, color=GRID_COLOR, linewidth=0.6, linestyle="--")

    colorbar = fig.colorbar(
        plt.cm.ScalarMappable(cmap="plasma", norm=plt.Normalize(0, n_orbits - 1)),
        ax=ax_xy,
        fraction=0.03,
        pad=0.04,
    )
    _style_colorbar(colorbar, "Orbit index  (increasing amplitude →)")
    _set_orbit_limits(ax_xy, family, mu, L2)
    ax_xy.set_aspect("equal", adjustable="datalim")

    # ------------------------------------------------------------------
    # Right panel: period versus amplitude
    # ------------------------------------------------------------------

    ax_period = axes[1]

    amplitudes_y = [np.max(np.abs(orbit["trajectory"][1])) for orbit in family]
    periods = [orbit["full_period"] for orbit in family]
    orbit_colors = [cmap(orbit["index"] / max(n_orbits - 1, 1)) for orbit in family]

    for index in range(len(family) - 1):
        ax_period.plot(
            [amplitudes_y[index], amplitudes_y[index + 1]],
            [periods[index], periods[index + 1]],
            color=orbit_colors[index],
            linewidth=1.5,
        )

    ax_period.scatter(
        amplitudes_y,
        periods,
        c=range(n_orbits),
        cmap="plasma",
        s=18,
        zorder=5,
        edgecolors="none",
    )

    ax_period.set_xlabel("Y amplitude  [DU]", color=TEXT_COLOR, fontsize=9)
    ax_period.set_ylabel("Period T  [TU]", color=TEXT_COLOR, fontsize=9)
    ax_period.set_title(
        "Period versus Amplitude",
        fontsize=11,
        pad=10,
    )
    ax_period.tick_params(colors=MUTED_TEXT_COLOR)
    ax_period.grid(True, color=GRID_COLOR, linewidth=0.6, linestyle="--")

    t_linear = 2 * np.pi / abs(eigenvalues_on_L_points(mu)[1][0][2])
    ax_period.axhline(
        t_linear,
        color=ACCENT_COLOR,
        linewidth=0.8,
        linestyle=":",
        alpha=0.6,
    )
    ax_period.text(
        min(amplitudes_y) * 1.05,
        t_linear * 1.005,
        f"T linear = {t_linear:.3f}",
        color=ACCENT_COLOR,
        fontsize=7,
        va="bottom",
    )

    period_padding = max((max(periods) - min(periods)) * 0.08, 0.02)
    ax_period.set_xlim(min(amplitudes_y) - 0.01, max(amplitudes_y) + 0.01)
    ax_period.set_ylim(min(periods) - period_padding, max(periods) + period_padding)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.suptitle(
        f"Lyapunov Orbit Family — L2  (μ = {mu})",
        fontsize=13,
        y=0.99,
    )
    return _finalize_figure(
        fig,
        show=show,
        save=save,
        output_dir=output_dir,
        filename=filename,
        dpi=dpi,
    )


def plot_family_summary(
    family: list[dict[str, Any]],
    mu: float,
    L2: float,
    *,
    show: bool = True,
    save: bool = False,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    filename: str = "lyapunov_family_summary.png",
    dpi: int = 300,
):
    """
    Plot a compact summary of a Lyapunov orbit family in the XY plane.

    Parameters
    ----------
    family : list[dict[str, Any]]
        Orbit family data.
    mu : float
        Mass parameter of the CR3BP.
    L2 : float
        Position of the L2 point.
    show : bool, optional
        Display the figure.
    save : bool, optional
        Save the figure to disk.
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
    n_orbits = len(family)
    cmap = plt.colormaps["cool"].resampled(n_orbits)

    fig, ax = plt.subplots(figsize=(9, 7))
    _style_figure(fig)
    _style_axes(ax)

    orbit_step = max(1, n_orbits // 30)
    for orbit in family[::orbit_step]:
        orbit_color = cmap(orbit["index"] / max(n_orbits - 1, 1))
        ax.plot(
            orbit["trajectory"][0],
            orbit["trajectory"][1],
            color=orbit_color,
            linewidth=0.9,
            alpha=0.9,
        )

    moon = plt.Circle((1 - mu, 0), 0.0173, color="#dddde8", zorder=10, label="Moon")
    ax.add_patch(moon)
    ax.text(
        1 - mu,
        0,
        "Moon",
        color=TEXT_COLOR,
        fontsize=8,
        ha="center",
        va="center",
        fontweight="bold",
        zorder=11,
    )

    ax.scatter(
        [L2],
        [0],
        marker="+",
        s=120,
        color=ACCENT_COLOR,
        linewidths=2,
        zorder=10,
        label="L2",
    )
    ax.annotate(
        "L2",
        xy=(L2, 0),
        xytext=(L2 + 0.005, 0.012),
        color=ACCENT_COLOR,
        fontsize=9,
        arrowprops=dict(arrowstyle="-", color=ACCENT_COLOR, lw=0.8),
    )

    _set_orbit_limits(ax, family, mu, L2, x_padding_factor=0.16, y_padding_factor=0.35)

    ax.set_xlabel("x  [canonical units]", color=TEXT_COLOR, fontsize=10)
    ax.set_ylabel("y  [canonical units]", color=TEXT_COLOR, fontsize=10)
    ax.set_title(
        f"Lyapunov Family — L2   ({n_orbits} orbits)",
        fontsize=12,
        pad=12,
    )
    ax.tick_params(colors=MUTED_TEXT_COLOR, labelsize=8)
    ax.grid(True, color=GRID_COLOR, linewidth=0.6, linestyle="--")

    colorbar = fig.colorbar(
        plt.cm.ScalarMappable(cmap="cool", norm=plt.Normalize(0, n_orbits - 1)),
        ax=ax,
        fraction=0.025,
        pad=0.02,
    )
    _style_colorbar(colorbar, "Increasing amplitude →")
    ax.set_aspect("equal", adjustable="datalim")

    fig.tight_layout()
    return _finalize_figure(
        fig,
        show=show,
        save=save,
        output_dir=output_dir,
        filename=filename,
        dpi=dpi,
    )


def plot_beta_evolution(
    family: list[dict[str, Any]],
    *,
    show: bool = True,
    save: bool = False,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    filename: str = "bifurcation_detection.png",
    dpi: int = 300,
):
    """
    Plot the evolution of the z-mode angle and its distance to unity.

    Parameters
    ----------
    family : list[dict[str, Any]]
        Orbit family data.
    show : bool, optional
        Display the figure.
    save : bool, optional
        Save the figure to disk.
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
    amplitudes = [orbit["y_amplitude"] for orbit in family]
    beta_values = [orbit["beta_z"] for orbit in family]
    distances = [orbit["distance_to_unity"] for orbit in family]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    _style_figure(fig)

    for axis in axes:
        axis.set_facecolor(AXES_BG)
        for spine in axis.spines.values():
            spine.set_edgecolor(SPINE_COLOR)

    ax_beta = axes[0]
    ax_beta.plot(
        amplitudes,
        beta_values,
        color=SECONDARY_COLOR,
        lw=1.5,
        marker="o",
        markersize=3,
    )
    ax_beta.axhline(
        0,
        color=ACCENT_COLOR,
        lw=1.0,
        linestyle="--",
        alpha=0.7,
        label="β = 0 (bifurcation)",
    )
    ax_beta.set_xlabel("Y amplitude  [DU]", color=TEXT_COLOR)
    ax_beta.set_ylabel("β_z  [°]", color=TEXT_COLOR)
    ax_beta.set_title("Out-of-plane Mode Angle")
    ax_beta.tick_params(colors=MUTED_TEXT_COLOR)
    ax_beta.grid(True, color=GRID_COLOR, lw=0.6)
    ax_beta.legend(
        fontsize=8,
        labelcolor=TEXT_COLOR,
        edgecolor="none",
    )

    ax_distance = axes[1]
    ax_distance.semilogy(
        amplitudes,
        distances,
        color=ACCENT_COLOR,
        lw=1.5,
        marker="o",
        markersize=3,
    )
    ax_distance.axhline(
        0.05,
        color=ACCENT_COLOR,
        lw=0.8,
        linestyle=":",
        alpha=0.6,
        label="Detection threshold",
    )
    ax_distance.set_xlabel("Y amplitude  [DU]", color=TEXT_COLOR)
    ax_distance.set_ylabel("|λ_z − 1|", color=TEXT_COLOR)
    ax_distance.set_title("Eigenvalue Distance to λ = 1")
    ax_distance.tick_params(colors=MUTED_TEXT_COLOR)
    ax_distance.grid(True, color=GRID_COLOR, lw=0.6, which="both")
    ax_distance.legend(
        fontsize=8,
        labelcolor=TEXT_COLOR,
        edgecolor="none",
    )

    fig.suptitle(
        "Lyapunov → Halo Bifurcation Detection",
        fontsize=12,
    )
    fig.tight_layout()
    return _finalize_figure(
        fig,
        show=show,
        save=save,
        output_dir=output_dir,
        filename=filename,
        dpi=dpi,
    )


def plot_halo(
    sol_halo: Any,
    bif_orbit: dict[str, Any],
    mu: float,
    L2: float,
    *,
    show: bool = True,
    save: bool = False,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    filename: str = "halo_orbit.png",
    dpi: int = 300,
):
    """
    Plot a halo orbit together with its associated Lyapunov orbit.

    Parameters
    ----------
    sol_halo : Any
        Numerical solution containing the halo orbit trajectory.
    bif_orbit : dict[str, Any]
        Lyapunov orbit data under the ``"trajectory"`` key.
    mu : float
        Mass parameter of the CR3BP.
    L2 : float
        Position of the L2 point.
    show : bool, optional
        Display the figure.
    save : bool, optional
        Save the figure to disk.
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
    fig = plt.figure(figsize=(14, 10))
    _style_figure(fig)

    x_halo = sol_halo.y[0]
    y_halo = sol_halo.y[1]
    z_halo = sol_halo.y[2]

    x_lyapunov = bif_orbit["trajectory"][0]
    y_lyapunov = bif_orbit["trajectory"][1]

    # ------------------------------------------------------------------
    # 3D view
    # ------------------------------------------------------------------

    ax_3d = fig.add_subplot(221, projection="3d")

    ax_3d.plot(
        x_halo,
        y_halo,
        z_halo,
        color=SECONDARY_COLOR,
        lw=1.4,
        label="Halo Orbit",
    )
    ax_3d.plot(
        x_lyapunov,
        y_lyapunov,
        np.zeros_like(x_lyapunov),
        color=ACCENT_COLOR,
        lw=0.8,
        alpha=0.5,
        label="Lyapunov Orbit",
    )
    ax_3d.scatter([L2], [0], [0], marker="x", s=60, color=ACCENT_COLOR, lw=1.5)
    ax_3d.scatter([1 - mu], [0], [0], s=40, color=TEXT_COLOR, zorder=5)

    _style_3d_axes(
        ax_3d,
        title="3D View",
        xlabel="x",
        ylabel="y",
        zlabel="z",
    )
    ax_3d.legend(
        fontsize=7,
        labelcolor=TEXT_COLOR,
        edgecolor="none",
    )
    ax_3d.xaxis.pane.fill = False
    ax_3d.yaxis.pane.fill = False
    ax_3d.zaxis.pane.fill = False

    # ------------------------------------------------------------------
    # Orthogonal projections
    # ------------------------------------------------------------------

    projection_data = [
        (222, x_halo, y_halo, "XY Projection", "x", "y"),
        (223, x_halo, z_halo, "XZ Projection", "x", "z"),
        (224, y_halo, z_halo, "YZ Projection", "y", "z"),
    ]

    for subplot_position, x_data, y_data, title, xlabel, ylabel in projection_data:
        axis = fig.add_subplot(subplot_position)
        axis.plot(x_data, y_data, color=LINE_COLOR, lw=1.0)

        if title == "XY Projection":
            axis.plot(x_lyapunov, y_lyapunov, color=ACCENT_COLOR, lw=0.6, alpha=0.4)
            axis.scatter([L2], [0], marker="x", s=40, color=ACCENT_COLOR)

        _style_axes(axis, title=title, xlabel=xlabel, ylabel=ylabel, equal_aspect=True)

    # ------------------------------------------------------------------
    # Figure title
    # ------------------------------------------------------------------

    distance_unit_km = 384400
    y_amplitude = np.max(np.abs(y_halo))
    z_amplitude = np.max(np.abs(z_halo))

    fig.suptitle(
        (
            f"Halo Orbit around Earth-Moon L2   "
            f"Y Amplitude = {y_amplitude * distance_unit_km:.0f} km   "
            f"Z Amplitude = {z_amplitude * distance_unit_km:.0f} km"
        ),
        fontsize=11,
        y=0.99,
    )

    fig.tight_layout()
    return _finalize_figure(
        fig,
        show=show,
        save=save,
        output_dir=output_dir,
        filename=filename,
        dpi=dpi,
    )