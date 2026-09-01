from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.figure import Figure

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
SECONDARY_COLOR = "#37ee40"
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

def style_figure(fig: Figure) -> None:
    """
    Apply the shared figure background style.

    Parameters
    ----------
    fig : Figure
        Figure to style.
    """
    fig.patch.set_facecolor(FIGURE_BG)

def style_axes(
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


def style_3d_axes(
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


def finalize_figure(
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


def style_colorbar(colorbar: Any, label: str) -> None:
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