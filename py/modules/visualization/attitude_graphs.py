from pathlib import Path
from typing import Any
import sys

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from py.modules.controllers import PDAttitudeController
from py.modules.math import euler_from_quaternion, quaternion_error, quaternion_to_DCM

FIGURE_BG = "#0b1020"
AXES_BG = "#111a33"
SPINE_COLOR = "#2c3558"
TEXT_COLOR = "#e4e8f3"
MUTED_TEXT_COLOR = "#a5b0cf"
GRID_COLOR = "#26304d"

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


def _can_show_plots() -> bool:
    return not plt.get_backend().lower().startswith(("agg", "pdf", "ps", "svg", "cairo", "pgf", "template"))


def _style_figure(fig: Figure) -> None:
    fig.patch.set_facecolor(FIGURE_BG)


def _style_2d_axes(
    ax: Any,
    *,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
) -> None:
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


def _style_3d_axes(
    ax: Any,
    *,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    zlabel: str | None = None,
) -> None:
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


def _show_or_close(fig: Figure | None = None) -> None:
    if _can_show_plots():
        plt.show()
    elif fig is not None:
        plt.close(fig)


def plot_frames(q1: np.ndarray, q2: np.ndarray) -> None:
    origin = np.zeros(3)
    basis = np.eye(3)
    basis1 = quaternion_to_DCM(q1) @ basis
    basis2 = quaternion_to_DCM(q2) @ basis
    euler1 = euler_from_quaternion(q1)
    euler2 = euler_from_quaternion(q2)

    fig = plt.figure(figsize=(8, 8))
    _style_figure(fig)
    ax = fig.add_subplot(111, projection="3d")
    for index, color in enumerate(["r", "g", "b"]):
        ax.quiver(*origin, *basis[:, index], color=color, linestyle="dashed", alpha=0.3)
        ax.quiver(*origin, *basis1[:, index], color=color, linewidth=2, label=f"Initial axis {index}")
        ax.quiver(*origin, *basis2[:, index], color=color, linestyle="dotted", linewidth=2, label=f"Target axis {index}")
    title = (
        "Attitude Comparison\n"
        f"Initial Euler [deg]: R={np.rad2deg(euler1[0]):.1f}, P={np.rad2deg(euler1[1]):.1f}, Y={np.rad2deg(euler1[2]):.1f}\n"
        f"Target Euler [deg]: R={np.rad2deg(euler2[0]):.1f}, P={np.rad2deg(euler2[1]):.1f}, Y={np.rad2deg(euler2[2]):.1f}"
    )
    _style_3d_axes(ax, title=title, xlabel="X", ylabel="Y", zlabel="Z")
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_box_aspect([1, 1, 1])
    ax.legend()
    plt.tight_layout()
    _show_or_close(fig)


def animate_attitude(sol: Any, args: dict[str, Any], interval: int = 50):
    t = sol.t
    q_hist = sol.y[0:4, :]
    q_hist = q_hist / np.linalg.norm(q_hist, axis=0)
    desired_axes = quaternion_to_DCM(args["desired_quat"]) @ np.eye(3)

    fig = plt.figure(figsize=(8, 8))
    _style_figure(fig)
    ax = fig.add_subplot(111, projection="3d")

    def update(frame: int) -> None:
        ax.cla()
        body_axes = quaternion_to_DCM(q_hist[:, frame]) @ np.eye(3)
        origin = np.zeros(3)
        basis = np.eye(3)
        for index, color in enumerate(["r", "g", "b"]):
            ax.quiver(*origin, *basis[:, index], color=color, linestyle="dashed", alpha=0.3)
            ax.quiver(*origin, *desired_axes[:, index], color=color, linestyle=":", alpha=0.6)
            ax.quiver(*origin, *body_axes[:, index], color=color, linewidth=2)
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        ax.set_box_aspect([1, 1, 1])
        ax.set_title(f"Attitude Animation\n t = {t[frame]:.2f} s")

    if not _can_show_plots():
        plt.close(fig)
        return None

    animation = FuncAnimation(fig, update, frames=len(t), interval=interval, repeat=True)
    _show_or_close(fig)
    return animation


def plot_results(sol: Any, args: dict[str, Any]) -> None:
    t = sol.t
    q_hist = sol.y[0:4, :] / np.linalg.norm(sol.y[0:4, :], axis=0)
    w_hist = sol.y[4:7, :]
    qd = args["desired_quat"]

    qe_hist = []
    for index in range(q_hist.shape[1]):
        qe = quaternion_error(qd, q_hist[:, index])
        if qe[3] < 0:
            qe = -qe
        qe_hist.append(qe)
    qe_hist = np.array(qe_hist).T
    theta_deg = np.rad2deg(2 * np.arccos(np.clip(qe_hist[3], -1, 1)))
    omega_norm = np.linalg.norm(w_hist, axis=0)
    tau_hist = np.array([PDAttitudeController(np.concatenate((q_hist[:, i], w_hist[:, i])), args) for i in range(q_hist.shape[1])]).T

    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    _style_figure(fig)
    for axis in axes:
        _style_2d_axes(axis)
    axes[0].plot(t, theta_deg)
    axes[0].axhline(0.5, linestyle="--")
    axes[0].set_title("Attitude Error θ [deg]")
    axes[0].set_ylabel("Degrees")
    axes[1].plot(t, omega_norm)
    axes[1].axhline(0.01, linestyle="--")
    axes[1].set_title("Angular Velocity Norm ||ω|| [rad/s]")
    axes[1].set_ylabel("rad/s")
    axes[2].plot(t, tau_hist[0], label="τx")
    axes[2].plot(t, tau_hist[1], label="τy")
    axes[2].plot(t, tau_hist[2], label="τz")
    axes[2].axhline(args["tau_max"], linestyle="--")
    axes[2].axhline(args["tau_min"], linestyle="--")
    axes[2].set_title("Control Torque [Nm]")
    axes[2].set_ylabel("Nm")
    axes[2].legend()
    axes[3].plot(t, w_hist[0], label="wx")
    axes[3].plot(t, w_hist[1], label="wy")
    axes[3].plot(t, w_hist[2], label="wz")
    axes[3].set_title("Angular Velocity Components [rad/s]")
    axes[3].set_xlabel("Time [s]")
    axes[3].set_ylabel("rad/s")
    axes[3].legend()
    plt.tight_layout()
    _show_or_close(fig)


def plot_results_2(sol: Any, args: dict[str, Any], fun_evaluated: Any) -> None:
    t = sol.t
    q_hist = sol.y[0:4, :] / np.linalg.norm(sol.y[0:4, :], axis=0)
    w_hist = sol.y[4:7, :]
    qd = args["Control_args"]["Desired_quat"]
    qd = qd / np.linalg.norm(qd)

    q_hist_aligned = q_hist.copy()
    for index in range(q_hist.shape[1]):
        if np.dot(q_hist[:, index], qd) < 0:
            q_hist_aligned[:, index] = -q_hist[:, index]
    qd_line = np.tile(qd.reshape(4, 1), (1, len(t)))

    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    _style_figure(fig)
    for axis in axes:
        _style_2d_axes(axis)
    labels = ["qx", "qy", "qz", "qw"]
    for index in range(4):
        axes[index].plot(t, q_hist_aligned[index], label="Actual")
        axes[index].plot(t, qd_line[index], linestyle="--", label="Desired")
        axes[index].set_ylabel(labels[index])
        if index == 0:
            axes[index].set_title("Quaternion Components vs Desired")
            axes[index].legend()
        if index == 3:
            axes[index].set_xlabel("Time [s]")
    plt.tight_layout()
    _show_or_close(fig)

    args_new = args.copy()
    args_new["Extended_output"] = True
    taus = []
    torque_applied_hist = []
    for index in range(len(t)):
        state = np.concatenate((q_hist[:, index], w_hist[:, index]))
        result = fun_evaluated(t[index], state, args_new)
        taus.append(result["tau_control"])
        torque_applied_hist.append(result["apply_control"] / 100)
    fig, ax = plt.subplots(figsize=(10, 6))
    _style_figure(fig)
    _style_2d_axes(ax, title="Control Torques Over Time", xlabel="Time [s]", ylabel="Torque [Nm]")
    ax.plot(t, np.array(taus), label=["τx", "τy", "τz"], linewidth=2, alpha=0.7, marker="o")
    ax.plot(t, torque_applied_hist, label="Torque Applied", marker="x", color="k")
    ax.legend()
    _show_or_close(fig)


def plot_quaternions(sol: Any, args: dict[str, Any]) -> None:
    t = sol.t
    q_hist = sol.y[0:4, :] / np.linalg.norm(sol.y[0:4, :], axis=0)
    qd = args["Control_args"]["Desired_quat"]
    qd = qd / np.linalg.norm(qd)
    q_hist_aligned = q_hist.copy()
    for index in range(q_hist.shape[1]):
        if np.dot(q_hist[:, index], qd) < 0:
            q_hist_aligned[:, index] = -q_hist[:, index]
    qd_line = np.tile(qd.reshape(4, 1), (1, len(t)))
    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    _style_figure(fig)
    for axis in axes:
        _style_2d_axes(axis)
    labels = ["qx", "qy", "qz", "qw"]
    for index in range(4):
        axes[index].plot(t, q_hist_aligned[index], label="Actual")
        axes[index].plot(t, qd_line[index], linestyle="--", label="Desired")
        axes[index].set_ylabel(labels[index])
        if index == 0:
            axes[index].set_title("Quaternion Components vs Desired")
            axes[index].legend()
        if index == 3:
            axes[index].set_xlabel("Time [s]")
    plt.tight_layout()
    _show_or_close(fig)


def plot_state(sol: Any, args: dict[str, Any], fun_evaluated: Any) -> None:
    t = sol.t
    q_hist = sol.y[0:4, :] / np.linalg.norm(sol.y[0:4, :], axis=0)
    w_hist = sol.y[4:7, :]
    qd = args["Control_args"]["Desired_quat"]
    qd = qd / np.linalg.norm(qd)
    qs_org = np.zeros((len(t), 4))
    ws_org = np.zeros((len(t), 3))
    qs_control = np.zeros((len(t), 4))
    ws_control = np.zeros((len(t), 3))
    args_new = args.copy()
    args_new["Extended_output"] = True
    for index in range(len(t)):
        state = np.concatenate((q_hist[:, index], w_hist[:, index]))
        result = fun_evaluated(t[index], state, args_new)
        qs_org[index, :] = result["q"]
        ws_org[index, :] = result["w"]
        qs_control[index, :] = result["q_control"]
        ws_control[index, :] = result["w_control"]
    q_org_aligned = qs_org.copy()
    q_control_aligned = qs_control.copy()
    for index in range(len(t)):
        if np.dot(q_hist[:, index], qd) < 0:
            q_org_aligned[index] = -q_hist[:, index]
            q_control_aligned[index] = -q_hist[:, index]
    qd_line = np.tile(qd.reshape(4, 1), (1, len(t)))
    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    _style_figure(fig)
    for axis in axes:
        _style_2d_axes(axis)
    labels = ["qx", "qy", "qz", "qw"]
    for index in range(4):
        axes[index].plot(t, q_org_aligned[:, index], label="Actual")
        axes[index].plot(t, q_control_aligned[:, index], label="Control Input")
        axes[index].plot(t, qd_line[index], linestyle="--", label="Desired")
        axes[index].set_ylabel(labels[index])
        if index == 0:
            axes[index].set_title("Quaternion Components vs Desired")
            axes[index].legend()
        if index == 3:
            axes[index].set_xlabel("Time [s]")
    plt.tight_layout()
    _show_or_close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    _style_figure(fig)
    for axis in axes:
        _style_2d_axes(axis)
    for index in range(3):
        axes[index].plot(t, ws_org[:, index], label="Actual")
        axes[index].plot(t, ws_control[:, index], label="Control Input")
        axes[index].set_title(f"Angular Velocity Component w{index} [rad/s]")
        axes[index].set_ylabel("rad/s")
        axes[index].legend()
        if index == 2:
            axes[index].set_xlabel("Time [s]")
    plt.tight_layout()
    _show_or_close(fig)


def plot_torques(sol: Any, args: dict[str, Any], fun_evaluated: Any) -> None:
    ts = sol.t
    q_hist = sol.y[0:4, :]
    w_hist = sol.y[4:7, :]
    args_new = args.copy()
    args_new["Extended_output"] = True
    control_taus = []
    control_tau_applied_bool = []
    tau_applied = []
    srp_taus = []
    grav_grad_taus = []
    srp_active = False
    gravity_gradient_active = False
    if "Perturbations" in args and args["Perturbations"] is not None:
        perturbations = args["Perturbations"]
        srp_active = perturbations.get("SRP_active", False)
        gravity_gradient_active = perturbations.get("Gravity_gradient_active", False)
    for index in range(len(ts)):
        state = np.concatenate((q_hist[:, index], w_hist[:, index]))
        result = fun_evaluated(ts[index], state, args_new)
        control_taus.append(result["tau_control"])
        control_tau_applied_bool.append(result["apply_control_bool"])
        tau_applied.append(result["tau_applied"])
        if srp_active:
            srp_taus.append(result["SRP_torque"])
        if gravity_gradient_active:
            grav_grad_taus.append(result["grav_grad_torque"])
    control_taus = np.array(control_taus).T
    applied_taus = np.array(tau_applied).T
    applied = np.array(control_tau_applied_bool).astype(bool)
    fig, ax = plt.subplots(figsize=(10, 6))
    _style_figure(fig)
    _style_2d_axes(ax, title="Control Torque [Nm]", xlabel="Time [s]", ylabel="Nm")
    ax.plot(ts, control_taus[0], label="τx", color="r")
    ax.plot(ts, control_taus[1], label="τy", color="g")
    ax.plot(ts, control_taus[2], label="τz", color="b")
    ax.scatter(ts[applied], control_taus[0][applied], marker="x", color="r", label="Applied τx")
    ax.scatter(ts[applied], control_taus[1][applied], marker="*", color="g", label="Applied τy")
    ax.scatter(ts[applied], control_taus[2][applied], marker="^", color="b", label="Applied τz")
    ax.plot(ts, applied_taus[0], linestyle="--", color="r", label="Torque Applied τx")
    ax.plot(ts, applied_taus[1], linestyle="--", color="g", label="Torque Applied τy")
    ax.plot(ts, applied_taus[2], linestyle="--", color="b", label="Torque Applied τz")
    ax.legend()
    plt.tight_layout()
    _show_or_close(fig)
    if srp_active:
        fig, ax = plt.subplots(figsize=(10, 6))
        _style_figure(fig)
        _style_2d_axes(ax, title="Solar Radiation Pressure Torque [Nm]", xlabel="Time [s]", ylabel="Nm")
        srp_taus = np.array(srp_taus).T
        ax.plot(ts, srp_taus[0], label="SRP τx")
        ax.plot(ts, srp_taus[1], label="SRP τy")
        ax.plot(ts, srp_taus[2], label="SRP τz")
        ax.legend()
        plt.tight_layout()
        _show_or_close(fig)
    if gravity_gradient_active:
        fig, ax = plt.subplots(figsize=(10, 6))
        _style_figure(fig)
        _style_2d_axes(ax, title="Gravity Gradient Torque [Nm]", xlabel="Time [s]", ylabel="Nm")
        grav_grad_taus = np.array(grav_grad_taus).T
        ax.plot(ts, grav_grad_taus[0], label="Gravity Gradient τx")
        ax.plot(ts, grav_grad_taus[1], label="Gravity Gradient τy")
        ax.plot(ts, grav_grad_taus[2], label="Gravity Gradient τz")
        ax.legend()
        plt.tight_layout()
        _show_or_close(fig)


def plot_states_NEW(results: dict[str, np.ndarray], args: dict[str, Any]) -> None:
    t = results["time"]
    q = results["q"] / np.linalg.norm(results["q"], axis=1, keepdims=True)
    w = results["w"]
    q_measured = results["q_measured"] / np.linalg.norm(results["q_measured"], axis=1, keepdims=True)
    w_measured = results["w_measured"]
    q_desired = args["Controller"]["Desired quat"]
    q_desired = q_desired / np.linalg.norm(q_desired)
    q_aligned = q.copy()
    q_measured_aligned = q_measured.copy()
    for index in range(len(t)):
        if np.dot(q[index], q_desired) < 0:
            q_aligned[index] = -q[index]
        if np.dot(q_measured[index], q_desired) < 0:
            q_measured_aligned[index] = -q_measured[index]
    q_desired_line = np.tile(q_desired, (len(t), 1))
    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    _style_figure(fig)
    for axis in axes:
        _style_2d_axes(axis)
    labels_q = ["qx", "qy", "qz", "qw"]
    for index in range(4):
        axes[index].plot(t, q_aligned[:, index], label="Actual")
        axes[index].plot(t, q_measured_aligned[:, index], linestyle=":", label="Measured")
        axes[index].plot(t, q_desired_line[:, index], linestyle="--", label="Desired")
        axes[index].set_ylabel(labels_q[index])
        if index == 0:
            axes[index].set_title("Quaternion Components vs Time")
            axes[index].legend()
        if index == 3:
            axes[index].set_xlabel("Time [s]")
    plt.tight_layout()
    _show_or_close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    _style_figure(fig)
    for axis in axes:
        _style_2d_axes(axis)
    labels_w = ["wx", "wy", "wz"]
    for index in range(3):
        axes[index].plot(t, w[:, index], label="Actual")
        axes[index].plot(t, w_measured[:, index], linestyle=":", label="Measured")
        axes[index].set_ylabel("rad/s")
        axes[index].set_title(f"{labels_w[index]} vs Time")
        axes[index].legend()
        if index == 2:
            axes[index].set_xlabel("Time [s]")
    plt.tight_layout()
    _show_or_close(fig)


def plot_torques_NEW(results: dict[str, np.ndarray]) -> None:
    t = results["time"]
    tau_control = results["tau_control"]
    tau_applied = results["tau_applied"]
    labels = ["τx", "τy", "τz"]
    colors = ["r", "g", "b"]
    fig, axes = plt.subplots(3, 1, figsize=(12, 6), sharex=True)
    _style_figure(fig)
    for axis in axes:
        _style_2d_axes(axis)
    for index in range(3):
        axes[index].plot(t, tau_control[:, index], color=colors[index], label=f"Control {labels[index]}")
        axes[index].plot(t, tau_applied[:, index], linestyle="--", color=colors[index], label=f"Applied {labels[index]}")
        axes[index].set_ylabel("Nm")
        axes[index].set_title(f"{labels[index]} vs Time")
        axes[index].legend()
        if index == 2:
            axes[index].set_xlabel("Time [s]")
    plt.tight_layout()
    _show_or_close(fig)


def animate_attitude_NEW(results: dict[str, np.ndarray], args: dict[str, Any], interval: int = 20, frame_skip: int = 10):
    t = results["time"]
    q_hist = results["q"] / np.linalg.norm(results["q"], axis=1, keepdims=True)
    q_desired = args["Controller"]["Desired quat"]
    q_desired = q_desired / np.linalg.norm(q_desired)
    r_desired = quaternion_to_DCM(q_desired)
    fig = plt.figure(figsize=(8, 8))
    _style_figure(fig)
    ax = fig.add_subplot(111, projection="3d")
    _style_3d_axes(ax, title="Attitude Animation", xlabel="X", ylabel="Y", zlabel="Z")

    def update(frame: int) -> None:
        ax.cla()
        body_axes = quaternion_to_DCM(q_hist[frame]) @ np.eye(3)
        origin = np.zeros(3)
        basis = np.eye(3)
        for index, color in enumerate(["r", "g", "b"]):
            ax.quiver(*origin, *basis[:, index], color=color, linestyle="dashed", alpha=0.3)
            ax.quiver(*origin, *r_desired[:, index], color=color, linestyle=":", alpha=0.6)
            ax.quiver(*origin, *body_axes[:, index], color=color, linewidth=2)
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        ax.set_title(f"Attitude Animation\n t = {t[frame]:.2f} s")
        ax.set_box_aspect([1, 1, 1])

    if not _can_show_plots():
        plt.close(fig)
        return None
    animation = FuncAnimation(fig, update, frames=range(0, len(t), frame_skip), interval=interval, repeat=True)
    _show_or_close(fig)
    return animation
