from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from py.general.data_classes_declaration import *
from py.general.general_data import *
from py.modules.attitude_dynamics import rotational_dynamics_quaternion
from py.modules.solvers.RK45 import rk45


# =============================================================================
# GNC TASKS
# =============================================================================

def compute_navigation(shared_data, sim_settings):
    """Execute navigation estimation step."""

    current_state = shared_data.actual_state.copy()

    shared_data.measured_state = (
        sim_settings.sensor.measure(current_state)
    )


def compute_guidance(shared_data, sim_settings):
    """Execute guidance computation step."""
    pass


def compute_control(shared_data, sim_settings):
    """Compute control torque command."""

    tau = sim_settings.controller.compute_control_old(shared_data)

    shared_data.tau_from_control = tau

    for actuator in sim_settings.actuators:
        actuator.set_command(shared_data.tau_from_control)


def compute_actuation(shared_data, sim_settings):
    """Convert actuator commands into applied torque."""

    tau_applied = np.zeros(3)

    for actuator in sim_settings.actuators:

        if actuator.command_mode == "PWM":
            local_time = (
                shared_data.t % actuator.modulation_window
            )
        else:
            local_time = shared_data.t

        tau_from_actuator = actuator.apply_command(local_time)

        tau_applied += tau_from_actuator

    shared_data.tau_to_apply = tau_applied


def compute_perturbations(shared_data, sim_settings):
    """Compute external disturbance torques."""

    perturbation_tau = np.zeros(3)

    for perturbation in sim_settings.perturbations:

        perturbation_tau += perturbation.compute_perturbation(
            shared_data,
            sim_settings.dynamics_settings.I
        )

    shared_data.perturbation_tau = perturbation_tau


# =============================================================================
# DYNAMICS
# =============================================================================

def propagate_dynamics(
    t,
    state,
    shared_data,
    sim_settings
):
    """Propagate rotational dynamics."""

    I = sim_settings.dynamics_settings.I
    I_inv = sim_settings.dynamics_settings.I_inv

    q_dot, w_dot = rotational_dynamics_quaternion(
        state,
        shared_data,
        I,
        I_inv,
        shared_data.perturbation_tau
    )

    return np.concatenate(
        (q_dot, w_dot)
    )


def compute_dynamics(shared_data, sim_settings):
    """Execute spacecraft attitude dynamics propagation."""

    compute_actuation(
        shared_data,
        sim_settings
    )

    compute_perturbations(
        shared_data,
        sim_settings
    )

    t_end = min(
        shared_data.t
        + sim_settings.general_settings.dt_dynamics,
        sim_settings.general_settings.total_sim_time
    )

    state0 = np.concatenate(
        (
            shared_data.actual_q,
            shared_data.actual_omega
        )
    )

    sol = rk45(
        propagate_dynamics,
        [shared_data.t, t_end],
        state0,
        args=(shared_data, sim_settings),
        h0=sim_settings.general_settings.dt_dynamics,
        h_adaptative=False
    )

    state_end = sol.y[:, -1]

    shared_data.actual_q = state_end[0:4]
    shared_data.actual_omega = state_end[4:7]


# =============================================================================
# LOGGING
# =============================================================================

def log_sim_step(shared_data, history):
    """Store current simulation state."""

    history.time.append(
        shared_data.t
    )

    history.actual_state.append(
        np.concatenate(
            (
                shared_data.actual_q,
                shared_data.actual_omega
            )
        )
    )

    history.measured_state.append(
        np.concatenate(
            (
                shared_data.measured_state[0:4],
                shared_data.measured_state[4:7]
            )
        )
    )

    history.estimated_state.append(
        np.concatenate(
            (
                shared_data.estimated_q,
                shared_data.estimated_omega
            )
        )
    )

    history.reference_state.append(
        np.concatenate(
            (
                shared_data.reference_q,
                shared_data.reference_omega
            )
        )
    )

    history.tau_control.append(
        shared_data.tau_from_control.copy()
    )

    history.tau_applied.append(
        shared_data.tau_to_apply.copy()
    )

    history.perturbation_tau.append(
        shared_data.perturbation_tau.copy()
    )


def history_to_legacy(history):
    """Convert simulation history into legacy dictionary format."""

    actual_state = np.array(
        history.actual_state
    )

    measured_state = np.array(
        history.measured_state
    )

    return {
        "time": np.array(history.time),

        "q": actual_state[:, 0:4],
        "w": actual_state[:, 4:7],

        "q_measured": measured_state[:, 0:4],
        "w_measured": measured_state[:, 4:7],

        "tau_control": np.array(history.tau_control),
        "tau_applied": np.array(history.tau_applied),

        "perturbation_tau": np.array(
            history.perturbation_tau
        ),

        "q_dot": np.zeros_like(
            actual_state[:, 0:4]
        ),

        "w_dot": np.zeros_like(
            actual_state[:, 4:7]
        ),
    }


# =============================================================================
# MAIN SIMULATION LOOP
# =============================================================================

def sim(
    sim_settings: SimulationSettings,
    desired_quat
):
    """Run spacecraft attitude simulation."""

    print("Setting simulation...")

    total_sim_time = (
        sim_settings.general_settings.total_sim_time
    )

    q0 = sim_settings.general_settings.q0
    w0 = sim_settings.general_settings.w0

    initial_state = (
        sim_settings.general_settings.initial_state
    )

    dt_nav = sim_settings.general_settings.dt_nav
    dt_guid = sim_settings.general_settings.dt_guid
    dt_control = sim_settings.general_settings.dt_control
    dt_dynamics = sim_settings.general_settings.dt_dynamics


    # -------------------------------------------------------------------------
    # Simulation timing
    # -------------------------------------------------------------------------

    dt_master = min(
        [
            dt_nav,
            dt_guid,
            dt_control,
            dt_dynamics
        ]
    )

    nav_tick = round(dt_nav / dt_master)
    guid_tick = round(dt_guid / dt_master)
    control_tick = round(dt_control / dt_master)
    dynamics_tick = round(dt_dynamics / dt_master)


    tick = 0


    # -------------------------------------------------------------------------
    # Shared simulation data
    # -------------------------------------------------------------------------

    shared_data = SimSharedData(
        t=0.0,
        tick=0,

        actual_q=q0.copy(),
        actual_omega=w0.copy(),
        actual_alpha=np.zeros(3),

        sensors_measurements={},

        reference_q=desired_quat.copy(),
        reference_omega=np.zeros(3),
        reference_alpha=np.zeros(3),

        estimated_q=np.zeros(4),
        estimated_omega=np.zeros(3),
        estimated_alpha=np.zeros(3),

        actual_state=initial_state.copy(),
        measured_state=np.zeros(initial_state.shape),
        estimated_state=np.zeros(initial_state.shape),

        reference_state=desired_quat,

        tau_from_control=np.zeros(3),
        tau_to_apply=np.zeros(3),

        perturbation_tau=np.zeros(3)
    )


    history = SimHistory(
        time=[],

        actual_state=[],
        measured_state=[],
        estimated_state=[],

        reference_state=[],

        tau_control=[],
        tau_applied=[],

        perturbation_tau=[]
    )


    # -------------------------------------------------------------------------
    # Simulation execution
    # -------------------------------------------------------------------------

    print("Running simulation...")

    while tick * dt_master < total_sim_time:

        shared_data.t = tick * dt_master


        if tick % nav_tick == 0:
            compute_navigation(
                shared_data,
                sim_settings
            )


        if tick % guid_tick == 0:
            compute_guidance(
                shared_data,
                sim_settings
            )


        if tick % control_tick == 0:
            compute_control(
                shared_data,
                sim_settings
            )


        if tick % dynamics_tick == 0:
            compute_dynamics(
                shared_data,
                sim_settings
            )


        log_sim_step(
            shared_data,
            history
        )


        tick += 1
        shared_data.tick = tick


    print("Simulation completed.")

    return history, history_to_legacy(history)