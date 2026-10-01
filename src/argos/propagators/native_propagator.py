from argos.propagators.propagator_base import TranslationalPropagatorBase, RotationalPropagatorBase
import numpy as np

from argos.solvers.solvers_base import solve
from argos.propagators.dynamic_models import (
    quaternion_dynamics,
    cr3bp,
    rel2bp,
    newton
)

from argos import _cpp

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from argos.core.spacecraft import Spacecraft
    from argos.general.dataclasses import SimulationData
    from argos.enviroments.environment_base import EnvironmentBase

class NativeRotationalPropagator(RotationalPropagatorBase):
    """
    Propagate spacecraft rotational dynamics using a native numerical solver.

    The rotational state consists of the attitude quaternion and angular
    velocity. Quaternion dynamics are integrated together with the rigid-body
    rotational equations of motion.

    Parameters
    ----------
    integration_method : str, optional
        Numerical integration method used by the solver.
        Defaults to ``"NATIVE_RK45"``.
    """
    def __init__(self, integration_method: str = "NATIVE_RK45"):
        super().__init__(integration_method)

        if self.using_cpp:
            self.dynamic_model = "QUATERNION_DYNAMICS"
        else:
            self.dynamic_model = quaternion_dynamics

        # ================================================================
        # Profiling
        # ================================================================

        self.profile = {
            "prepare": 0.0,
            "solve": 0.0,
            "state_end": 0.0,
            "state_assignment": 0.0,
            "evaluate_dynamics": 0.0,
            "acceleration_assignment": 0.0,
            "calls": 0,
        }


    def evaluate_dynamics(
        self,
        t_end,
        state_end,
        I,
        I_inv,
        applied_torque,
        disturbance_torque,
    ):
        """
        Evaluate the rotational dynamics at a given state.

        When the C++ backend is active, the state is explicitly converted
        to a contiguous float64 NumPy array before crossing the pybind11
        boundary.
        """

        if self.using_cpp:

            state_end = np.ascontiguousarray(
                state_end,
                dtype=np.float64,
            )

            derivative = _cpp.evaluate_rotational_dynamics(
                state_end,
                t_end,
                I,
                I_inv,
                applied_torque,
                disturbance_torque,
                self.dynamic_model,
            )

            return derivative

        else:

            derivative = self.dynamic_model(
                t_end,
                state_end,
                I,
                I_inv,
                applied_torque,
                disturbance_torque,
            )

        return derivative
        

    from time import perf_counter


    def propagate(
            self,
            spacecraft: Spacecraft,
            simulation_data: SimulationData,
            environment: EnvironmentBase,
        ):
        """
        Propagate the spacecraft rotational state over the current
        propagation interval.

        Profiling is performed internally to separate:

        1. State and argument preparation.
        2. Numerical solver execution.
        3. Extraction of the final state.
        4. State assignment.
        5. Final dynamics evaluation.
        6. Angular acceleration assignment.

        Parameters
        ----------
        spacecraft : Spacecraft
            Spacecraft whose rotational state is propagated.
        simulation_data : SimulationData
            Current simulation data and maximum simulation time.
        environment : EnvironmentBase
            Environment model associated with the simulation.
        """

        # ================================================================
        # 1. Prepare state and solver arguments
        # ================================================================

        t0 = perf_counter()

        q = spacecraft.spacecraft_data.true_state.attitude
        omega = spacecraft.spacecraft_data.true_state.angular_velocity

        I = spacecraft.inertia_tensor
        I_inv = spacecraft.inertia_tensor_inv
        applied_torque = spacecraft.spacecraft_data.current_torque_exerted
        disturbance_torque = environment.get_perturbation_torque()

        state = np.concatenate(
            (
                q,
                omega,
            )
        )

        t_end = min(
            spacecraft.spacecraft_data.t
            + spacecraft.spacecraft_data.current_propagation_dt,
            simulation_data.max_sim_time,
        )

        t1 = perf_counter()

        self.profile["prepare"] += t1 - t0

        # ================================================================
        # 2. Numerical integration
        # ================================================================

        sol = solve(
            self.dynamic_model,
            [
                spacecraft.spacecraft_data.t,
                t_end,
            ],
            state,
            self.integration_method,
            args=(
                I,
                I_inv,
                applied_torque,
                disturbance_torque,
            ),
            h0=spacecraft.spacecraft_data.current_propagation_dt,
            h_adaptative=False,
        )

        t2 = perf_counter()

        self.profile["solve"] += t2 - t1

        # ================================================================
        # 3. Extract final state
        # ================================================================

        state_end = sol.y[:, -1]

        t3 = perf_counter()

        self.profile["state_end"] += t3 - t2

        # ================================================================
        # 4. Assign final attitude and angular velocity
        # ================================================================

        spacecraft.spacecraft_data.true_state.attitude = (
            state_end[0:4]
        )

        spacecraft.spacecraft_data.true_state.attitude /= (
            np.linalg.norm(
                spacecraft.spacecraft_data.true_state.attitude
            )
        )

        spacecraft.spacecraft_data.true_state.angular_velocity = (
            state_end[4:7]
        )

        t4 = perf_counter()

        self.profile["state_assignment"] += t4 - t3

        # ================================================================
        # 5. Evaluate final angular acceleration
        # ================================================================

        angular_acceleration = self.evaluate_dynamics(
            t_end,
            state_end,
            I,
            I_inv,
            applied_torque,
            disturbance_torque,
        )[4:7]

        t5 = perf_counter()

        self.profile["evaluate_dynamics"] += t5 - t4

        # ================================================================
        # 6. Assign angular acceleration
        # ================================================================

        spacecraft.spacecraft_data.true_state.angular_acceleration = (
            angular_acceleration
        )

        t6 = perf_counter()

        self.profile["acceleration_assignment"] += t6 - t5

        # ================================================================
        # Number of propagation calls
        # ================================================================

        self.profile["calls"] += 1

    # ====================================================================
    # Profiling
    # ====================================================================

    def reset_profile(self):
        """Reset all accumulated propagation profiling measurements."""

        self.profile = {
            "prepare": 0.0,
            "solve": 0.0,
            "state_end": 0.0,
            "state_assignment": 0.0,
            "evaluate_dynamics": 0.0,
            "acceleration_assignment": 0.0,
            "calls": 0,
        }

    def print_profile(self):
        """
        Print the accumulated profiling results.

        The reported percentages are relative to the total measured
        time inside ``propagate()``.
        """

        calls = self.profile["calls"]

        if calls == 0:
            print("\nNo propagation calls were recorded.")
            return

        timings = {
            key: value
            for key, value in self.profile.items()
            if key != "calls"
        }

        total = sum(timings.values())

        print()
        print("=" * 72)
        print("NativeTranslationalPropagator Profile")
        print("=" * 72)

        print(f"Calls: {calls}")
        print()

        print(
            f"{'Stage':<25}"
            f"{'Time [s]':>12}"
            f"{'Percent':>12}"
            f"{'Per call [ms]':>16}"
        )

        print("-" * 72)

        for key, value in timings.items():

            percentage = (
                100.0 * value / total
                if total > 0.0
                else 0.0
            )

            per_call = (
                1000.0 * value / calls
                if calls > 0
                else 0.0
            )

            print(
                f"{key:<25}"
                f"{value:>12.6f}"
                f"{percentage:>11.2f}%"
                f"{per_call:>16.6f}"
            )

        print("-" * 72)

        print(
            f"{'TOTAL':<25}"
            f"{total:>12.6f}"
            f"{100.0:>11.2f}%"
            f"{1000.0 * total / calls:>16.6f}"
        )

        print("=" * 72)

    def initialize(self, simulation_data):
        """
        Initialize the native rotational propagator.

        Parameters
        ----------
        simulation_data : SimulationData
            Current simulation data used during initialization.
        """
        # Implement any initialization logic needed for the native rotational propagator
        pass

import numpy as np
from time import perf_counter

SUPPORTED_DYNAMICS = ["REL2BP", "CR3BP", "NEWTON"]


class NativeTranslationalPropagator(TranslationalPropagatorBase):
    """
    Propagate spacecraft translational dynamics using a native numerical
    solver.

    The propagator supports two-body relative dynamics, the circular
    restricted three-body problem, and a pure Newtonian force model with
    externally applied forces.

    Parameters
    ----------
    orbital_model : str, optional
        Orbital dynamics model. Supported values are ``"REL2BP"``,
        ``"CR3BP"``, and ``"NEWTON"``.
    integration_method : str, optional
        Numerical integration method used by the solver.
        Defaults to ``"NATIVE_RK45"``.
    """

    def __init__(
        self,
        orbital_model: str | None = None,
        integration_method: str = "NATIVE_RK45",
    ):
        super().__init__(integration_method)

        self.dynamic_model = None
        self.orbital_model = None
        self.orbital_model_arguments = ()

        self._set_orbital_model(orbital_model)
        self._set_dynamics()

        # ================================================================
        # Profiling
        # ================================================================

        self.profile = {
            "prepare": 0.0,
            "solve": 0.0,
            "state_end": 0.0,
            "state_assignment": 0.0,
            "evaluate_dynamics": 0.0,
            "acceleration_assignment": 0.0,
            "calls": 0,
        }

    # ====================================================================
    # Configuration
    # ====================================================================

    def _set_orbital_model(self, orbital_model: str | None):
        """
        Select the translational dynamics model.

        Parameters
        ----------
        orbital_model : str, optional
            Dynamics model to use.

        Raises
        ------
        ValueError
            If the selected dynamics model is not supported.
        """

        if (
            orbital_model not in SUPPORTED_DYNAMICS
            and orbital_model is not None
        ):
            raise ValueError(
                f"Unsupported dynamics model: {orbital_model}. "
                f"Supported models are: {SUPPORTED_DYNAMICS}"
            )

        if orbital_model == "REL2BP":

            self.orbital_model = rel2bp

            if self.using_cpp:
                self.orbital_model = "REL2BP"

            mu = 6.67430e-11 * 5.972e24

            self.orbital_model_arguments = (mu,)

        elif orbital_model == "CR3BP":

            self.orbital_model = cr3bp

            if self.using_cpp:
                self.orbital_model = "CR3BP"

            # Earth-Moon CR3BP parameters
            mu = 1.215e-2
            length_factor = 384400.0e3
            time_factor = (
                27.321661 / (2.0 * np.pi) * 24 * 3600
            )

            self.orbital_model_arguments = (
                mu,
                length_factor,
                time_factor,
            )

    def _set_dynamics(self):

        if self.using_cpp:
            self.dynamic_model = "NEWTON"
        else:
            self.dynamic_model = newton

    # ====================================================================
    # Dynamics evaluation
    # ====================================================================

    def evaluate_dynamics(
        self,
        t_end,
        state_end,
        mass,
        applied_force,
        disturbance_force,
    ):
        """
        Evaluate the translational dynamics at a given state.

        When the C++ backend is active, the state is explicitly converted
        to a contiguous float64 NumPy array before crossing the pybind11
        boundary.
        """

        if self.using_cpp:

            if self.orbital_model == "CR3BP":

                mu, length_factor, time_factor = (
                    self.orbital_model_arguments
                )

            elif self.orbital_model == "REL2BP":

                mu = self.orbital_model_arguments[0]
                length_factor = 1.0
                time_factor = 1.0

            else:

                mu = 0.0
                length_factor = 1.0
                time_factor = 1.0

            state_end = np.ascontiguousarray(
                state_end,
                dtype=np.float64,
            )

            derivative = _cpp.evaluate_translational_dynamics(
                state_end,
                t_end,
                mass,
                applied_force,
                disturbance_force,
                self.dynamic_model,
                self.orbital_model,
                mu,
                length_factor,
                time_factor,
            )

            return derivative

        else:

            derivative = self.dynamic_model(
                t_end,
                state_end,
                mass,
                applied_force,
                disturbance_force,
                self.orbital_model,
                self.orbital_model_arguments,
            )

        return derivative

    # ====================================================================
    # Propagation
    # ====================================================================

    def propagate(
        self,
        spacecraft,
        simulation_data,
        environment,
    ):
        """
        Propagate the spacecraft translational state over the current
        propagation interval.

        Profiling is performed internally to separate:

        1. State and argument preparation.
        2. Numerical solver execution.
        3. Extraction of the final state.
        4. State assignment.
        5. Final dynamics evaluation.
        6. Acceleration assignment.

        Parameters
        ----------
        spacecraft : Spacecraft
            Spacecraft whose translational state is propagated.
        simulation_data : SimulationData
            Current simulation data and maximum simulation time.
        environment : EnvironmentBase
            Environment model associated with the simulation.
        """

        # ================================================================
        # 1. Prepare state and solver arguments
        # ================================================================

        t0 = perf_counter()

        position = spacecraft.spacecraft_data.true_state.position
        velocity = spacecraft.spacecraft_data.true_state.velocity

        mass = spacecraft.mass
        applied_force = spacecraft.spacecraft_data.current_force_exerted

        state = np.concatenate(
            (
                position,
                velocity,
            )
        )

        t_end = min(
            spacecraft.spacecraft_data.t
            + spacecraft.spacecraft_data.current_propagation_dt,
            simulation_data.max_sim_time,
        )

        # Keep this outside the solver call so that the allocation is
        # included explicitly in the preparation measurement.
        disturbance_force = np.zeros(3)

        t1 = perf_counter()

        self.profile["prepare"] += t1 - t0

        # ================================================================
        # 2. Numerical integration
        # ================================================================

        sol = solve(
            self.dynamic_model,
            [
                spacecraft.spacecraft_data.t,
                t_end,
            ],
            state,
            self.integration_method,
            args=(
                mass,
                applied_force,
                disturbance_force,
                self.orbital_model,
                self.orbital_model_arguments,
            ),
            h0=spacecraft.spacecraft_data.current_propagation_dt,
            h_adaptative=True,
        )

        t2 = perf_counter()

        self.profile["solve"] += t2 - t1

        # ================================================================
        # 3. Extract final state
        # ================================================================

        state_end = sol.y[:, -1]

        t3 = perf_counter()

        self.profile["state_end"] += t3 - t2

        # ================================================================
        # 4. Assign final position and velocity
        # ================================================================

        spacecraft.spacecraft_data.true_state.position = (
            state_end[0:3]
        )

        spacecraft.spacecraft_data.true_state.velocity = (
            state_end[3:6]
        )

        t4 = perf_counter()

        self.profile["state_assignment"] += t4 - t3

        # ================================================================
        # 5. Evaluate final acceleration
        # ================================================================

        acceleration = self.evaluate_dynamics(
            t_end,
            state_end,
            mass,
            applied_force,
            disturbance_force,
        )[3:6]

        t5 = perf_counter()

        self.profile["evaluate_dynamics"] += t5 - t4

        # ================================================================
        # 6. Assign acceleration
        # ================================================================

        spacecraft.spacecraft_data.true_state.acceleration = acceleration

        t6 = perf_counter()

        self.profile["acceleration_assignment"] += t6 - t5

        # ================================================================
        # Number of propagation calls
        # ================================================================

        self.profile["calls"] += 1

    # ====================================================================
    # Profiling
    # ====================================================================

    def reset_profile(self):
        """Reset all accumulated propagation profiling measurements."""

        self.profile = {
            "prepare": 0.0,
            "solve": 0.0,
            "state_end": 0.0,
            "state_assignment": 0.0,
            "evaluate_dynamics": 0.0,
            "acceleration_assignment": 0.0,
            "calls": 0,
        }

    def print_profile(self):
        """
        Print the accumulated profiling results.

        The reported percentages are relative to the total measured
        time inside ``propagate()``.
        """

        calls = self.profile["calls"]

        if calls == 0:
            print("\nNo propagation calls were recorded.")
            return

        timings = {
            key: value
            for key, value in self.profile.items()
            if key != "calls"
        }

        total = sum(timings.values())

        print()
        print("=" * 72)
        print("NativeTranslationalPropagator Profile")
        print("=" * 72)

        print(f"Calls: {calls}")
        print()

        print(
            f"{'Stage':<25}"
            f"{'Time [s]':>12}"
            f"{'Percent':>12}"
            f"{'Per call [ms]':>16}"
        )

        print("-" * 72)

        for key, value in timings.items():

            percentage = (
                100.0 * value / total
                if total > 0.0
                else 0.0
            )

            per_call = (
                1000.0 * value / calls
                if calls > 0
                else 0.0
            )

            print(
                f"{key:<25}"
                f"{value:>12.6f}"
                f"{percentage:>11.2f}%"
                f"{per_call:>16.6f}"
            )

        print("-" * 72)

        print(
            f"{'TOTAL':<25}"
            f"{total:>12.6f}"
            f"{100.0:>11.2f}%"
            f"{1000.0 * total / calls:>16.6f}"
        )

        print("=" * 72)