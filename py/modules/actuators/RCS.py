# NOTAS POR TODO LADO
# HACE FALTA IMPLEMENTAR EL GET_OUTPUT

import warnings
import numpy as np
from py.modules.actuators.actuators_base import ActuatorBase
from py.general.dataclasses import ActuatorOutput


class RCSThruster(ActuatorBase):

    def __init__(
        self,
        nominal_thrust: float,
        position: np.ndarray | None = None,
        direction: np.ndarray | None = None,
        
        command_mode: str = "PWM",        
        modulation_window: float = 0.5,
        minimum_on_time: float = 0.01,
        activation_threshold: float = 0.0,

        override_torque: bool = False,
        override_torque_value: float | None = None
    ):
        
        super().__init__()

        self.nominal_thrust = nominal_thrust
        self.position = position

        self.command_mode = command_mode

        self.modulation_window = modulation_window
        self.minimum_on_time = minimum_on_time

        self.activation_threshold = activation_threshold

        self._check_initialization()

        if self.position is None:
            self.position = np.zeros(3)  # Default position at the origin

        self.direction = direction

        if self.direction is None:
            self.direction = np.array([1, 0, 0])  # Default direction along the X-axis
        

        self.direction = self.direction / np.linalg.norm(self.direction)  # Normalize the direction vector

        self.override_torque = override_torque        

        # ==========================================================
        # Internal actuator state
        # ==========================================================

        self.on_time = 0.0
        self.command_active = False


        if self.command_mode == "PWM":
            self._compute_command = self._compute_pwm_command
            self._update_method = self._update_pwm_method

        elif self.command_mode == "bangbang":
            self._compute_command = self._compute_bangbang_command
            self._update_method = self._update_bangbang_method

        self.actual_force = np.zeros(3)  # Initialize the actual force output
        self._actual_torque = np.zeros(3)  # Initialize the actual torque output

    def _check_initialization(self):
        if self.nominal_thrust <= 0:
            raise ValueError("Nominal thrust must be positive.")

        if self.modulation_window <= 0:
            raise ValueError("Modulation window must be positive.")

        if self.minimum_on_time < 0:
            raise ValueError("Minimum on time must be non-negative.")

        if self.activation_threshold < 0:
            raise ValueError("Activation threshold must be non-negative.")

        if self.command_mode not in ["PWM", "bangbang"]:
            raise ValueError("Command mode must be either 'PWM' or 'bangbang'.")

        if self.minimum_on_time > self.modulation_window:
            raise ValueError("Minimum on time cannot be greater than modulation window.")

        if np.linalg.norm(self.direction) == 0:
                    raise ValueError("Direction vector cannot be zero.")


    def set_command(self, desired_force: float): # Saves the value of force or torque command for later application.

        if not np.isscalar(desired_force):
            raise ValueError("Desired force must be a scalar value.")


        if desired_force < 0:
            warnings.warn("Warning: Desired force is negative. Setting to zero. Check the control allocation logic.")
            desired_force = 0.0

        self._compute_command(desired_force)        

    def update(self, local_time: float) -> np.ndarray:

        if not self.command_active:
            self.actual_force = np.zeros(3)
            self._actual_torque = np.zeros(3)                            

        
            # NOTAS:
            # - esta ya no devuelve nada, todo se actualiza internamente y se puede obtener con get_output()


        return np.zeros(3)  # Default case, should not reach here

    # ==========================================================
    # Internal command models
    # ==========================================================

    def _compute_pwm_command(self, commanded_force: float):
        """
        Compute PWM valve opening time.
        """

        duty_cycle = commanded_force / self.nominal_thrust
        duty_cycle = np.clip(duty_cycle, 0.0, 1.0)

        self.on_time = (duty_cycle * self.modulation_window)
        self.command_active = True

        if self.on_time < self.minimum_on_time:
            self.on_time = 0.0
            self.command_active = False

    def _compute_bangbang_command(self, commanded_force: float):
        """
        Compute bang-bang firing state.
        """

        self.command_active = commanded_force >= self.activation_threshold

    def _update_pwm_method(self, local_time: float):
        window_time = local_time % self.modulation_window
        if window_time < self.on_time:
            return self.nominal_thrust * self.direction
        else:
            return np.zeros(3)  # No force applied outside the on-time window

        # NOTA
        # - esta función ya no devuelve nada, todo se actualiza internamente y se puede obtener con get_output()
        # - falta hacer el calculo del torque aplicado, que depende de la posición del thruster y la fuerza aplicada
        # - falta hacer el override del torque, que es un flag que permite forzar un torque fijo en lugar de calcularlo a partir de la fuerza y la posición del thruster

    def _update_bangbang_method(self):
        return self.nominal_thrust * self.direction
    
        # NOTA
        # - esta función ya no devuelve nada, todo se actualiza internamente y se puede obtener con get_output()
        # - falta hacer el calculo del torque aplicado, que depende de la posición del thruster y la fuerza aplicada
        # - falta hacer el override del torque, que es un flag que permite forzar un torque fijo en lugar de calcularlo a partir de la fuerza y la posición del thruster
        
        