import numpy as np
from argos.controllers.controller_base import ControllerBase, ControlAllocatorBase
from argos.general.dataclasses import ControlOutput

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from argos.general.dataclasses import ControlOutput, GuidanceOutput, NavigationOutput
    from argos.actuators.actuators_base import ActuatorBase

class CustomController(ControllerBase):
    """
    Controller that delegates control computation to a user-defined function.

    Parameters
    ----------
    control_function : callable
        Function used to compute the control output. It receives the
        navigation output and guidance output as arguments and must return
        a ``ControlOutput`` instance.
    """
    def __init__(self, control_function):
        self.control_function = control_function

        super().__init__()
        # Initialize any additional attributes if needed

    def _check_initialization(self):
        if not callable(self.control_function):
            raise ValueError("control_function must be a callable function.")

    def compute_control(self, estimated_state: NavigationOutput, guidance_output: GuidanceOutput) -> ControlOutput:
        """
        Compute the control output using the user-defined control function.

        Parameters
        ----------
        estimated_state : NavigationOutput
            Estimated spacecraft state provided to the control function.
        guidance_output : GuidanceOutput
            Guidance information provided to the control function.

        Returns
        -------
        ControlOutput
            Control output returned by the user-defined control function.

        Raises
        ------
        TypeError
            If ``control_function`` does not return a ``ControlOutput`` instance.
        """

        output = self.control_function(estimated_state, guidance_output)

        if not isinstance(output, ControlOutput):
            raise TypeError("The control_function must return a ControlOutput object.")        

        return output

class CustomControlAllocator(ControlAllocatorBase):
    """
    Control allocator that delegates actuator allocation to a user-defined function.

    Parameters
    ----------
    allocation_function : callable
        Function used to allocate the control output. It receives the
        ``ControlOutput`` and the allocator's actuator list.
    """
    def __init__(self, allocation_function):
        self.allocation_function = allocation_function

        super().__init__()
        # Initialize any additional attributes if needed

    def _check_initialization(self):
        if not callable(self.allocation_function):
            raise ValueError("allocation_function must be a callable function.")

    def allocate(self, control_output: ControlOutput):
        """
        Allocate a control output using the user-defined allocation function.

        Parameters
        ----------
        control_output : ControlOutput
            Control force and torque to be allocated to the actuators.
        """

        self.allocation_function(control_output, self.actuators)


class BasicRCSAllocator(ControlAllocatorBase):
    """
    Allocate commanded torque among RCS actuators.

    The allocator determines the component of each actuator direction along
    the requested torque direction and commands actuators that can
    contribute positively to the requested torque.
    """
    def __init__(self):
        super().__init__()

    def _check_initialization(self):
        pass

    def allocate(self, control_output: ControlOutput):
        """
        Allocate the commanded torque to the available RCS actuators.

        Parameters
        ----------
        control_output : ControlOutput
            Control output containing the commanded torque to be allocated.
        """
        
        torque_to_allocate = control_output.torque
        # Extract direction and magnitude of the torque to allocate
        torque_magnitude = np.linalg.norm(torque_to_allocate)
        torque_direction = torque_to_allocate / torque_magnitude if torque_magnitude != 0 else np.zeros(3)
        
        for actuator in self.actuators:
            # Calculate the dot product between the actuator's direction and the torque direction
            dot_product = np.dot(actuator.direction, torque_direction)
            # If the dot product is positive, the actuator can contribute to the torque
            if dot_product > 0:
                # Allocate a portion of the torque to this actuator based on its direction
                allocated_torque = dot_product * torque_magnitude
        
                # Set the actuator's command based on the allocated torque                
                actuator.set_command(allocated_torque)
            else:
                # If the actuator cannot contribute, set its command to zero
                actuator.set_command(0)

class MultiRCSAllocator(ControlAllocatorBase):
    """
    Allocate translational and rotational control commands among multiple RCS actuators.

    Translational force and rotational torque are allocated independently
    along the three Cartesian axes. For each requested component, the
    allocator selects an actuator capable of producing the requested
    direction, preferring the smallest actuator with sufficient capacity.
    """
    def __init__(self):
        super().__init__()

    def _check_initialization(self):
        pass

    def _allocate_vector(self, command, actuator_type, actuators):
        command = np.asarray(command, dtype=float)

        # Select actuators of the required type
        actuators = [
            actuator
            for actuator in actuators
            if actuator.name.startswith(actuator_type)
        ]

        # Allocate X, Y and Z independently
        for axis_idx in range(3):

            requested = command[axis_idx]

            if np.isclose(requested, 0.0):
                continue

            sign = np.sign(requested)
            magnitude = abs(requested)

            # --------------------------------------------------
            # Find actuators capable of producing this direction
            # --------------------------------------------------

            candidates = []

            for actuator in actuators:

                # Check whether actuator produces the requested
                # direction along this axis
                if actuator.direction[axis_idx] * sign <= 0:
                    continue

                # Determine actuator capacity
                if actuator_type == "translational_thruster":
                    capacity = actuator.nominal_thrust
                else:
                    capacity = actuator.override_torque_value

                candidates.append((capacity, actuator))

            if not candidates:
                continue

            # --------------------------------------------------
            # Select the smallest actuator capable of satisfying
            # the complete demand
            # --------------------------------------------------

            candidates.sort(key=lambda x: x[0])

            selected_actuator = None

            for capacity, actuator in candidates:

                if capacity >= magnitude:
                    selected_actuator = actuator
                    break

            # --------------------------------------------------
            # If no single actuator is large enough, use the
            # largest available actuator and let it saturate
            # --------------------------------------------------

            if selected_actuator is None:
                _, selected_actuator = candidates[-1]

            # -----------------------------------   ---------------
            # Send the COMPLETE requested command to the actuator
            # --------------------------------------------------

            selected_actuator.set_command(
                magnitude
            )    

    def allocate(self, control_output):
        """
        Allocate commanded force and torque to RCS actuators.

        All actuator commands are reset before translational force and
        rotational torque commands are allocated independently along the
        Cartesian axes.

        Parameters
        ----------
        control_output : ControlOutput
            Control output containing the commanded force and torque vectors.
        """
        force_to_allocate = np.asarray(control_output.force,dtype=float)
        torque_to_allocate = np.asarray(control_output.torque,dtype=float)
    
        # Reset all actuators
        for actuator in self.actuators:
            actuator.set_command(0.0)
    
        # Allocate translation
        self._allocate_vector(
            command=force_to_allocate,
            actuator_type="translational_thruster",
            actuators=self.actuators
        )
    
        # Allocate rotation
        self._allocate_vector(
            command=torque_to_allocate,
            actuator_type="rotational_thruster",
            actuators=self.actuators
        )
            


class ControlAllocatorPlaceholder(ControlAllocatorBase):
    """
    Placeholder control allocator for unsupported or unimplemented allocation logic.
    """
    def __init__(self):
        super().__init__()

    def _check_initialization(self):
        # No specific initialization checks for ControlAllocatorPlaceholder
        pass

    def allocate(self, control_output: ControlOutput):
        """
        Placeholder method for allocating a control output.

        Parameters
        ----------
        control_output : ControlOutput
            Control output to be allocated.

        Raises
        ------
        NotImplementedError
            Always raised because allocation is not implemented.
        """
        raise NotImplementedError("ControlAllocatorPlaceholder does not implement the allocate method.")

class ControllerPlaceholder(ControllerBase):
    """
    Placeholder controller for unsupported or unimplemented control logic.
    """
    def __init__(self):
        super().__init__()

    def _check_initialization(self):
        # No specific initialization checks for ControllerPlaceholder
        pass

    def compute_control(self, estimated_state: NavigationOutput, guidance_output: GuidanceOutput) -> ControlOutput:
        """
        Placeholder method for computing a control output.

        Parameters
        ----------
        estimated_state : NavigationOutput
            Estimated spacecraft state provided to the controller.
        guidance_output : GuidanceOutput
            Guidance information provided to the controller.

        Raises
        ------
        NotImplementedError
            Always raised because control computation is not implemented.
        """
        raise NotImplementedError("ControllerPlaceholder does not implement the compute_control method.")