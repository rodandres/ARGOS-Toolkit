import numpy as np
from py.modules.controllers.controller_base import ControllerBase, ControlAllocatorBase
from py.general.dataclasses import ControlOutput

class CustomController(ControllerBase):
    def __init__(self, control_function):
        self.control_function = control_function

        super().__init__()
        # Initialize any additional attributes if needed

    def _check_initialization(self):
        if not callable(self.control_function):
            raise ValueError("control_function must be a callable function.")

    def compute_control(self, estimated_state, reference):
        """
        Compute the control output based on the estimated state and reference.

        Args:
            estimated_state (EstimationOutput): The estimated state of the spacecraft.
            reference (GuidanceReference): The reference guidance data.
        Returns:
            ControlOutput: The computed control output.
        """

        output = self.control_function(estimated_state, reference)

        if not isinstance(output, ControlOutput):
            raise TypeError("The control_function must return a ControlOutput object.")        

        return output

class CustomControlAllocator(ControlAllocatorBase):
    def __init__(self, allocation_function):
        self.allocation_function = allocation_function

        super().__init__()
        # Initialize any additional attributes if needed

    def _check_initialization(self):
        if not callable(self.allocation_function):
            raise ValueError("allocation_function must be a callable function.")

    def allocate(self, control_output):
        """
        Allocate the control output to actuators.

        Args:
            control_output (ControlOutput): The computed control output.
        Returns:
            list: The allocated actuator commands.
        """

        self.allocation_function(control_output, self.actuators)



class BasicRCSAllocator(ControlAllocatorBase):
    def __init__(self):
        super().__init__()

    def _check_initialization(self):
        pass

    def allocate(self, control_output):
        """
        Allocate the control output to RCS actuators.

        Args:
            control_output (ControlOutput): The computed control output.
        Returns:
            list: The allocated RCS commands.
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
