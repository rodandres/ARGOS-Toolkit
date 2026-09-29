from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


class FaultMode(ABC):
    """
    Abstract base class for spacecraft fault modes.

    A fault mode defines how a fault is activated, deactivated, and applied
    to a component value.
    """
    def __init__(self):
        self.active = False

    def activate(self):
        """
        Activate the fault mode.

        Sets the fault mode to active and calls the subclass-specific activation
        hook.
        """
        self.active = True
        self.on_activate()

    def deactivate(self):
        """
        Deactivate the fault mode.

        Calls the subclass-specific deactivation hook and sets the fault mode
        to inactive.
        """
        self.on_deactivate()
        self.active = False

    @abstractmethod
    def on_activate(self):
        """
        Perform subclass-specific actions when the fault is activated.
        """
        pass

    @abstractmethod
    def on_deactivate(self):
        """
        Perform subclass-specific actions when the fault is deactivated.
        """
        pass

    @abstractmethod
    def apply(self, value, context=None):
        """
        Apply the fault behavior to a component value.

        Parameters
        ----------
        value
            Value received from the affected component.
        context, optional
            Additional context available to the fault mode.

        Returns
        -------
        object
            Fault-modified value.
        """
        return value

class FaultInjector:
    """
    Manage and apply multiple fault modes to a component value.

    Faults are applied sequentially in the order in which they were added.
    """
    def __init__(self):
        self.faults = []

    def add_fault(self, fault: FaultMode):
        """
        Add a fault mode to the injector.

        Parameters
        ----------
        fault : FaultMode
            Fault mode to add to the injector.
        """        
        self.faults.append(fault)

    def remove_fault(self, fault: FaultMode):
        """
        Remove a fault mode from the injector.

        Parameters
        ----------
        fault : FaultMode
            Fault mode to remove.
        """
        self.faults.remove(fault)

    def apply(self, value, context=None):
        """
        Apply all registered fault modes sequentially.

        Parameters
        ----------
        value
            Original component value to which the faults are applied.
        context, optional
            Additional context passed to each fault mode.

        Returns
        -------
        object
            Value after all registered fault modes have been applied.
        """
        for fault in self.faults:
            value = fault.apply(value, context)

        return value

@dataclass 
class FaultEvent:
    """
    Define the conditions and fault mode associated with a fault event.

    Attributes
    ----------
    event_name : str
        Unique name identifying the fault event.
    component_name : str
        Name of the spacecraft component affected by the fault.
    fault_mode : FaultMode
        Fault mode applied when the event is activated.
    activation_condition : callable
        Function that receives the simulation data and returns whether the
        fault event should be activated.
    deactivation_condition : callable, optional
        Function that receives the simulation data and returns whether the
        fault event should be deactivated.
    active : bool
        Current activation state of the fault event.
    """
    event_name: str
    component_name: str
    fault_mode: FaultMode
    activation_condition: callable
    deactivation_condition: Optional[callable] = None

    active: bool = False
    
class FaultManager:
    """
    Manage fault events associated with a spacecraft.

    The fault manager evaluates fault activation and deactivation
    conditions during the simulation and applies the corresponding fault
    modes to spacecraft sensors and actuators.
    """
    def __init__(self, spacecraft):
        self.fault_events = []
        self.spacecraft = spacecraft

    def _check_event(self, fault_event: FaultEvent):
        if not callable(fault_event.activation_condition):
            raise TypeError(
                f"Activation condition for fault event on component '{fault_event.component_name}' must be callable."
            )
        if fault_event.deactivation_condition is not None and not callable(fault_event.deactivation_condition):
            raise TypeError(
                f"Deactivation condition for fault event on component '{fault_event.component_name}' must be callable."
            )

        for sensor in self.spacecraft.sensors:
            if sensor.name == fault_event.component_name:
                if not isinstance(fault_event.fault_mode, FaultMode):
                    raise TypeError(
                        f"Fault mode for sensor '{sensor.name}' must be an instance of FaultMode."
                    )
                break

        for actuator in self.spacecraft.actuators:
            if actuator.name == fault_event.component_name:
                if not isinstance(fault_event.fault_mode, FaultMode):
                    raise TypeError(
                        f"Fault mode for actuator '{actuator.name}' must be an instance of FaultMode."
                    )
                break

        if fault_event in self.fault_events:
            raise ValueError(
                f"Fault event '{fault_event.event_name}' already added."
            )

        for event in self.fault_events:
            if event.event_name == fault_event.event_name:
                raise ValueError(
                    f"Fault event name '{fault_event.event_name}' already exists."
                )

    def _get_component(self, component_name):
        for sensor in self.spacecraft.sensors:
            if sensor.name == component_name:
                return sensor

        for actuator in self.spacecraft.actuators:
            if actuator.name == component_name:
                return actuator

        raise ValueError(f"Component '{component_name}' not found.")

    def add_fault_event(self, fault_event: FaultEvent | list[FaultEvent]):
        """
        Add one or more fault events to the fault manager.

        Parameters
        ----------
        fault_event : FaultEvent or list of FaultEvent
            Fault event or events to add.

        Raises
        ------
        TypeError
            If an activation or deactivation condition is not callable, or if
            the fault mode is incompatible with the target component.
        ValueError
            If an event or event name has already been added.
        """
        if isinstance(fault_event, list):
            for event in fault_event:
                self._check_event(event)
                self.fault_events.append(event)
        else:
            self._check_event(fault_event)
            self.fault_events.append(fault_event)

    def remove_fault_event(self, fault_event: FaultEvent):
        """
        Remove a fault event from the fault manager.

        Parameters
        ----------
        fault_event : FaultEvent
            Fault event to remove.

        Raises
        ------
        ValueError
            If the fault event is not registered with the fault manager.
        """
        if fault_event not in self.fault_events:
            raise ValueError(
                f"Fault event '{fault_event.event_name}' not found."
            )
        
        self.fault_events.remove(fault_event)

    def update(self, simulation_data):
        """
        Evaluate and update all registered fault events.

        Activation and deactivation conditions are evaluated using the current
        simulation data. When a condition is satisfied, the corresponding fault
        mode is added to or removed from the affected component's fault
        injector.

        Parameters
        ----------
        simulation_data : SimulationData
            Current simulation data used to evaluate fault event conditions.
        """
        for fault_event in self.fault_events:

            if (
                not fault_event.active
                and fault_event.activation_condition(simulation_data)
            ):
                component = self._get_component(fault_event.component_name)

                component.fault_injector.add_fault(fault_event.fault_mode)

                fault_event.active = True

            elif (
                fault_event.active
                and fault_event.deactivation_condition is not None
                and fault_event.deactivation_condition(simulation_data)):

                component = self._get_component(fault_event.component_name)

                component.fault_injector.remove_fault(fault_event.fault_mode)

                fault_event.active = False

    def show_faults_info(self):
        """
        Print the current status of all registered fault events.

        If no fault events are registered, an informational message is printed.
        """
        if not self.fault_events:
            print("No fault events added.")
            return

        print("="*50)
        print("Fault Events:")
        for event in self.fault_events:            
            status = "Active" if event.active else "Inactive"
            print("="*15)
            print(f"  - Event Name: {event.event_name}")
            print(f"    Component: {event.component_name}")
            print(f"    Fault Mode: {type(event.fault_mode).__name__}")
            print(f"    Status: {status}")