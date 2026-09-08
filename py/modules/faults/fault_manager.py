from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


class FaultMode(ABC):
    def __init__(self):
        self.active = False

    def activate(self):
        self.active = True
        self.on_activate()

    def deactivate(self):
        self.on_deactivate()
        self.active = False

    @abstractmethod
    def on_activate(self):
        pass

    @abstractmethod
    def on_deactivate(self):
        pass

    @abstractmethod
    def apply(self, value, context=None):
        return value

class FaultInjector:
    def __init__(self):
        self.faults = []

    def add_fault(self, fault):
        self.faults.append(fault)

    def remove_fault(self, fault):
        self.faults.remove(fault)

    def apply(self, value, context=None):

        for fault in self.faults:
            value = fault.apply(value, context)

        return value

@dataclass 
class FaultEvent:
    event_name: str
    component_name: str
    fault_mode: FaultMode
    activation_condition: callable
    deactivation_condition: Optional[callable] = None

    active: bool = False
    
class FaultManager:
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
        
        if isinstance(fault_event, list):
            for event in fault_event:
                self._check_event(event)
                self.fault_events.append(event)
        else:
            self._check_event(fault_event)
            self.fault_events.append(fault_event)

    def remove_fault_event(self, fault_event: FaultEvent):
        if fault_event not in self.fault_events:
            raise ValueError(
                f"Fault event '{fault_event.event_name}' not found."
            )
        
        self.fault_events.remove(fault_event)

    def update(self, simulation_data):

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