import pytest

from argos.faults.fault_manager import (
    FaultEvent,
    FaultInjector,
    FaultManager,
    FaultMode,
)


class DummyFault(FaultMode):
    def __init__(self):
        super().__init__()
        self.activate_calls = 0
        self.deactivate_calls = 0

    def on_activate(self):
        self.activate_calls += 1

    def on_deactivate(self):
        self.deactivate_calls += 1

    def apply(self, value, context=None):
        return value


class MockComponent:
    def __init__(self, name):
        self.name = name
        self.fault_injector = FaultInjector()


class MockSpacecraft:
    def __init__(self):
        self.sensors = []
        self.actuators = []


def make_manager():
    spacecraft = MockSpacecraft()
    manager = FaultManager(spacecraft)

    return manager, spacecraft


class TestFaultEvent:

    def test_default_active_state_is_false(self):
        event = FaultEvent(
            event_name="test_event",
            component_name="sensor_1",
            fault_mode=DummyFault(),
            activation_condition=lambda data: True,
        )

        assert event.active is False

    def test_default_deactivation_condition_is_none(self):
        event = FaultEvent(
            event_name="test_event",
            component_name="sensor_1",
            fault_mode=DummyFault(),
            activation_condition=lambda data: True,
        )

        assert event.deactivation_condition is None


class TestFaultManager:

    def test_initialization(self):
        spacecraft = MockSpacecraft()

        manager = FaultManager(spacecraft)

        assert manager.spacecraft is spacecraft
        assert manager.fault_events == []

    def test_add_single_fault_event(self):
        manager, spacecraft = make_manager()

        component = MockComponent("sensor_1")
        spacecraft.sensors.append(component)

        event = FaultEvent(
            event_name="event_1",
            component_name="sensor_1",
            fault_mode=DummyFault(),
            activation_condition=lambda data: True,
        )

        manager.add_fault_event(event)

        assert manager.fault_events == [event]

    def test_add_multiple_fault_events(self):
        manager, spacecraft = make_manager()

        spacecraft.sensors.append(MockComponent("sensor_1"))
        spacecraft.sensors.append(MockComponent("sensor_2"))

        event_1 = FaultEvent(
            event_name="event_1",
            component_name="sensor_1",
            fault_mode=DummyFault(),
            activation_condition=lambda data: True,
        )

        event_2 = FaultEvent(
            event_name="event_2",
            component_name="sensor_2",
            fault_mode=DummyFault(),
            activation_condition=lambda data: True,
        )

        manager.add_fault_event([event_1, event_2])

        assert manager.fault_events == [
            event_1,
            event_2,
        ]

    def test_add_fault_event_rejects_non_callable_activation_condition(self):
        manager, spacecraft = make_manager()

        spacecraft.sensors.append(MockComponent("sensor_1"))

        event = FaultEvent(
            event_name="event_1",
            component_name="sensor_1",
            fault_mode=DummyFault(),
            activation_condition=True,
        )

        with pytest.raises(
            TypeError,
            match="Activation condition",
        ):
            manager.add_fault_event(event)

    def test_add_fault_event_rejects_non_callable_deactivation_condition(self):
        manager, spacecraft = make_manager()

        spacecraft.sensors.append(MockComponent("sensor_1"))

        event = FaultEvent(
            event_name="event_1",
            component_name="sensor_1",
            fault_mode=DummyFault(),
            activation_condition=lambda data: True,
            deactivation_condition=True,
        )

        with pytest.raises(
            TypeError,
            match="Deactivation condition",
        ):
            manager.add_fault_event(event)

    def test_add_fault_event_rejects_invalid_fault_mode_for_sensor(self):
        manager, spacecraft = make_manager()

        spacecraft.sensors.append(MockComponent("sensor_1"))

        event = FaultEvent(
            event_name="event_1",
            component_name="sensor_1",
            fault_mode=object(),
            activation_condition=lambda data: True,
        )

        with pytest.raises(
            TypeError,
            match="Fault mode for sensor",
        ):
            manager.add_fault_event(event)

    def test_add_fault_event_rejects_invalid_fault_mode_for_actuator(self):
        manager, spacecraft = make_manager()

        spacecraft.actuators.append(MockComponent("actuator_1"))

        event = FaultEvent(
            event_name="event_1",
            component_name="actuator_1",
            fault_mode=object(),
            activation_condition=lambda data: True,
        )

        with pytest.raises(
            TypeError,
            match="Fault mode for actuator",
        ):
            manager.add_fault_event(event)

    def test_add_same_event_object_twice_raises(self):
        manager, spacecraft = make_manager()

        spacecraft.sensors.append(MockComponent("sensor_1"))

        event = FaultEvent(
            event_name="event_1",
            component_name="sensor_1",
            fault_mode=DummyFault(),
            activation_condition=lambda data: True,
        )

        manager.add_fault_event(event)

        with pytest.raises(
            ValueError,
            match="already added",
        ):
            manager.add_fault_event(event)

    def test_add_event_with_duplicate_name_raises(self):
        manager, spacecraft = make_manager()

        spacecraft.sensors.append(MockComponent("sensor_1"))
        spacecraft.sensors.append(MockComponent("sensor_2"))

        event_1 = FaultEvent(
            event_name="same_name",
            component_name="sensor_1",
            fault_mode=DummyFault(),
            activation_condition=lambda data: True,
        )

        event_2 = FaultEvent(
            event_name="same_name",
            component_name="sensor_2",
            fault_mode=DummyFault(),
            activation_condition=lambda data: True,
        )

        manager.add_fault_event(event_1)

        with pytest.raises(
            ValueError,
            match="already exists",
        ):
            manager.add_fault_event(event_2)

    def test_remove_fault_event(self):
        manager, spacecraft = make_manager()

        spacecraft.sensors.append(MockComponent("sensor_1"))

        event = FaultEvent(
            event_name="event_1",
            component_name="sensor_1",
            fault_mode=DummyFault(),
            activation_condition=lambda data: True,
        )

        manager.add_fault_event(event)
        manager.remove_fault_event(event)

        assert manager.fault_events == []

    def test_remove_unregistered_fault_event_raises(self):
        manager, spacecraft = make_manager()

        event = FaultEvent(
            event_name="event_1",
            component_name="sensor_1",
            fault_mode=DummyFault(),
            activation_condition=lambda data: True,
        )

        with pytest.raises(
            ValueError,
            match="not found",
        ):
            manager.remove_fault_event(event)

    def test_get_sensor_component(self):
        manager, spacecraft = make_manager()

        sensor = MockComponent("sensor_1")
        spacecraft.sensors.append(sensor)

        result = manager._get_component("sensor_1")

        assert result is sensor

    def test_get_actuator_component(self):
        manager, spacecraft = make_manager()

        actuator = MockComponent("actuator_1")
        spacecraft.actuators.append(actuator)

        result = manager._get_component("actuator_1")

        assert result is actuator

    def test_get_component_raises_when_not_found(self):
        manager, spacecraft = make_manager()

        with pytest.raises(
            ValueError,
            match="Component 'missing' not found",
        ):
            manager._get_component("missing")

    def test_update_activates_fault(self):
        manager, spacecraft = make_manager()

        component = MockComponent("sensor_1")
        spacecraft.sensors.append(component)

        fault = DummyFault()

        event = FaultEvent(
            event_name="event_1",
            component_name="sensor_1",
            fault_mode=fault,
            activation_condition=lambda data: True,
        )

        manager.add_fault_event(event)

        simulation_data = object()

        manager.update(simulation_data)

        assert event.active is True
        assert component.fault_injector.faults == [fault]

    def test_update_does_not_activate_fault_when_condition_is_false(self):
        manager, spacecraft = make_manager()

        component = MockComponent("sensor_1")
        spacecraft.sensors.append(component)

        fault = DummyFault()

        event = FaultEvent(
            event_name="event_1",
            component_name="sensor_1",
            fault_mode=fault,
            activation_condition=lambda data: False,
        )

        manager.add_fault_event(event)

        manager.update(object())

        assert event.active is False
        assert component.fault_injector.faults == []

    def test_update_does_not_activate_already_active_fault(self):
        manager, spacecraft = make_manager()

        component = MockComponent("sensor_1")
        spacecraft.sensors.append(component)

        fault = DummyFault()

        event = FaultEvent(
            event_name="event_1",
            component_name="sensor_1",
            fault_mode=fault,
            activation_condition=lambda data: True,
        )

        manager.add_fault_event(event)

        manager.update(object())
        manager.update(object())

        assert event.active is True
        assert component.fault_injector.faults == [fault]

    def test_update_deactivates_fault(self):
        manager, spacecraft = make_manager()

        component = MockComponent("sensor_1")
        spacecraft.sensors.append(component)

        fault = DummyFault()

        event = FaultEvent(
            event_name="event_1",
            component_name="sensor_1",
            fault_mode=fault,
            activation_condition=lambda data: True,
            deactivation_condition=lambda data: True,
        )

        manager.add_fault_event(event)

        manager.update(object())

        assert event.active is True
        assert component.fault_injector.faults == [fault]

        manager.update(object())

        assert event.active is False
        assert component.fault_injector.faults == []

    def test_update_does_not_deactivate_when_condition_is_false(self):
        manager, spacecraft = make_manager()

        component = MockComponent("sensor_1")
        spacecraft.sensors.append(component)

        fault = DummyFault()

        event = FaultEvent(
            event_name="event_1",
            component_name="sensor_1",
            fault_mode=fault,
            activation_condition=lambda data: True,
            deactivation_condition=lambda data: False,
        )

        manager.add_fault_event(event)

        manager.update(object())
        manager.update(object())

        assert event.active is True
        assert component.fault_injector.faults == [fault]

    def test_update_passes_simulation_data_to_activation_condition(self):
        manager, spacecraft = make_manager()

        component = MockComponent("sensor_1")
        spacecraft.sensors.append(component)

        received_data = []

        def activation_condition(data):
            received_data.append(data)
            return False

        event = FaultEvent(
            event_name="event_1",
            component_name="sensor_1",
            fault_mode=DummyFault(),
            activation_condition=activation_condition,
        )

        manager.add_fault_event(event)

        simulation_data = object()

        manager.update(simulation_data)

        assert received_data == [simulation_data]

    def test_update_passes_simulation_data_to_deactivation_condition(self):
        manager, spacecraft = make_manager()

        component = MockComponent("sensor_1")
        spacecraft.sensors.append(component)

        received_data = []

        def deactivation_condition(data):
            received_data.append(data)
            return True

        event = FaultEvent(
            event_name="event_1",
            component_name="sensor_1",
            fault_mode=DummyFault(),
            activation_condition=lambda data: True,
            deactivation_condition=deactivation_condition,
        )

        manager.add_fault_event(event)

        simulation_data = object()

        manager.update(simulation_data)
        manager.update(simulation_data)

        assert received_data == [
            simulation_data,
        ]

    def test_show_faults_info_when_empty(self, capsys):
        manager, _ = make_manager()

        manager.show_faults_info()

        captured = capsys.readouterr()

        assert captured.out == "No fault events added.\n"

    def test_show_faults_info_displays_event(self, capsys):
        manager, spacecraft = make_manager()

        spacecraft.sensors.append(MockComponent("sensor_1"))

        event = FaultEvent(
            event_name="event_1",
            component_name="sensor_1",
            fault_mode=DummyFault(),
            activation_condition=lambda data: True,
        )

        manager.add_fault_event(event)

        manager.show_faults_info()

        captured = capsys.readouterr()

        assert "Fault Events:" in captured.out
        assert "Event Name: event_1" in captured.out
        assert "Component: sensor_1" in captured.out
        assert "Fault Mode: DummyFault" in captured.out
        assert "Status: Inactive" in captured.out

    def test_show_faults_info_displays_active_status(self, capsys):
        manager, spacecraft = make_manager()

        spacecraft.sensors.append(MockComponent("sensor_1"))

        event = FaultEvent(
            event_name="event_1",
            component_name="sensor_1",
            fault_mode=DummyFault(),
            activation_condition=lambda data: True,
        )

        manager.add_fault_event(event)
        manager.update(object())

        manager.show_faults_info()

        captured = capsys.readouterr()

        assert "Status: Active" in captured.out