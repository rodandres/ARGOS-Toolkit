from argos.faults.fault_manager import FaultMode, FaultInjector


class AddFault(FaultMode):
    def __init__(self, amount):
        super().__init__()
        self.amount = amount

    def on_activate(self):
        pass

    def on_deactivate(self):
        pass

    def apply(self, value, context=None):
        return value + self.amount


class ContextFault(FaultMode):
    def __init__(self):
        super().__init__()
        self.received_context = None

    def on_activate(self):
        pass

    def on_deactivate(self):
        pass

    def apply(self, value, context=None):
        self.received_context = context
        return value


class TestFaultMode:

    def test_initial_state_is_inactive(self):
        fault = AddFault(1)

        assert fault.active is False

    def test_activate_sets_active(self):
        fault = AddFault(1)

        fault.activate()

        assert fault.active is True

    def test_activate_calls_on_activate(self):
        class TrackingFault(FaultMode):
            def __init__(self):
                super().__init__()
                self.activation_called = False

            def on_activate(self):
                self.activation_called = True

            def on_deactivate(self):
                pass

            def apply(self, value, context=None):
                return value

        fault = TrackingFault()

        fault.activate()

        assert fault.activation_called is True

    def test_deactivate_sets_inactive(self):
        fault = AddFault(1)

        fault.activate()
        fault.deactivate()

        assert fault.active is False

    def test_deactivate_calls_on_deactivate(self):
        class TrackingFault(FaultMode):
            def __init__(self):
                super().__init__()
                self.deactivation_called = False

            def on_activate(self):
                pass

            def on_deactivate(self):
                self.deactivation_called = True

            def apply(self, value, context=None):
                return value

        fault = TrackingFault()

        fault.activate()
        fault.deactivate()

        assert fault.deactivation_called is True


class TestFaultInjector:

    def test_initialization_has_no_faults(self):
        injector = FaultInjector()

        assert injector.faults == []

    def test_add_fault(self):
        injector = FaultInjector()
        fault = AddFault(1)

        injector.add_fault(fault)

        assert injector.faults == [fault]

    def test_add_multiple_faults_preserves_order(self):
        injector = FaultInjector()

        first_fault = AddFault(1)
        second_fault = AddFault(2)

        injector.add_fault(first_fault)
        injector.add_fault(second_fault)

        assert injector.faults == [
            first_fault,
            second_fault,
        ]

    def test_remove_fault(self):
        injector = FaultInjector()
        fault = AddFault(1)

        injector.add_fault(fault)
        injector.remove_fault(fault)

        assert injector.faults == []

    def test_remove_fault_raises_if_not_registered(self):
        injector = FaultInjector()
        fault = AddFault(1)

        try:
            injector.remove_fault(fault)
        except ValueError:
            pass
        else:
            raise AssertionError("Expected ValueError")

    def test_apply_without_faults_returns_original_value(self):
        injector = FaultInjector()

        value = 10

        result = injector.apply(value)

        assert result == value

    def test_apply_single_fault(self):
        injector = FaultInjector()

        injector.add_fault(AddFault(5))

        result = injector.apply(10)

        assert result == 15

    def test_apply_faults_sequentially(self):
        injector = FaultInjector()

        injector.add_fault(AddFault(2))
        injector.add_fault(AddFault(3))

        result = injector.apply(10)

        assert result == 15

    def test_apply_preserves_fault_order(self):
        class MultiplyFault(FaultMode):
            def on_activate(self):
                pass

            def on_deactivate(self):
                pass

            def apply(self, value, context=None):
                return value * 2

        injector = FaultInjector()

        injector.add_fault(AddFault(3))
        injector.add_fault(MultiplyFault())

        result = injector.apply(10)

        assert result == 26

    def test_context_is_passed_to_fault(self):
        injector = FaultInjector()
        fault = ContextFault()

        injector.add_fault(fault)

        context = object()

        injector.apply(10, context)

        assert fault.received_context is context