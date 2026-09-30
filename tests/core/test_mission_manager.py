import pytest

from argos.general.dataclasses import MissionPhase
from argos.core.mission_manager import MissionManager


def make_phase(
    name,
    guidance=None,
    navigation=None,
    controller=None,
    dt_nav=None,
    dt_guid=None,
    dt_control=None,
    translational_model=None,
    rotational_model=None,
    dt_propagation=None,
):
    return MissionPhase(
        name=name,
        guidance=guidance,
        navigation=navigation,
        controller=controller,
        dt_nav=dt_nav,
        dt_guid=dt_guid,
        dt_control=dt_control,
        translational_model=translational_model,
        rotational_model=rotational_model,
        dt_propagation=dt_propagation,
    )


class TestMissionManagerInitialization:

    def test_initialization(self):
        phase = make_phase(
            "phase_1",
            translational_model=object(),
            dt_propagation=0.1,
        )

        manager = MissionManager(phase)

        assert manager.current_phase is phase
        assert manager.phases == {"phase_1": phase}
        assert manager.transitions == {}
        assert manager.verbose is False

    def test_verbose_can_be_enabled(self):
        phase = make_phase(
            "phase_1",
            translational_model=object(),
            dt_propagation=0.1,
        )

        manager = MissionManager(phase, verbose=True)

        assert manager.verbose is True


class TestMissionManagerPhaseValidation:

    def test_check_phase_rejects_non_mission_phase(self):
        initial_phase = make_phase(
            "phase_1",
            translational_model=object(),
            dt_propagation=0.1,
        )

        manager = MissionManager(initial_phase)

        with pytest.raises(
            TypeError,
            match="phase must be an instance of MissionPhase",
        ):
            manager._check_phase(object())

    def test_check_phase_rejects_duplicate_phase(self):
        initial_phase = make_phase(
            "phase_1",
            translational_model=object(),
            dt_propagation=0.1,
        )

        manager = MissionManager(initial_phase)

        duplicate_phase = make_phase(
            "phase_1",
            translational_model=object(),
            dt_propagation=0.1,
        )

        with pytest.raises(
            ValueError,
            match="Phase 'phase_1' already exists",
        ):
            manager._check_phase(duplicate_phase)

    def test_guidance_requires_dt_guid(self):
        initial_phase = make_phase(
            "phase_1",
            translational_model=object(),
            dt_propagation=0.1,
        )

        manager = MissionManager(initial_phase)

        phase = make_phase(
            "phase_2",
            guidance=object(),
            translational_model=object(),
            dt_propagation=0.1,
        )

        with pytest.raises(
            ValueError,
            match="dt_guid must be provided",
        ):
            manager._check_phase(phase)

    def test_navigation_requires_dt_nav(self):
        initial_phase = make_phase(
            "phase_1",
            translational_model=object(),
            dt_propagation=0.1,
        )

        manager = MissionManager(initial_phase)

        phase = make_phase(
            "phase_2",
            navigation=object(),
            translational_model=object(),
            dt_propagation=0.1,
        )

        with pytest.raises(
            ValueError,
            match="dt_nav must be provided",
        ):
            manager._check_phase(phase)

    def test_controller_requires_dt_control(self):
        initial_phase = make_phase(
            "phase_1",
            translational_model=object(),
            dt_propagation=0.1,
        )

        manager = MissionManager(initial_phase)

        phase = make_phase(
            "phase_2",
            controller=object(),
            translational_model=object(),
            dt_propagation=0.1,
        )

        with pytest.raises(
            ValueError,
            match="dt_control must be provided",
        ):
            manager._check_phase(phase)

    def test_dt_nav_requires_navigation(self):
        initial_phase = make_phase(
            "phase_1",
            translational_model=object(),
            dt_propagation=0.1,
        )

        manager = MissionManager(initial_phase)

        phase = make_phase(
            "phase_2",
            dt_nav=0.1,
            translational_model=object(),
            dt_propagation=0.1,
        )

        with pytest.raises(
            ValueError,
            match="Navigation law must be specified",
        ):
            manager._check_phase(phase)

    def test_dt_guid_requires_guidance(self):
        initial_phase = make_phase(
            "phase_1",
            translational_model=object(),
            dt_propagation=0.1,
        )

        manager = MissionManager(initial_phase)

        phase = make_phase(
            "phase_2",
            dt_guid=0.1,
            translational_model=object(),
            dt_propagation=0.1,
        )

        with pytest.raises(
            ValueError,
            match="Guidance law must be specified",
        ):
            manager._check_phase(phase)

    def test_dt_control_requires_controller(self):
        initial_phase = make_phase(
            "phase_1",
            translational_model=object(),
            dt_propagation=0.1,
        )

        manager = MissionManager(initial_phase)

        phase = make_phase(
            "phase_2",
            dt_control=0.1,
            translational_model=object(),
            dt_propagation=0.1,
        )

        with pytest.raises(
            ValueError,
            match="Control law must be specified",
        ):
            manager._check_phase(phase)

    def test_phase_requires_at_least_one_propagation_model(self):
        initial_phase = make_phase(
            "phase_1",
            translational_model=object(),
            dt_propagation=0.1,
        )

        manager = MissionManager(initial_phase)

        phase = make_phase("phase_2")

        with pytest.raises(
            ValueError,
            match="At least one of translational_model or rotational_model",
        ):
            manager._check_phase(phase)

    def test_propagation_model_requires_dt_propagation(self):
        initial_phase = make_phase(
            "phase_1",
            translational_model=object(),
            dt_propagation=0.1,
        )

        manager = MissionManager(initial_phase)

        phase = make_phase(
            "phase_2",
            translational_model=object(),
        )

        with pytest.raises(
            ValueError,
            match="dt_propagation must be provided",
        ):
            manager._check_phase(phase)

    def test_valid_phase_with_translational_model(self):
        initial_phase = make_phase(
            "phase_1",
            translational_model=object(),
            dt_propagation=0.1,
        )

        manager = MissionManager(initial_phase)

        phase = make_phase(
            "phase_2",
            translational_model=object(),
            dt_propagation=0.1,
        )

        manager._check_phase(phase)

    def test_valid_phase_with_rotational_model(self):
        initial_phase = make_phase(
            "phase_1",
            translational_model=object(),
            dt_propagation=0.1,
        )

        manager = MissionManager(initial_phase)

        phase = make_phase(
            "phase_2",
            rotational_model=object(),
            dt_propagation=0.1,
        )

        manager._check_phase(phase)


class TestMissionManagerPhases:

    def test_add_single_phase(self):
        initial_phase = make_phase(
            "phase_1",
            translational_model=object(),
            dt_propagation=0.1,
        )
        new_phase = make_phase(
            "phase_2",
            translational_model=object(),
            dt_propagation=0.1,
        )

        manager = MissionManager(initial_phase)

        manager.add_phases(new_phase)

        assert manager.phases["phase_2"] is new_phase

    def test_add_multiple_phases(self):
        initial_phase = make_phase(
            "phase_1",
            translational_model=object(),
            dt_propagation=0.1,
        )
        phase_2 = make_phase(
            "phase_2",
            translational_model=object(),
            dt_propagation=0.1,
        )
        phase_3 = make_phase(
            "phase_3",
            rotational_model=object(),
            dt_propagation=0.1,
        )

        manager = MissionManager(initial_phase)

        manager.add_phases([phase_2, phase_3])

        assert manager.phases["phase_2"] is phase_2
        assert manager.phases["phase_3"] is phase_3

    def test_add_phase_rejects_invalid_phase(self):
        initial_phase = make_phase(
            "phase_1",
            translational_model=object(),
            dt_propagation=0.1,
        )

        manager = MissionManager(initial_phase)

        with pytest.raises(TypeError):
            manager.add_phases(object())

    def test_add_duplicate_phase_raises(self):
        initial_phase = make_phase(
            "phase_1",
            translational_model=object(),
            dt_propagation=0.1,
        )

        manager = MissionManager(initial_phase)

        duplicate_phase = make_phase(
            "phase_1",
            translational_model=object(),
            dt_propagation=0.1,
        )

        with pytest.raises(
            ValueError,
            match="already exists",
        ):
            manager.add_phases(duplicate_phase)

    def test_add_multiple_phases_stops_on_invalid_phase(self):
        initial_phase = make_phase(
            "phase_1",
            translational_model=object(),
            dt_propagation=0.1,
        )
        valid_phase = make_phase(
            "phase_2",
            translational_model=object(),
            dt_propagation=0.1,
        )

        manager = MissionManager(initial_phase)

        with pytest.raises(TypeError):
            manager.add_phases([valid_phase, object()])

        assert "phase_2" in manager.phases


class TestMissionManagerTransitions:

    def test_add_transition(self):
        initial_phase = make_phase(
            "phase_1",
            translational_model=object(),
            dt_propagation=0.1,
        )
        target_phase = make_phase(
            "phase_2",
            translational_model=object(),
            dt_propagation=0.1,
        )

        manager = MissionManager(initial_phase)
        manager.add_phases(target_phase)

        condition = lambda data: True

        manager.add_transition(
            initial_phase,
            target_phase,
            condition,
        )

        transition = manager.transitions["phase_1"]["phase_1_to_phase_2"]

        assert transition["target_phase"] is target_phase
        assert transition["condition"] is condition

    def test_transition_name_is_generated_when_not_provided(self):
        initial_phase = make_phase(
            "phase_1",
            translational_model=object(),
            dt_propagation=0.1,
        )
        target_phase = make_phase(
            "phase_2",
            translational_model=object(),
            dt_propagation=0.1,
        )

        manager = MissionManager(initial_phase)
        manager.add_phases(target_phase)

        manager.add_transition(
            initial_phase,
            target_phase,
            lambda data: True,
        )

        assert "phase_1_to_phase_2" in manager.transitions["phase_1"]

    def test_custom_transition_name_is_used(self):
        initial_phase = make_phase(
            "phase_1",
            translational_model=object(),
            dt_propagation=0.1,
        )
        target_phase = make_phase(
            "phase_2",
            translational_model=object(),
            dt_propagation=0.1,
        )

        manager = MissionManager(initial_phase)
        manager.add_phases(target_phase)

        manager.add_transition(
            initial_phase,
            target_phase,
            lambda data: True,
            transition_name="custom_transition",
        )

        assert "custom_transition" in manager.transitions["phase_1"]

    def test_transition_rejects_unknown_source_phase(self):
        initial_phase = make_phase(
            "phase_1",
            translational_model=object(),
            dt_propagation=0.1,
        )
        unknown_phase = make_phase(
            "unknown",
            translational_model=object(),
            dt_propagation=0.1,
        )
        target_phase = make_phase(
            "phase_2",
            translational_model=object(),
            dt_propagation=0.1,
        )

        manager = MissionManager(initial_phase)
        manager.add_phases(target_phase)

        with pytest.raises(
            ValueError,
            match="Initial phase 'unknown' does not exist",
        ):
            manager.add_transition(
                unknown_phase,
                target_phase,
                lambda data: True,
            )

    def test_transition_rejects_unknown_target_phase(self):
        initial_phase = make_phase(
            "phase_1",
            translational_model=object(),
            dt_propagation=0.1,
        )
        unknown_phase = make_phase(
            "unknown",
            translational_model=object(),
            dt_propagation=0.1,
        )

        manager = MissionManager(initial_phase)

        with pytest.raises(
            ValueError,
            match="Target phase 'unknown' does not exist",
        ):
            manager.add_transition(
                initial_phase,
                unknown_phase,
                lambda data: True,
            )

    def test_transition_rejects_non_callable_condition(self):
        initial_phase = make_phase(
            "phase_1",
            translational_model=object(),
            dt_propagation=0.1,
        )
        target_phase = make_phase(
            "phase_2",
            translational_model=object(),
            dt_propagation=0.1,
        )

        manager = MissionManager(initial_phase)
        manager.add_phases(target_phase)

        with pytest.raises(
            TypeError,
            match="condition must be callable",
        ):
            manager.add_transition(
                initial_phase,
                target_phase,
                condition=True,
            )

    def test_duplicate_transition_name_raises(self):
        initial_phase = make_phase(
            "phase_1",
            translational_model=object(),
            dt_propagation=0.1,
        )
        target_phase = make_phase(
            "phase_2",
            translational_model=object(),
            dt_propagation=0.1,
        )

        manager = MissionManager(initial_phase)
        manager.add_phases(target_phase)

        manager.add_transition(
            initial_phase,
            target_phase,
            lambda data: True,
            transition_name="transition",
        )

        with pytest.raises(
            ValueError,
            match="Transition 'transition' already exists",
        ):
            manager.add_transition(
                initial_phase,
                target_phase,
                lambda data: True,
                transition_name="transition",
            )


class TestMissionManagerUpdate:

    def test_update_returns_false_when_no_transitions_exist(self):
        phase = make_phase(
            "phase_1",
            translational_model=object(),
            dt_propagation=0.1,
        )

        manager = MissionManager(phase)

        changed, info = manager.update(object())

        assert changed is False
        assert info is None
        assert manager.current_phase is phase

    def test_update_does_not_transition_when_condition_is_false(self):
        initial_phase = make_phase(
            "phase_1",
            translational_model=object(),
            dt_propagation=0.1,
        )
        target_phase = make_phase(
            "phase_2",
            translational_model=object(),
            dt_propagation=0.1,
        )

        manager = MissionManager(initial_phase)
        manager.add_phases(target_phase)

        manager.add_transition(
            initial_phase,
            target_phase,
            lambda data: False,
        )

        changed, info = manager.update(object())

        assert changed is False
        assert info is None
        assert manager.current_phase is initial_phase

    def test_update_transitions_when_condition_is_true(self):
        initial_phase = make_phase(
            "phase_1",
            translational_model=object(),
            dt_propagation=0.1,
        )
        target_phase = make_phase(
            "phase_2",
            translational_model=object(),
            dt_propagation=0.1,
        )

        manager = MissionManager(initial_phase)
        manager.add_phases(target_phase)

        manager.add_transition(
            initial_phase,
            target_phase,
            lambda data: True,
        )

        changed, info = manager.update(object())

        assert changed is True
        assert manager.current_phase is target_phase

        assert info == {
            "from_phase": "phase_1",
            "to_phase": "phase_2",
            "via": "phase_1_to_phase_2",
        }

    def test_update_passes_simulation_data_to_condition(self):
        initial_phase = make_phase(
            "phase_1",
            translational_model=object(),
            dt_propagation=0.1,
        )
        target_phase = make_phase(
            "phase_2",
            translational_model=object(),
            dt_propagation=0.1,
        )

        manager = MissionManager(initial_phase)
        manager.add_phases(target_phase)

        received_data = []

        def condition(data):
            received_data.append(data)
            return False

        manager.add_transition(
            initial_phase,
            target_phase,
            condition,
        )

        simulation_data = object()

        manager.update(simulation_data)

        assert received_data == [simulation_data]

    def test_update_uses_first_satisfied_transition(self):
        initial_phase = make_phase(
            "phase_1",
            translational_model=object(),
            dt_propagation=0.1,
        )
        phase_2 = make_phase(
            "phase_2",
            translational_model=object(),
            dt_propagation=0.1,
        )
        phase_3 = make_phase(
            "phase_3",
            translational_model=object(),
            dt_propagation=0.1,
        )

        manager = MissionManager(initial_phase)
        manager.add_phases([phase_2, phase_3])

        manager.add_transition(
            initial_phase,
            phase_2,
            lambda data: True,
            transition_name="first",
        )

        manager.add_transition(
            initial_phase,
            phase_3,
            lambda data: True,
            transition_name="second",
        )

        changed, info = manager.update(object())

        assert changed is True
        assert manager.current_phase is phase_2
        assert info["via"] == "first"

    def test_update_verbose_prints_transition(self, capsys):
        initial_phase = make_phase(
            "phase_1",
            translational_model=object(),
            dt_propagation=0.1,
        )
        target_phase = make_phase(
            "phase_2",
            translational_model=object(),
            dt_propagation=0.1,
        )

        manager = MissionManager(initial_phase, verbose=True)
        manager.add_phases(target_phase)

        manager.add_transition(
            initial_phase,
            target_phase,
            lambda data: True,
        )

        manager.update(object())

        captured = capsys.readouterr()

        assert "Transitioned from 'phase_1' to 'phase_2'" in captured.out
        assert "via 'phase_1_to_phase_2'" in captured.out

    def test_update_verbose_false_does_not_print_transition(self, capsys):
        initial_phase = make_phase(
            "phase_1",
            translational_model=object(),
            dt_propagation=0.1,
        )
        target_phase = make_phase(
            "phase_2",
            translational_model=object(),
            dt_propagation=0.1,
        )

        manager = MissionManager(initial_phase, verbose=False)
        manager.add_phases(target_phase)

        manager.add_transition(
            initial_phase,
            target_phase,
            lambda data: True,
        )

        manager.update(object())

        captured = capsys.readouterr()

        assert captured.out == ""

    def test_update_after_transition_evaluates_target_phase_transitions(self):
        initial_phase = make_phase(
            "phase_1",
            translational_model=object(),
            dt_propagation=0.1,
        )
        phase_2 = make_phase(
            "phase_2",
            translational_model=object(),
            dt_propagation=0.1,
        )
        phase_3 = make_phase(
            "phase_3",
            translational_model=object(),
            dt_propagation=0.1,
        )

        manager = MissionManager(initial_phase)
        manager.add_phases([phase_2, phase_3])

        manager.add_transition(
            initial_phase,
            phase_2,
            lambda data: True,
        )

        manager.add_transition(
            phase_2,
            phase_3,
            lambda data: True,
        )

        changed, _ = manager.update(object())

        assert changed is True
        assert manager.current_phase is phase_2

        changed, info = manager.update(object())

        assert changed is True
        assert manager.current_phase is phase_3
        assert info["from_phase"] == "phase_2"
        assert info["to_phase"] == "phase_3"