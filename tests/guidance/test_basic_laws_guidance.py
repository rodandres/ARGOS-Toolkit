import numpy as np
import pytest

from argos.guidance.basic_laws import (
    ConstantReferenceGuidance,
    CustomGuidanceLaw,
    GuidancePlaceholder,
)
from argos.general.dataclasses import GuidanceOutput


# ============================================================
# ConstantReferenceGuidance
# ============================================================


class TestConstantReferenceGuidanceInitialization:

    def test_initialization_with_position_only(self):
        desired_pos = np.array([1.0, 2.0, 3.0])

        guidance = ConstantReferenceGuidance(
            desired_pos=desired_pos
        )

        np.testing.assert_array_equal(
            guidance.desired_pos,
            desired_pos,
        )

    def test_initialization_with_all_values(self):
        desired_pos = np.array([1.0, 2.0, 3.0])
        desired_vel = np.array([4.0, 5.0, 6.0])
        desired_accel = np.array([7.0, 8.0, 9.0])
        desired_quat = np.array([0.1, 0.2, 0.3, 0.4])
        desired_ang_vel = np.array([0.5, 0.6, 0.7])
        desired_ang_accel = np.array([0.8, 0.9, 1.0])

        guidance = ConstantReferenceGuidance(
            desired_pos=desired_pos,
            desired_vel=desired_vel,
            desired_accel=desired_accel,
            desired_quat=desired_quat,
            desired_ang_vel=desired_ang_vel,
            desired_ang_accel=desired_ang_accel,
        )

        np.testing.assert_array_equal(
            guidance.desired_pos,
            desired_pos,
        )
        np.testing.assert_array_equal(
            guidance.desired_vel,
            desired_vel,
        )
        np.testing.assert_array_equal(
            guidance.desired_accel,
            desired_accel,
        )
        np.testing.assert_array_equal(
            guidance.desired_quat,
            desired_quat,
        )
        np.testing.assert_array_equal(
            guidance.desired_ang_vel,
            desired_ang_vel,
        )
        np.testing.assert_array_equal(
            guidance.desired_ang_accel,
            desired_ang_accel,
        )

    def test_no_desired_state_is_rejected(self):
        with pytest.raises(ValueError):
            ConstantReferenceGuidance()

    @pytest.mark.parametrize(
        "parameter",
        [
            "desired_pos",
            "desired_vel",
            "desired_accel",
            "desired_ang_vel",
            "desired_ang_accel",
        ],
    )
    def test_vector_parameters_must_be_numpy_arrays(self, parameter):
        kwargs = {
            parameter: [1.0, 2.0, 3.0]
        }

        with pytest.raises(TypeError):
            ConstantReferenceGuidance(**kwargs)

    def test_quaternion_must_be_numpy_array(self):
        with pytest.raises(TypeError):
            ConstantReferenceGuidance(
                desired_quat=[1.0, 0.0, 0.0, 0.0]
            )

    @pytest.mark.parametrize(
        "parameter",
        [
            "desired_pos",
            "desired_vel",
            "desired_accel",
            "desired_ang_vel",
            "desired_ang_accel",
        ],
    )
    def test_vector_parameters_must_have_shape_3(self, parameter):
        kwargs = {
            parameter: np.zeros(2)
        }

        with pytest.raises(ValueError):
            ConstantReferenceGuidance(**kwargs)

    def test_quaternion_must_have_shape_4(self):
        with pytest.raises(ValueError):
            ConstantReferenceGuidance(
                desired_quat=np.zeros(3)
            )

    def test_missing_vector_components_default_to_zero(self):
        guidance = ConstantReferenceGuidance(
            desired_pos=np.array([1.0, 2.0, 3.0])
        )

        np.testing.assert_array_equal(
            guidance.desired_vel,
            np.zeros(3),
        )

        np.testing.assert_array_equal(
            guidance.desired_accel,
            np.zeros(3),
        )

        np.testing.assert_array_equal(
            guidance.desired_ang_vel,
            np.zeros(3),
        )

        np.testing.assert_array_equal(
            guidance.desired_ang_accel,
            np.zeros(3),
        )

    def test_missing_quaternion_defaults_to_identity(self):
        guidance = ConstantReferenceGuidance(
            desired_pos=np.array([1.0, 2.0, 3.0])
        )

        np.testing.assert_array_equal(
            guidance.desired_quat,
            np.array([1.0, 0.0, 0.0, 0.0]),
        )

    def test_missing_components_are_float_arrays(self):
        guidance = ConstantReferenceGuidance(
            desired_pos=np.array([1.0, 2.0, 3.0])
        )

        assert guidance.desired_vel.dtype == float
        assert guidance.desired_accel.dtype == float
        assert guidance.desired_ang_vel.dtype == float
        assert guidance.desired_ang_accel.dtype == float
        assert guidance.desired_quat.dtype == float


class TestConstantReferenceGuidanceComputeReference:

    def test_returns_guidance_output(self):
        guidance = ConstantReferenceGuidance(
            desired_pos=np.array([1.0, 2.0, 3.0])
        )

        output = guidance.compute_reference(
            navigation_data=None,
            simulation_data=None,
        )

        assert isinstance(output, GuidanceOutput)

    def test_returns_configured_state(self):
        desired_pos = np.array([1.0, 2.0, 3.0])
        desired_vel = np.array([4.0, 5.0, 6.0])
        desired_accel = np.array([7.0, 8.0, 9.0])
        desired_quat = np.array([0.1, 0.2, 0.3, 0.4])
        desired_ang_vel = np.array([0.5, 0.6, 0.7])
        desired_ang_accel = np.array([0.8, 0.9, 1.0])

        guidance = ConstantReferenceGuidance(
            desired_pos=desired_pos,
            desired_vel=desired_vel,
            desired_accel=desired_accel,
            desired_quat=desired_quat,
            desired_ang_vel=desired_ang_vel,
            desired_ang_accel=desired_ang_accel,
        )

        output = guidance.compute_reference(
            navigation_data=None,
            simulation_data=None,
        )

        np.testing.assert_array_equal(
            output.state.position,
            desired_pos,
        )
        np.testing.assert_array_equal(
            output.state.velocity,
            desired_vel,
        )
        np.testing.assert_array_equal(
            output.state.acceleration,
            desired_accel,
        )
        np.testing.assert_array_equal(
            output.state.attitude,
            desired_quat,
        )
        np.testing.assert_array_equal(
            output.state.angular_velocity,
            desired_ang_vel,
        )
        np.testing.assert_array_equal(
            output.state.angular_acceleration,
            desired_ang_accel,
        )

    def test_reference_is_constant_across_calls(self):
        guidance = ConstantReferenceGuidance(
            desired_pos=np.array([1.0, 2.0, 3.0]),
            desired_vel=np.array([4.0, 5.0, 6.0]),
        )

        output_1 = guidance.compute_reference(
            navigation_data=None,
            simulation_data=None,
        )

        output_2 = guidance.compute_reference(
            navigation_data=None,
            simulation_data=None,
        )

        np.testing.assert_array_equal(
            output_1.state.position,
            output_2.state.position,
        )

        np.testing.assert_array_equal(
            output_1.state.velocity,
            output_2.state.velocity,
        )

    def test_navigation_data_does_not_change_reference(self):
        guidance = ConstantReferenceGuidance(
            desired_pos=np.array([10.0, 20.0, 30.0])
        )

        navigation_1 = object()
        navigation_2 = object()

        output_1 = guidance.compute_reference(
            navigation_data=navigation_1,
            simulation_data=None,
        )

        output_2 = guidance.compute_reference(
            navigation_data=navigation_2,
            simulation_data=None,
        )

        np.testing.assert_array_equal(
            output_1.state.position,
            output_2.state.position,
        )

    def test_default_components_are_present_in_output(self):
        guidance = ConstantReferenceGuidance(
            desired_pos=np.array([1.0, 2.0, 3.0])
        )

        output = guidance.compute_reference(
            navigation_data=None,
            simulation_data=None,
        )

        np.testing.assert_array_equal(
            output.state.velocity,
            np.zeros(3),
        )

        np.testing.assert_array_equal(
            output.state.acceleration,
            np.zeros(3),
        )

        np.testing.assert_array_equal(
            output.state.attitude,
            np.array([1.0, 0.0, 0.0, 0.0]),
        )

        np.testing.assert_array_equal(
            output.state.angular_velocity,
            np.zeros(3),
        )

        np.testing.assert_array_equal(
            output.state.angular_acceleration,
            np.zeros(3),
        )


# ============================================================
# CustomGuidanceLaw
# ============================================================


class TestCustomGuidanceLaw:

    def test_initialization_with_callable(self):
        def reference_function(navigation, simulation):
            return GuidanceOutput(state=None)

        guidance = CustomGuidanceLaw(reference_function)

        assert guidance.custom_reference_function is reference_function

    def test_non_callable_is_rejected(self):
        with pytest.raises(ValueError):
            CustomGuidanceLaw("not callable")

    def test_compute_reference_calls_custom_function(self):
        received = {}

        expected_output = GuidanceOutput(state=None)

        def reference_function(navigation, simulation):
            received["navigation"] = navigation
            received["simulation"] = simulation

            return expected_output

        guidance = CustomGuidanceLaw(reference_function)

        navigation = object()
        simulation = object()

        output = guidance.compute_reference(
            navigation,
            simulation,
        )

        assert received["navigation"] is navigation
        assert received["simulation"] is simulation
        assert output is expected_output

    def test_compute_reference_rejects_invalid_output(self):
        def reference_function(navigation, simulation):
            return np.zeros(3)

        guidance = CustomGuidanceLaw(reference_function)

        with pytest.raises(TypeError):
            guidance.compute_reference(
                object(),
                object(),
            )


# ============================================================
# GuidancePlaceholder
# ============================================================


class TestGuidancePlaceholder:

    def test_compute_reference_raises_not_implemented(self):
        guidance = GuidancePlaceholder()

        with pytest.raises(NotImplementedError):
            guidance.compute_reference(
                navigation_data=None,
                simulation_data=None,
            )