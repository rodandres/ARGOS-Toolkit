import numpy as np
import pytest

from argos.sensors.error_models import (
    Bias,
    WhiteNoise,
    RandomWalk,
    Saturation,
    Quantization,
)


# ============================================================================
# Bias
# ============================================================================


def test_bias_adds_constant_offset():
    measurement = np.array([1.0, 2.0, 3.0])
    bias = np.array([0.1, -0.2, 0.3])

    model = Bias(bias)

    result = model.apply(measurement)

    assert np.allclose(
        result,
        [1.1, 1.8, 3.3],
    )


def test_bias_does_not_modify_input():
    measurement = np.array([1.0, 2.0, 3.0])
    original = measurement.copy()

    Bias(np.ones(3)).apply(measurement)

    assert np.array_equal(measurement, original)


# ============================================================================
# WhiteNoise
# ============================================================================


def test_white_noise_is_deterministic_with_seed(monkeypatch):
    measurement = np.zeros(3)
    model = WhiteNoise(np.ones(3))

    np.random.seed(42)
    result_1 = model.apply(measurement)

    np.random.seed(42)
    result_2 = model.apply(measurement)

    assert np.allclose(result_1, result_2)


def test_white_noise_with_zero_std_preserves_measurement():
    measurement = np.array([1.0, -2.0, 3.0])

    result = WhiteNoise(np.zeros(3)).apply(measurement)

    assert np.allclose(result, measurement)


# ============================================================================
# RandomWalk
# ============================================================================


def test_random_walk_requires_dt():
    model = RandomWalk(np.ones(3))

    with pytest.raises(ValueError):
        model.apply(np.zeros(3))


def test_random_walk_zero_std_preserves_measurement():
    measurement = np.array([1.0, 2.0, 3.0])

    model = RandomWalk(np.zeros(3))

    result = model.apply(
        measurement,
        dt=1.0,
    )

    assert np.allclose(result, measurement)
    assert np.allclose(model.bias, np.zeros(3))


def test_random_walk_updates_internal_bias():
    model = RandomWalk(np.ones(3))

    np.random.seed(42)

    measurement = np.zeros(3)

    result = model.apply(
        measurement,
        dt=1.0,
    )

    assert np.allclose(result, model.bias)
    assert not np.allclose(model.bias, np.zeros(3))


def test_random_walk_accumulates_bias():
    model = RandomWalk(np.ones(3))

    np.random.seed(42)

    first = model.apply(
        np.zeros(3),
        dt=1.0,
    )

    second = model.apply(
        np.zeros(3),
        dt=1.0,
    )

    assert not np.allclose(first, second)
    assert np.allclose(second, model.bias)


# ============================================================================
# Saturation
# ============================================================================


def test_saturation_clips_measurement():
    measurement = np.array([-5.0, -1.0, 0.5, 2.0, 5.0])

    model = Saturation(1.0)

    result = model.apply(measurement)

    assert np.allclose(
        result,
        [-1.0, -1.0, 0.5, 1.0, 1.0],
    )


def test_saturation_supports_component_limits():
    measurement = np.array([5.0, 5.0, 5.0])
    limits = np.array([1.0, 2.0, 3.0])

    result = Saturation(limits).apply(measurement)

    assert np.allclose(
        result,
        [1.0, 2.0, 3.0],
    )


# ============================================================================
# Quantization
# ============================================================================


def test_quantization_can_be_disabled():
    measurement = np.array([0.123, -0.456, 0.789])

    result = Quantization(
        limit=1.0,
        bits=None,
    ).apply(measurement)

    assert np.array_equal(result, measurement)


def test_quantization_with_infinite_limit_is_disabled():
    measurement = np.array([0.123, -0.456, 0.789])

    result = Quantization(
        limit=np.inf,
        bits=8,
    ).apply(measurement)

    assert np.array_equal(result, measurement)


def test_quantization_rounds_to_quantization_levels():
    model = Quantization(
        limit=1.0,
        bits=2,
    )

    # step = 2 / (2² - 1) = 2/3
    measurement = np.array(
        [
            0.1,
            0.7,
            -0.7,
        ]
    )

    result = model.apply(measurement)

    step = 2.0 / 3.0

    expected = np.round(measurement / step) * step

    assert np.allclose(result, expected)