import numpy as np

from argos.sensors.sensor_base import SensorBase
from argos.sensors.error_models import Bias


class DummySensor(SensorBase):
    def __init__(
        self,
        measurement,
        position=np.zeros(3),
        rotation=np.zeros(3),
        **kwargs,
    ):
        super().__init__(
            type="DummySensor",
            position=position,
            rotation=rotation,
            **kwargs,
        )
        self._ideal_measurement = np.asarray(measurement)

    def get_ideal_measurement(self, spacecraft_data, simulation_data):
        return self._ideal_measurement.copy()

class DummySpacecraftData:
    def __init__(
        self,
        t,
        current_propagation_dt=0.1,
    ):
        self.t = t
        self.current_propagation_dt = current_propagation_dt


# ============================================================================
# Initialization
# ============================================================================


def test_sensor_initialization():
    sensor = DummySensor(
        [1.0, 2.0, 3.0],
        sample_rate_freq=10.0,
        name="test_sensor",
    )

    assert sensor.type == "DummySensor"
    assert sensor.name == "test_sensor"

    assert np.allclose(
        sensor.position,
        [0.0, 0.0, 0.0],
    )

    assert np.allclose(
        sensor.rotation,
        [0.0, 0.0, 0.0],
    )

    assert sensor.sample_rate_sec == 0.1


def test_sensor_initializes_identity_orientation():
    sensor = DummySensor(
        [1.0, 2.0, 3.0],
        position=np.zeros(3),
        rotation=np.zeros(3),
    )

    assert np.allclose(
        sensor.rotation_quat,
        [0.0, 0.0, 0.0, 1.0],
    )

    assert np.allclose(
        sensor.DCM_sensor_to_body,
        np.eye(3),
    )


# ============================================================================
# Measurement updates
# ============================================================================


def test_sensor_takes_first_measurement_at_initial_time():
    sensor = DummySensor(
        [1.0, 2.0, 3.0],
        sample_rate_freq=10.0,
    )

    spacecraft_data = DummySpacecraftData(t=0.0)

    sensor.update(
        spacecraft_data,
        simulation_data=None,
    )

    assert np.allclose(
        sensor.measurement,
        [1.0, 2.0, 3.0],
    )


def test_sensor_retains_previous_measurement_before_next_sample():
    sensor = DummySensor(
        [1.0, 2.0, 3.0],
        sample_rate_freq=10.0,
    )

    sensor.update(
        DummySpacecraftData(t=0.0),
        None,
    )

    sensor._ideal_measurement = np.array(
        [10.0, 20.0, 30.0]
    )

    sensor.update(
        DummySpacecraftData(t=0.05),
        None,
    )

    assert np.allclose(
        sensor.measurement,
        [1.0, 2.0, 3.0],
    )


def test_sensor_updates_at_sampling_period():
    sensor = DummySensor(
        [1.0, 2.0, 3.0],
        sample_rate_freq=10.0,
    )

    sensor.update(
        DummySpacecraftData(t=0.0),
        None,
    )

    sensor._ideal_measurement = np.array(
        [10.0, 20.0, 30.0]
    )

    sensor.update(
        DummySpacecraftData(t=0.1),
        None,
    )

    assert np.allclose(
        sensor.measurement,
        [10.0, 20.0, 30.0],
    )


# ============================================================================
# Error models
# ============================================================================


def test_sensor_applies_error_models():
    sensor = DummySensor(
        [1.0, 2.0, 3.0],
        sample_rate_freq=10.0,
        error_models=(
            Bias(np.array([1.0, 1.0, 1.0])),
            Bias(np.array([2.0, 2.0, 2.0])),
        ),
    )

    sensor.update(
        DummySpacecraftData(t=0.0),
        None,
    )

    assert np.allclose(
        sensor.measurement,
        [4.0, 5.0, 6.0],
    )


def test_sensor_get_measurement_returns_current_measurement():
    sensor = DummySensor(
        [1.0, 2.0, 3.0],
    )

    sensor.update(
        DummySpacecraftData(t=0.0),
        None,
    )

    assert np.allclose(
        sensor.get_measurement(),
        sensor.measurement,
    )