import numpy as np

from py.modules.sensors.sensor_base import SensorBase


class GaussianStdSensor(SensorBase):
    """
    Sensor model with independent Gaussian noise.

    Each state component is corrupted by zero-mean Gaussian noise with the
    same standard deviation.
    """

    def __init__(
        self,
        noise_standard_deviation: float,
        random_seed: int | None = None,        
    ):
        self.noise_standard_deviation = noise_standard_deviation
        self.random_seed = random_seed
        self.sensor_type = "TBD - GAUSSIAN_STD_SENSOR"  # Placeholder for sensor type, to be defined in subclasses

    def measure(
        self,
        true_state: np.ndarray,
    ) -> np.ndarray:
        """
        Generate a noisy measurement.

        Parameters
        ----------
        true_state : np.ndarray
            True state vector.

        Returns
        -------
        np.ndarray
            Noisy measurement.
        """

        # ==========================================================
        # Random generator initialization
        # ==========================================================

        if self.random_seed is not None:
            np.random.seed(
                self.random_seed
            )

        # ==========================================================
        # Gaussian measurement noise
        # ==========================================================

        measurement_noise = np.random.normal(
            loc=0.0,
            scale=self.noise_standard_deviation,
            size=true_state.shape,
        )

        return true_state + measurement_noise

    def get_output(self, true_state: np.ndarray, true_ref_state: np.ndarray | None = None) -> np.ndarray:
        """
        Get the sensor output, which may include noise or other effects.

        Parameters
        ----------
        true_state : np.ndarray
            True state vector.

        Returns
        -------
        np.ndarray
            Sensor output.
        """
        return self.measure(true_state)

class GaussianCovarianceSensor(SensorBase):
    """
    Sensor model with correlated Gaussian noise.

    The measurement noise is defined by a full covariance matrix.
    """

    def __init__(
        self,
        noise_covariance: np.ndarray,
        random_seed: int | None = None,
    ):
        self.noise_covariance = noise_covariance
        self.random_seed = random_seed
        self.sensor_type = "TBD - GAUSSIAN_COV_SENSOR"  # Placeholder for sensor type, to be defined in subclasses

    def measure(
        self,
        true_state: np.ndarray,
    ) -> np.ndarray:
        """
        Generate a noisy measurement.

        Parameters
        ----------
        true_state : np.ndarray
            True state vector.

        Returns
        -------
        np.ndarray
            Noisy measurement.
        """

        # ==========================================================
        # Random generator initialization
        # ==========================================================

        if self.random_seed is not None:
            np.random.seed(
                self.random_seed
            )

        # ==========================================================
        # Correlated Gaussian measurement noise
        # ==========================================================

        measurement_noise = np.random.multivariate_normal(
            mean=np.zeros(
                len(true_state)
            ),
            cov=self.noise_covariance,
        )

        return true_state + measurement_noise
    
    def get_output(self, true_state: np.ndarray, true_ref_state: np.ndarray | None = None) -> np.ndarray:
        """
        Get the sensor output, which may include noise or other effects.

        Parameters
        ----------
        true_state : np.ndarray
            True state vector.

        Returns
        -------
        np.ndarray
            Sensor output.
        """
        return self.measure(true_state)