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

    def get_ideal_measurement(self, spacecraft_data, simulation_data) -> np.ndarray:
        """
        Get the ideal sensor measurement without any noise or errors.

        Parameters
        ----------
        spacecraft_data : SpacecraftData
            The current state of the spacecraft.
        simulation_data : SimulationData
            The current state of the simulation.

        Returns
        -------
        np.ndarray
            Ideal sensor measurement.
        """
        # This method should be implemented in subclasses to return the ideal measurement based on the spacecraft and simulation data.
        raise NotImplementedError("Subclasses must implement this method.")

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

    def get_ideal_measurement(self, spacecraft_data, simulation_data) -> np.ndarray:
        """
        Get the ideal sensor measurement without any noise or errors.

        Parameters
        ----------
        spacecraft_data : SpacecraftData
            The current state of the spacecraft.
        simulation_data : SimulationData
            The current state of the simulation.

        Returns
        -------
        np.ndarray
            Ideal sensor measurement.
        """
        # This method should be implemented in subclasses to return the ideal measurement based on the spacecraft and simulation data.
        raise NotImplementedError("Subclasses must implement this method.")
    

class AbsoluteSensor(SensorBase):

    def __init__(self, sample_rate_freq, verbose=False):

        super().__init__(
            sensor_type="AbsoluteSensor",
            sensor_pos=np.zeros(3),
            sensor_rotation=np.zeros(3),
            sample_rate_freq= sample_rate_freq,
            error_models=(),
            verbose=verbose
        )            

    def get_ideal_measurement(self, spacecraft_data, simulation_data) -> np.ndarray:

        parent_spacecraft_data_to_return = (            
            spacecraft_data.true_state.position,
            spacecraft_data.true_state.velocity,
            spacecraft_data.true_state.acceleration,
            spacecraft_data.true_state.attitude,
            spacecraft_data.true_state.angular_velocity,
            spacecraft_data.true_state.angular_acceleration
        )

        reference_spacecraft_data_to_return = None
        reference_spacecraft_name = spacecraft_data.target_name        
        
        if reference_spacecraft_name is not None:
            
            for spacecraft in simulation_data.spacecrafts:
                if spacecraft.name == reference_spacecraft_name:                    
                    reference_spacecraft_data = spacecraft.spacecraft_data
                    reference_spacecraft_data_to_return = (
                        reference_spacecraft_data.true_state.position,
                        reference_spacecraft_data.true_state.velocity,
                        reference_spacecraft_data.true_state.acceleration,
                        reference_spacecraft_data.true_state.attitude,
                        reference_spacecraft_data.true_state.angular_velocity,
                        reference_spacecraft_data.true_state.angular_acceleration
                    )
                    
                    break
        else:
            reference_spacecraft_data_to_return = (            
                np.zeros(3),
                np.zeros(3),
                np.zeros(3),
                np.array([0, 0, 0, 1]),
                np.zeros(3),
                np.zeros(3)
            )
                       
        return parent_spacecraft_data_to_return, reference_spacecraft_data_to_return
    