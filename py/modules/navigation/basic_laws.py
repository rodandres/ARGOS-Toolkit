import numpy as np
from py.modules.navigation.navigation_base import NavigationBase
from py.general.dataclasses import EstimationOutput

class IdealNavigation(NavigationBase):
    def __init__(self):
        super().__init__()        

    def _check_initialization(self):
        # No specific initialization checks for IdealNavigation
        pass

    def estimate(self, sensors: list) -> dict:
        """
        Estimate the spacecraft's state based on sensor data.
        For IdealNavigation, it simply returns the true state from the sensors.

        Args:
            sensors (list): A list of sensor objects providing data for estimation.
        """
        absolute_sensor = None
        for sensor in sensors:
            if sensor.type == "AbsoluteSensor":  # Ensure the sensor is of the correct type
                absolute_sensor = sensor
                break

        if absolute_sensor is None:
            raise ValueError("No AbsoluteSensor found in the provided sensors.")

        sensor_data = absolute_sensor.get_measurement()
        spacecraft_sensor_data = sensor_data[0]
        reference_sensor_data = sensor_data[1]

                 
        return EstimationOutput(
            spacecraft_position=spacecraft_sensor_data[0],
            spacecraft_velocity=spacecraft_sensor_data[1],
            spacecraft_acceleration=spacecraft_sensor_data[2],
            spacecraft_attitude=spacecraft_sensor_data[3],
            spacecraft_angular_velocity=spacecraft_sensor_data[4],
            spacecraft_angular_acceleration=spacecraft_sensor_data[5],

            reference_position=reference_sensor_data[0],
            reference_velocity=reference_sensor_data[1],
            reference_acceleration=reference_sensor_data[2],
            reference_attitude=reference_sensor_data[3],
            reference_angular_velocity=reference_sensor_data[4],
            reference_angular_acceleration=reference_sensor_data[5]
        )

class CustomNavigation(NavigationBase):
    def __init__(self, custom_estimation_function):
        self.custom_estimation_function = custom_estimation_function
        super().__init__()

    def _check_initialization(self):
        if not callable(self.custom_estimation_function):
            raise ValueError("custom_estimation_function must be callable.")

    def estimate(self, sensors: list) -> dict:
        """
        Estimate the spacecraft's state using a custom function.

        Args:
            sensors (list): A list of sensor objects providing data for estimation.
        Returns:
            An EstimationOutput object containing the estimated state of the spacecraft.
        """
        return self.custom_estimation_function(sensors)    