import numpy as np
from py.modules.navigation.navigation_base import NavigationBase
from py.general.dataclasses import EstimationOutput, StateVariables

class IdealNavigation(NavigationBase):
    def __init__(self, sensor_to_be_use=1):
        super().__init__()        
        self.sensor_to_be_use = sensor_to_be_use

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
        absolute_sensor_count = 0

        for sensor in sensors:
            if sensor.type == "AbsoluteSensor":
                absolute_sensor_count += 1

                if absolute_sensor_count == self.sensor_to_be_use:
                    absolute_sensor = sensor
                    break

        if absolute_sensor is None:
            raise ValueError("No AbsoluteSensor found in the provided sensors.")

        sensor_data = absolute_sensor.get_measurement()
        spacecraft_sensor_data = sensor_data[0]
        reference_sensor_data = sensor_data[1]

        spacecraft_state = StateVariables(
            position=spacecraft_sensor_data[0],
            velocity=spacecraft_sensor_data[1],
            acceleration=spacecraft_sensor_data[2],
            attitude=spacecraft_sensor_data[3],
            angular_velocity=spacecraft_sensor_data[4],
            angular_acceleration=spacecraft_sensor_data[5]
        )

        reference_state = StateVariables(
            position=reference_sensor_data[0],
            velocity=reference_sensor_data[1],
            acceleration=reference_sensor_data[2],
            attitude=reference_sensor_data[3],
            angular_velocity=reference_sensor_data[4],
            angular_acceleration=reference_sensor_data[5]
        )

        return EstimationOutput(
            spacecraft_state=spacecraft_state,
            reference_state=reference_state
        )

class CustomNavigation(NavigationBase):
    def __init__(self, custom_estimation_function):
        self.custom_estimation_function = custom_estimation_function
        super().__init__()

    def _check_initialization(self):
        if not callable(self.custom_estimation_function):
            raise ValueError("custom_estimation_function must be callable.")        

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

        output = self.custom_estimation_function(sensors)

        if not isinstance(output, EstimationOutput):
            raise TypeError("custom_estimation_function must return an EstimationOutput object.")

        return output