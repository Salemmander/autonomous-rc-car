"""
Pure Python actuator control for PCA9685 PWM controller.

Replaces the proprietary XiaoRGEEK .so binaries with standard libraries.
Uses Adafruit CircuitPython PCA9685 library.

Install: pip3 install adafruit-circuitpython-pca9685 --break-system-packages
"""

import time
import board
import busio
from adafruit_pca9685 import PCA9685


class PWMController:
    """
    PCA9685 PWM controller wrapper.
    """
    _instance = None

    def __init__(self, address=0x40, frequency=50):
        self.address = address
        self.frequency = frequency
        self._pca = None

    def _init_pca(self):
        if self._pca is None:
            i2c = busio.I2C(board.SCL, board.SDA)
            self._pca = PCA9685(i2c, address=self.address)
            self._pca.frequency = self.frequency
        return self._pca

    @classmethod
    def get_instance(cls, address=0x40, frequency=50):
        """Get singleton instance of PWM controller."""
        if cls._instance is None:
            cls._instance = cls(address, frequency)
        return cls._instance

    def set_pwm(self, channel, pulse):
        """
        Set PWM pulse on channel.

        Args:
            channel: PCA9685 channel (0-15)
            pulse: Pulse value (0-4095)
        """
        pca = self._init_pca()
        # PCA9685 uses 12-bit values (0-4095)
        # Convert our simple pulse value to duty cycle
        # For servos at 50Hz, pulse of 0-4095 maps to duty cycle
        pca.channels[channel].duty_cycle = int(pulse * 16)  # Scale to 16-bit

    def shutdown(self):
        """Turn off all PWM outputs."""
        if self._pca is not None:
            for i in range(16):
                self._pca.channels[i].duty_cycle = 0
            self._pca.deinit()
            self._pca = None


class PWMSteering:
    """
    Steering control via PWM.

    Maps angle (-1 to 1) to PWM pulse values.
    """

    def __init__(self, channel=1, left_pulse=40, right_pulse=150, address=0x40):
        self.channel = channel
        self.left_pulse = left_pulse
        self.right_pulse = right_pulse
        self.pwm = PWMController.get_instance(address)
        self.current_angle = 0.0

    def run(self, angle):
        """
        Set steering angle.

        Args:
            angle: -1.0 (full left) to 1.0 (full right)
        """
        if angle is None:
            return

        # Clamp angle to valid range
        angle = max(-1.0, min(1.0, angle))
        self.current_angle = angle

        # Map -1..1 to left_pulse..right_pulse
        pulse = int(self.left_pulse + (angle + 1) / 2 * (self.right_pulse - self.left_pulse))
        self.pwm.set_pwm(self.channel, pulse)

    def shutdown(self):
        """Center steering on shutdown."""
        self.run(0)


class PWMThrottle:
    """
    Throttle control via PWM.

    Maps throttle (-1 to 1) to PWM pulse values.
    """

    def __init__(self, channel=0, max_pulse=200, zero_pulse=100, min_pulse=0, address=0x40):
        self.channel = channel
        self.max_pulse = max_pulse
        self.zero_pulse = zero_pulse
        self.min_pulse = min_pulse
        self.pwm = PWMController.get_instance(address)
        self.current_throttle = 0.0

    def run(self, throttle):
        """
        Set throttle.

        Args:
            throttle: -1.0 (full reverse) to 1.0 (full forward)
        """
        if throttle is None:
            return

        # Clamp throttle to valid range
        throttle = max(-1.0, min(1.0, throttle))
        self.current_throttle = throttle

        # Map throttle to pulse
        if throttle >= 0:
            pulse = int(self.zero_pulse + throttle * (self.max_pulse - self.zero_pulse))
        else:
            pulse = int(self.zero_pulse + throttle * (self.zero_pulse - self.min_pulse))

        self.pwm.set_pwm(self.channel, pulse)

    def shutdown(self):
        """Stop motor on shutdown."""
        self.run(0)


# Test function
if __name__ == "__main__":
    print("Testing actuators...")

    steering = PWMSteering(channel=1, left_pulse=40, right_pulse=150)
    throttle = PWMThrottle(channel=0, max_pulse=200, zero_pulse=100, min_pulse=0)

    print("Centering steering...")
    steering.run(0)
    time.sleep(1)

    print("Steering left...")
    steering.run(-0.5)
    time.sleep(0.5)

    print("Steering right...")
    steering.run(0.5)
    time.sleep(0.5)

    print("Centering...")
    steering.run(0)
    time.sleep(0.5)

    print("Small throttle forward...")
    throttle.run(0.1)
    time.sleep(0.5)

    print("Stopping...")
    throttle.run(0)

    print("Done!")
