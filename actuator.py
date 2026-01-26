"""
Actuator module for XiaoRGEEK servo/motor control.

Controls steering and throttle via the XiaoRGEEK HAT's I2C interface.
Protocol: Address 0x17, command 0xFF, data [servo_num, value]
"""

import smbus
import time


class XRServoController:
    """
    XiaoRGEEK servo controller interface.

    Singleton pattern - only one instance talks to the hardware.
    """

    _instance = None

    @classmethod
    def get_instance(cls, address=0x17, bus_num=1):
        if cls._instance is None:
            cls._instance = cls(address, bus_num)
        return cls._instance

    def __init__(self, address=0x17, bus_num=1):
        self.address = address
        self.bus = smbus.SMBus(bus_num)
        # Prime the buffer by sending a few dummy commands
        self._prime_buffer()

    def _prime_buffer(self):
        """Prime the command buffer to clear any stale state."""
        # Send dummy commands to flush the buffer
        for _ in range(3):
            try:
                self.bus.write_i2c_block_data(self.address, 0xFF, [1, 85])
            except:
                pass
            time.sleep(0.02)

    def set_servo(self, channel, value):
        """
        Set servo position.

        Args:
            channel: Servo channel (0=throttle, 1=steering)
            value: PWM value (0=left, 100=center, 200=right for steering)

        Note: XiaoRGEEK controller has a one-command buffer, so we send
        the command twice to flush it and execute immediately.
        """
        cmd = [channel, int(value)]
        # Send 3 times to ensure command executes immediately
        self.bus.write_i2c_block_data(self.address, 0xFF, cmd)
        self.bus.write_i2c_block_data(self.address, 0xFF, cmd)
        self.bus.write_i2c_block_data(self.address, 0xFF, cmd)

    def shutdown(self):
        """Close the I2C bus."""
        if self.bus:
            self.bus.close()
            self.bus = None


# Alias for backwards compatibility
PWMController = XRServoController


class PWMSteering:
    """
    Steering control via XiaoRGEEK servo controller.

    Maps angle (-1 to 1) to PWM pulse values.
    """

    def __init__(self, channel=1, left_pulse=0, right_pulse=170, address=0x17):
        """
        Initialize steering controller.

        Args:
            channel: Servo channel (default 1 for steering)
            left_pulse: PWM value for full left
            right_pulse: PWM value for full right
            address: I2C address (default 0x17 for XiaoRGEEK)
        """
        self.channel = channel
        self.left_pulse = left_pulse
        self.right_pulse = right_pulse
        self.controller = XRServoController.get_instance(address)
        self.current_angle = 0.0

    def run(self, angle):
        """
        Set steering angle.

        Args:
            angle: -1.0 (full left) to 1.0 (full right), 0 = center
        """
        if angle is None:
            return

        # Clamp angle to valid range
        angle = max(-1.0, min(1.0, angle))
        self.current_angle = angle

        # Map -1..1 to left_pulse..right_pulse
        pulse = int(self.left_pulse + (angle + 1) / 2 * (self.right_pulse - self.left_pulse))
        self.controller.set_servo(self.channel, pulse)

    def shutdown(self):
        """Center steering on shutdown."""
        self.run(0)


class PWMThrottle:
    """
    Throttle control via XiaoRGEEK servo controller.

    Maps throttle (-1 to 1) to PWM pulse values.
    """

    def __init__(self, channel=0, max_pulse=200, zero_pulse=100, min_pulse=0, address=0x17):
        """
        Initialize throttle controller.

        Args:
            channel: Servo channel (default 0 for throttle)
            max_pulse: PWM value for full forward
            zero_pulse: PWM value for stop
            min_pulse: PWM value for full reverse
            address: I2C address (default 0x17 for XiaoRGEEK)
        """
        self.channel = channel
        self.max_pulse = max_pulse
        self.zero_pulse = zero_pulse
        self.min_pulse = min_pulse
        self.controller = XRServoController.get_instance(address)
        self.current_throttle = 0.0

    def run(self, throttle):
        """
        Set throttle.

        Args:
            throttle: -1.0 (full reverse) to 1.0 (full forward), 0 = stop
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

        self.controller.set_servo(self.channel, pulse)

    def shutdown(self):
        """Stop motor on shutdown."""
        self.run(0)


# Test function
if __name__ == "__main__":
    import sys

    print("Testing XiaoRGEEK actuators...")

    steering = PWMSteering(channel=1, left_pulse=0, right_pulse=170)
    throttle = PWMThrottle(channel=0, max_pulse=200, zero_pulse=100, min_pulse=0)

    print("Centering steering...")
    steering.run(0)
    time.sleep(1)

    print("Steering left...")
    steering.run(-1)
    time.sleep(1)

    print("Steering right...")
    steering.run(1)
    time.sleep(1)

    print("Centering...")
    steering.run(0)
    time.sleep(0.5)

    if "--throttle" in sys.argv:
        print("Small throttle forward...")
        throttle.run(0.2)
        time.sleep(1)

        print("Stopping...")
        throttle.run(0)

    # Flush buffer before exit
    steering.run(0)
    steering.run(0)

    # Reset singleton so next run starts fresh
    XRServoController._instance = None

    print("Done!")
