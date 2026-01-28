"""
Actuator module for XiaoRGEEK servo/motor control.

Steering: I2C to address 0x17, command 0xFF, data [servo_num, value]
Motors: GPIO pins (H-bridge control)
  - Motor 1: ENA=GPIO13, IN1=GPIO19, IN2=GPIO16
  - Motor 2: ENB=GPIO20, IN3=GPIO21, IN4=GPIO26
"""

import smbus
import lgpio
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
        pulse = int(
            self.left_pulse + (angle + 1) / 2 * (self.right_pulse - self.left_pulse)
        )
        self.controller.set_servo(self.channel, pulse)

    def shutdown(self):
        """Center steering on shutdown (servo stays locked until power cycle)."""
        center = (self.left_pulse + self.right_pulse) // 2
        self.controller.set_servo(self.channel, center)


class PWMThrottle:
    """
    Throttle control via GPIO H-bridge (Motor 1 on PWR.A53.A board).

    Maps throttle (-1 to 1) to motor speed and direction.
    """

    # Motor 1 GPIO pins
    M1_ENA = 13  # PWM/Enable
    M1_IN1 = 19  # Direction
    M1_IN2 = 16  # Direction

    _gpio_handle = None

    def __init__(self, pwm_freq=100):
        """
        Initialize throttle controller.

        Args:
            pwm_freq: PWM frequency in Hz (default 100)
        """
        self.pwm_freq = pwm_freq
        self.current_throttle = 0.0

        # Open GPIO if not already open
        if PWMThrottle._gpio_handle is None:
            PWMThrottle._gpio_handle = lgpio.gpiochip_open(0)

        self.h = PWMThrottle._gpio_handle

        # Setup pins as outputs
        for pin in [self.M1_ENA, self.M1_IN1, self.M1_IN2]:
            try:
                lgpio.gpio_claim_output(self.h, pin, 0)
            except:
                pass  # Already claimed

    def run(self, throttle):
        """Set throttle: -1.0 (reverse) to 1.0 (forward)."""
        if throttle is None:
            return

        # Clamp throttle to valid range
        throttle = max(-1.0, min(1.0, throttle))
        self.current_throttle = throttle

        # Convert to speed percentage (0-100)
        speed = abs(throttle) * 100

        if throttle > 0.01:
            # Forward: IN1=HIGH, IN2=LOW
            lgpio.gpio_write(self.h, self.M1_IN1, 1)
            lgpio.gpio_write(self.h, self.M1_IN2, 0)
            lgpio.tx_pwm(self.h, self.M1_ENA, self.pwm_freq, speed)
        elif throttle < -0.01:
            # Backward: IN1=LOW, IN2=HIGH
            lgpio.gpio_write(self.h, self.M1_IN1, 0)
            lgpio.gpio_write(self.h, self.M1_IN2, 1)
            lgpio.tx_pwm(self.h, self.M1_ENA, self.pwm_freq, speed)
        else:
            # Stop: IN1=LOW, IN2=LOW, PWM=0
            lgpio.gpio_write(self.h, self.M1_IN1, 0)
            lgpio.gpio_write(self.h, self.M1_IN2, 0)
            lgpio.tx_pwm(self.h, self.M1_ENA, self.pwm_freq, 0)

    def shutdown(self):
        """Stop motor on shutdown."""
        self.run(0)


# Test function
if __name__ == "__main__":
    import sys

    print("Testing XiaoRGEEK actuators...")

    steering = PWMSteering(channel=1, left_pulse=0, right_pulse=170)
    throttle = PWMThrottle()

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
        print("Throttle forward (50%)...")
        throttle.run(0.5)
        time.sleep(1.5)

        print("Stopping...")
        throttle.run(0)
        time.sleep(0.5)

        print("Throttle reverse (50%)...")
        throttle.run(-0.5)
        time.sleep(1.5)

        print("Stopping...")
        throttle.run(0)

    # Flush buffer before exit
    steering.run(0)
    steering.run(0)

    # Reset singleton so next run starts fresh
    XRServoController._instance = None

    print("Done!")
