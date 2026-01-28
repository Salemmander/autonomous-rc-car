"""
Vehicle class - unified interface for RC car control.
"""

import threading
import numpy as np

from actuator import PWMSteering, PWMThrottle
from camera import PiCamera
import config


class Vehicle:
    """
    Unified interface for controlling the RC car.

    Usage:
        with Vehicle() as car:
            frame = car.get_frame()
            car.drive(0.5, 0.3)
    """

    def __init__(self):
        self._camera = PiCamera(
            resolution=config.CAMERA_RESOLUTION,
            framerate=config.CAMERA_FRAMERATE,
        )
        self._steering = PWMSteering(
            channel=config.STEERING_CHANNEL,
            left_pulse=config.STEERING_LEFT_PWM,
            right_pulse=config.STEERING_RIGHT_PWM,
        )
        self._throttle = PWMThrottle()
        self._camera_thread = None
        self._running = False

        self.current_steering = 0.0
        self.current_throttle = 0.0

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self):
        """Start camera thread and initialize to stopped state."""
        if self._running:
            return

        self._camera_thread = threading.Thread(target=self._camera.update, daemon=True)
        self._camera_thread.start()

        self._steering.run(0.0)
        self._throttle.run(0.0)

        self._running = True

    def shutdown(self):
        """Stop motors, center steering, release resources."""
        if not self._running:
            return

        self._running = False
        self._throttle.shutdown()
        self._steering.shutdown()
        self._camera.shutdown()

        self.current_steering = 0.0
        self.current_throttle = 0.0

    def steer(self, angle: float):
        """Set steering: -1.0 (left) to 1.0 (right)."""
        if not self._running:
            return
        angle = max(-1.0, min(1.0, angle))
        self.current_steering = angle
        self._steering.run(angle)

    def throttle(self, value: float):
        """Set throttle: -1.0 (reverse) to 1.0 (forward)."""
        if not self._running:
            return
        value = max(-1.0, min(1.0, value))
        self.current_throttle = value
        self._throttle.run(value)

    def drive(self, steering: float, throttle: float):
        """Set both steering and throttle."""
        self.steer(steering)
        self.throttle(throttle)

    def stop(self):
        """Emergency stop."""
        self.throttle(0.0)
        self.steer(0.0)

    def get_frame(self) -> np.ndarray | None:
        """Get latest camera frame (H, W, 3) RGB array, or None if not ready."""
        return self._camera.run_threaded()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()
        return False
