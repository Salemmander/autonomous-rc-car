"""
Vehicle class - unified interface for RC car control.
"""

import threading
import time
import numpy as np
from picamera2 import Picamera2

from actuator import PWMSteering, PWMThrottle
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
        # Steering and throttle
        self._steering = PWMSteering(
            channel=config.STEERING_CHANNEL,
            left_pulse=config.STEERING_LEFT_PWM,
            right_pulse=config.STEERING_RIGHT_PWM,
        )
        self._throttle = PWMThrottle()
        self.current_steering = 0.0
        self.current_throttle = 0.0

        # Camera
        self._picam = None
        self._frame = None
        self._frame_lock = threading.Lock()
        self._camera_thread = None

        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running

    # --- Steering and throttle ---

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

    # --- Camera ---

    def _camera_loop(self):
        """Background thread that continuously captures frames."""
        height, width = config.CAMERA_RESOLUTION
        self._picam = Picamera2()
        cam_config = self._picam.create_video_configuration(
            main={"size": (width, height), "format": "RGB888"}
        )
        self._picam.configure(cam_config)
        self._picam.start()
        time.sleep(0.5)  # Camera warmup

        while self._running:
            try:
                frame = self._picam.capture_array()
                with self._frame_lock:
                    self._frame = frame
            except Exception as e:
                print(f"Camera error: {e}")
                time.sleep(0.1)

    def get_frame(self) -> np.ndarray | None:
        """Get latest camera frame (H, W, 3) RGB array, or None if not ready."""
        with self._frame_lock:
            return self._frame

    # --- Lifecycle ---

    def start(self):
        """Start camera thread and initialize to stopped state."""
        if self._running:
            return

        self._running = True

        self._camera_thread = threading.Thread(target=self._camera_loop, daemon=True)
        self._camera_thread.start()

        self._steering.run(0.0)
        self._throttle.run(0.0)

    def shutdown(self):
        """Stop motors, center steering, release resources."""
        if not self._running:
            return

        self.stop()
        self._running = False

        if self._picam:
            self._picam.stop()
            self._picam.close()
            self._picam = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()
        return False
