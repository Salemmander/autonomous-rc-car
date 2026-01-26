#!/usr/bin/env python3
"""
Donkey Car - Manual Driving Controller for Raspberry Pi 5

Simplified version that only supports manual web-based driving.
No AI/autopilot features - just basic remote control.

Usage:
    python3 manage.py drive [--mock]

Options:
    --mock    Use mock camera (for testing without hardware)
"""

import sys
import time
import threading
import signal

# Import local modules
import config as cfg
from actuator import PWMSteering, PWMThrottle, PWMController
from camera import PiCamera, MockCamera
from web_controller.web import LocalWebController


class Vehicle:
    """
    Simple vehicle controller.

    Runs the main control loop, updating actuators based on web controller input.
    """

    def __init__(self):
        self.running = False
        self.parts = []

    def add(self, part, name=None):
        """Add a part to the vehicle."""
        self.parts.append({'part': part, 'name': name})

    def start(self, rate_hz=20):
        """Start the vehicle control loop."""
        self.running = True
        loop_time = 1.0 / rate_hz

        print(f"Starting vehicle loop at {rate_hz} Hz")

        try:
            while self.running:
                start = time.time()

                # Run one iteration
                self._update()

                # Sleep to maintain loop rate
                elapsed = time.time() - start
                sleep_time = loop_time - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("\nShutting down...")

        finally:
            self.shutdown()

    def _update(self):
        """Run one control loop iteration."""
        # This is called by the main loop
        # Parts are updated via their threaded methods
        pass

    def shutdown(self):
        """Shutdown all parts."""
        self.running = False
        for entry in self.parts:
            part = entry['part']
            if hasattr(part, 'shutdown'):
                try:
                    part.shutdown()
                except Exception as e:
                    print(f"Error shutting down {entry['name']}: {e}")


def drive(use_mock_camera=False):
    """
    Run the vehicle in manual drive mode.

    Args:
        use_mock_camera: Use mock camera for testing
    """
    print("Initializing Donkey Car...")
    print(f"  Steering: Channel {cfg.STEERING_CHANNEL}, PWM {cfg.STEERING_LEFT_PWM}-{cfg.STEERING_RIGHT_PWM}")
    print(f"  Throttle: Channel {cfg.THROTTLE_CHANNEL}, PWM {cfg.THROTTLE_REVERSE_PWM}-{cfg.THROTTLE_FORWARD_PWM}")

    # Create vehicle
    V = Vehicle()

    # Initialize camera
    if use_mock_camera:
        print("Using mock camera (test mode)")
        cam = MockCamera(resolution=cfg.CAMERA_RESOLUTION, framerate=cfg.CAMERA_FRAMERATE)
    else:
        print("Initializing Pi camera...")
        cam = PiCamera(resolution=cfg.CAMERA_RESOLUTION, framerate=cfg.CAMERA_FRAMERATE)

    V.add(cam, name='camera')

    # Initialize web controller
    print("Initializing web controller...")
    ctr = LocalWebController()
    V.add(ctr, name='web_controller')

    # Initialize actuators
    print("Initializing actuators...")
    steering = PWMSteering(
        channel=cfg.STEERING_CHANNEL,
        left_pulse=cfg.STEERING_LEFT_PWM,
        right_pulse=cfg.STEERING_RIGHT_PWM,
        address=cfg.PCA9685_I2C_ADDR
    )
    V.add(steering, name='steering')

    throttle = PWMThrottle(
        channel=cfg.THROTTLE_CHANNEL,
        max_pulse=cfg.THROTTLE_FORWARD_PWM,
        zero_pulse=cfg.THROTTLE_STOPPED_PWM,
        min_pulse=cfg.THROTTLE_REVERSE_PWM,
        address=cfg.PCA9685_I2C_ADDR
    )
    V.add(throttle, name='throttle')

    # Center steering and stop throttle
    steering.run(0)
    throttle.run(0)

    # Start camera thread
    cam_thread = threading.Thread(target=cam.update, daemon=True)
    cam_thread.start()

    # Start web server thread
    web_thread = threading.Thread(target=ctr.update, daemon=True)
    web_thread.start()

    # Give threads time to start
    time.sleep(1)

    # Main control loop
    print("\nStarting control loop...")
    print("Press Ctrl+C to stop\n")

    loop_time = 1.0 / cfg.DRIVE_LOOP_HZ

    try:
        while True:
            start = time.time()

            # Get camera frame
            img = cam.run_threaded()

            # Get control inputs from web controller
            angle, throttle_val, mode, recording = ctr.run_threaded(img)

            # Apply controls
            steering.run(angle)
            throttle.run(throttle_val)

            # Maintain loop rate
            elapsed = time.time() - start
            sleep_time = loop_time - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n\nShutting down...")

    finally:
        # Stop actuators
        print("Stopping motors...")
        steering.shutdown()
        throttle.shutdown()

        # Stop camera
        print("Stopping camera...")
        cam.shutdown()

        # Stop PWM controller
        pwm = PWMController.get_instance()
        pwm.shutdown()

        print("Goodbye!")


def main():
    """Main entry point."""
    use_mock = "--mock" in sys.argv

    if "drive" in sys.argv or len(sys.argv) == 1:
        drive(use_mock_camera=use_mock)
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
