#!/usr/bin/env python3
"""
Donkey Car - Manual Driving Controller for Raspberry Pi 5

Simplified version that only supports manual web-based driving.
No AI/autopilot features - just basic remote control.

Usage:
    python3 manage.py drive
"""

import sys
import time
import threading

import config as cfg
from vehicle import Vehicle
from web_controller.web import LocalWebController


def drive():
    """Run the vehicle in manual drive mode."""
    print("Initializing Donkey Car...")
    print(f"  Steering: I2C channel {cfg.STEERING_CHANNEL}, PWM {cfg.STEERING_LEFT_PWM}-{cfg.STEERING_RIGHT_PWM}")  # fmt: skip
    print("  Throttle: GPIO motor control (pins 13, 16, 19)")

    vehicle = Vehicle()
    ctr = LocalWebController()

    print("Starting vehicle...")
    vehicle.start()

    print("Starting web controller...")
    web_thread = threading.Thread(target=ctr.update, daemon=True)
    web_thread.start()

    time.sleep(1)  # Let threads start

    print("\nStarting control loop...")
    print("Press Ctrl+C to stop\n")

    loop_time = 1.0 / cfg.DRIVE_LOOP_HZ

    try:
        while True:
            start = time.time()

            img = vehicle.get_frame()
            angle, throttle, mode, recording = ctr.run_threaded(img)
            vehicle.drive(angle, throttle)

            elapsed = time.time() - start
            sleep_time = loop_time - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n\nShutting down...")

    finally:
        vehicle.shutdown()
        print("Goodbye!")


def main():
    """Main entry point."""
    if "drive" in sys.argv or len(sys.argv) == 1:
        drive()
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
