#!/usr/bin/env python3
"""
Autonomous RC Car - Manual Driving Controller for Raspberry Pi 5

Web-based manual driving with optional data collection for training.

Usage:
    python3 main.py drive
"""

import sys
import time
import threading

import torch
from PIL import Image
from src.car import config as cfg
from src.car.datastore import DataStore
from src.car.vehicle import Vehicle
from src.car.web import LocalWebController
from src.training.pilotnet import PilotNet


def drive():
    """Run the vehicle in manual drive mode."""
    print("Initializing vehicle...")

    vehicle = Vehicle()
    ctr = LocalWebController()
    datastore = DataStore()

    print("Starting vehicle...")
    vehicle.start()

    print("Starting web controller...")
    web_thread = threading.Thread(target=ctr.update, daemon=True)
    web_thread.start()

    time.sleep(1)  # Let threads start

    print("\nStarting control loop...")
    print("Press Ctrl+C to stop\n")

    loop_time = 1.0 / cfg.DRIVE_LOOP_HZ
    record_every_n = max(1, cfg.DRIVE_LOOP_HZ // cfg.RECORD_FPS)
    loop_count = 0

    try:
        while True:
            start = time.time()

            img = vehicle.get_frame()
            angle, throttle, mode, recording = ctr.run_threaded(img)
            vehicle.drive(angle, throttle)

            if recording and not datastore.is_recording:
                datastore.start_session()
            elif not recording and datastore.is_recording:
                datastore.stop_session()

            if recording and loop_count % record_every_n == 0:
                datastore.record(img, angle, throttle)

            loop_count += 1

            elapsed = time.time() - start
            sleep_time = loop_time - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n\nShutting down...")

    finally:
        if datastore.is_recording:
            datastore.stop_session()
        vehicle.shutdown()


def run_pilotnet():
    print("Initializing Vehicle with PilotNet")

    FIXED_THROTTLE = 0.2

    vehicle = Vehicle()

    print("Loading PilotNet")
    model = PilotNet(input_height=84, input_width=160)
    transform = model.transform
    model_path = "models/pilotnet_2026-02-26_12-55-57.pth"
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    print("Starting vehicle...")
    vehicle.start()

    print("\nStarting control loop...")
    print("Press Ctrl+C to stop\n")

    try:
        with torch.no_grad():
            while True:
                frame = vehicle.get_frame()
                if frame is None:
                    continue
                img = transform(Image.fromarray(frame)).unsqueeze(0)

                steering = model(img).item()
                vehicle.drive(steering, FIXED_THROTTLE)

    except KeyboardInterrupt:
        print("\n\nShutting Down..")
    finally:
        vehicle.shutdown()


def main():
    """Main entry point."""
    if "drive" in sys.argv or len(sys.argv) == 1:
        drive()
    elif "pilotnet" in sys.argv:
        run_pilotnet()
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
