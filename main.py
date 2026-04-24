#!/usr/bin/env python3
"""
Autonomous RC Car - Raspberry Pi 5

Xbox controller driving with optional data collection for training.

Usage:
    uv run python main.py drive
"""

import os
import sys
import time

import cv2
import torch
from PIL import Image
from src.car import config as cfg
from src.car.datastore import DataStore
from src.car.vehicle import Vehicle
from src.car.controller import Controller
from src.training.pilotnet import PilotNet
from src.car.stream import start_stream_server


def drive():
    """Run the vehicle in manual drive mode."""
    print("Initializing vehicle...")

    vehicle = Vehicle()
    ctr = Controller()
    datastore = DataStore()

    print("Starting vehicle...")
    vehicle.start()

    print("Starting stream")
    start_stream_server(vehicle, ctr)

    time.sleep(1)  # Let threads start

    print("Starting control loop...")
    print("Press Ctrl+C to stop\n")

    loop_time = 1.0 / cfg.DRIVE_LOOP_HZ
    record_every_n = max(1, cfg.DRIVE_LOOP_HZ // cfg.RECORD_FPS)
    loop_count = 0

    try:
        while True:
            start = time.time()

            img = vehicle.get_frame()
            angle, throttle, recording = ctr.get_input()
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


def run_pilotnet(record=False):
    print("Initializing Vehicle with PilotNet")

    vehicle = Vehicle()

    print("Loading PilotNet")
    model = PilotNet(input_height=84, input_width=160)
    transform = model.transform
    model_path = "models/pilotnet_2026-04-22_16-39-13.pth"
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    writer = None
    if record:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        os.makedirs("recordings", exist_ok=True)
        writer = cv2.VideoWriter(
            "recordings/autopilot_recording.mp4", fourcc, 20.0, (1280, 720)
        )
        print("Recording enabled: recordings/autopilot_recording.mp4")

    print("Starting vehicle...")
    vehicle.start()

    print("\nStarting control loop...")
    print("Press Ctrl+C to stop\n")

    try:
        with torch.no_grad():
            loop_times = []
            last_print = time.time()
            while True:
                t0 = time.perf_counter()
                frame = vehicle.get_frame()
                if frame is None:
                    continue

                img = transform(Image.fromarray(frame)).unsqueeze(0)

                steering, throttle = model(img)
                steering = steering.item()
                throttle = throttle.item()
                if writer:
                    writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                vehicle.drive(steering, throttle)
                loop_times.append(time.perf_counter() - t0)
                if time.time() - last_print >= 5.0:
                    hz = 1.0 / (sum(loop_times) / len(loop_times))
                    print(f"Inference loop: {hz:.1f} Hz (n={len(loop_times)})")
                    loop_times.clear()
                    last_print = time.time()

    except KeyboardInterrupt:
        print("\n\nShutting Down..")
    finally:
        if writer:
            writer.release()
            print("Video saved: recordings/autopilot_recording.mp4")
        vehicle.shutdown()


def main():
    """Main entry point."""
    if "drive" in sys.argv or len(sys.argv) == 1:
        drive()
    elif "pilotnet" in sys.argv:
        record = "--record" in sys.argv
        run_pilotnet(record=record)
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
