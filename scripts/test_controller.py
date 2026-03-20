"""Test Xbox controller input."""

import time

from src.car.controller import Controller

c = Controller()
print("Move sticks and triggers, press A to toggle recording, Ctrl+C to stop\n")

while True:
    s, t = c.get_input()
    print(
        f"Steering: {s:+.2f}  Throttle: {t:+.2f}  Recording: {str(c.recording):<5}",
        end="\r",
    )
    time.sleep(0.05)
