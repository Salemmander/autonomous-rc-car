"""Read an Xbox controller with evdev, using the same mapping as include/controller.h:
left stick X = steering, right trigger = forward, left trigger = reverse.
"""

import evdev
from evdev import ecodes

DEADZONE = 0.05  # stick drift near center reads as zero


class XboxController:
    def __init__(self):
        # Find by name: /dev/input/eventN numbers differ per machine and per boot
        for path in evdev.list_devices():
            device = evdev.InputDevice(path)
            if "xbox" in device.name.lower():
                self.device = device
                break
        else:
            raise RuntimeError("No Xbox controller found. Plugged in?")

        # Each axis reports its own min/max (triggers are 0-255 on some
        # connections and 0-1023 on others), so normalize with those
        self.absinfo = {
            code: self.device.absinfo(code)
            for code in (ecodes.ABS_X, ecodes.ABS_Z, ecodes.ABS_RZ)
        }

        self.steering = 0.0  # -1 (stick left) to 1 (stick right)
        self.forward = 0.0  # 0 to 1
        self.reverse = 0.0  # 0 to 1
        self.reset_pressed = False

    def _scale(self, code: int, value: int) -> float:
        """Map a raw axis value to 0..1."""
        info = self.absinfo[code]
        return (value - info.min) / (info.max - info.min)

    def poll(self) -> None:
        """Apply every event since the last call. Never blocks."""
        while (event := self.device.read_one()) is not None:
            if event.type == ecodes.EV_ABS:
                if event.code == ecodes.ABS_X:
                    s = self._scale(event.code, event.value) * 2 - 1
                    self.steering = 0.0 if abs(s) < DEADZONE else s
                elif event.code == ecodes.ABS_RZ:
                    self.forward = self._scale(event.code, event.value)
                elif event.code == ecodes.ABS_Z:
                    self.reverse = self._scale(event.code, event.value)
            elif (
                event.type == ecodes.EV_KEY
                and event.code == ecodes.BTN_SOUTH
                and event.value == 1
            ):
                self.reset_pressed = True

    @property
    def throttle(self) -> float:
        """-1 (full reverse) to 1 (full forward)."""
        return self.forward - self.reverse


if __name__ == "__main__":
    import time

    pad = XboxController()
    print(f"Using {pad.device.name} at {pad.device.path}")
    while True:
        pad.poll()
        print(f"steering {pad.steering:+.2f}  throttle {pad.throttle:+.2f}", end="\r")
        time.sleep(0.05)
