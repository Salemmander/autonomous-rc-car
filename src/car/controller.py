import evdev
import threading


class Controller:
    def __init__(
        self, name="Microsoft Xbox Series S|X Controller", max_throttle=0.35
    ) -> None:
        self.max_throttle = max_throttle
        devices = [evdev.InputDevice(p) for p in evdev.list_devices()]
        self.steering = 0.0
        self._forward = 0.0
        self._reverse = 0.0
        self.throttle = 0.0
        self.recording = False
        self.controller = next(device for device in devices if  name == device.name)  # fmt: skip
        thread = threading.Thread(target=self._read_events, daemon=True)
        thread.start()

    def _read_events(self):
        for event in self.controller.read_loop():
            if event.type == 1 and event.code == 304 and event.value == 1:
                self.recording = not self.recording
                continue
            if event.type != 3:
                continue
            code = event.code
            if code == 0:
                self.steering = event.value / 32768
            elif code == 2:
                self._reverse = event.value / 1023 * self.max_throttle
                self.throttle = self._forward - self._reverse
            elif code == 5:
                self._forward = event.value / 1023 * self.max_throttle
                self.throttle = self._forward - self._reverse

    def get_input(self):
        return self.steering, self.throttle, self.recording
