"""
Pi 5 Camera module using picamera2.

Replaces the proprietary XRCamera .so binary.

Install: sudo apt install python3-picamera2 python3-libcamera
"""

import numpy as np
import threading
import time


class PiCamera:
    """
    Raspberry Pi 5 camera using picamera2 library.

    Runs in a separate thread to avoid blocking the main loop.
    """

    def __init__(self, resolution=(120, 160), framerate=20):
        """
        Initialize camera.

        Args:
            resolution: (height, width) tuple
            framerate: Target framerate
        """
        self.resolution = resolution
        self.framerate = framerate
        self.frame = None
        self.running = False
        self._lock = threading.Lock()
        self._camera = None

    def _init_camera(self):
        """Initialize picamera2 (works with Pi camera or USB webcam)."""
        from picamera2 import Picamera2

        self._camera = Picamera2()

        # Configure for video capture
        # Note: Don't set FrameRate control - USB cameras don't support it
        height, width = self.resolution
        config = self._camera.create_video_configuration(
            main={"size": (width, height), "format": "RGB888"}
        )
        self._camera.configure(config)
        self._camera.start()

        # Allow camera to warm up
        time.sleep(0.5)

    def update(self):
        """
        Threaded update loop - continuously captures frames.
        Called by Vehicle when running in threaded mode.
        """
        self._init_camera()
        self.running = True

        while self.running:
            try:
                # Capture frame as numpy array
                frame = self._camera.capture_array()

                with self._lock:
                    self.frame = frame

            except Exception as e:
                print(f"Camera error: {e}")
                time.sleep(0.1)

    def run_threaded(self):
        """
        Return the latest frame.
        Called by Vehicle on each loop iteration.
        """
        with self._lock:
            return self.frame

    def run(self):
        """
        Capture and return a single frame (non-threaded mode).
        """
        if self._camera is None:
            self._init_camera()

        return self._camera.capture_array()

    def shutdown(self):
        """Stop camera."""
        self.running = False
        if self._camera is not None:
            self._camera.stop()
            self._camera.close()
            self._camera = None


class MockCamera:
    """
    Mock camera for testing without hardware.
    Returns random colored frames.
    """

    def __init__(self, resolution=(120, 160), framerate=20):
        self.resolution = resolution
        self.framerate = framerate
        self.frame = None
        self.running = False

    def update(self):
        """Threaded update - generate fake frames."""
        self.running = True
        height, width = self.resolution

        while self.running:
            # Generate a random colored frame
            self.frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            time.sleep(1.0 / self.framerate)

    def run_threaded(self):
        return self.frame

    def run(self):
        height, width = self.resolution
        return np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)

    def shutdown(self):
        self.running = False


# Test function
if __name__ == "__main__":
    import sys

    if "--mock" in sys.argv:
        print("Testing mock camera...")
        cam = MockCamera()
    else:
        print("Testing Pi camera...")
        cam = PiCamera()

    print(f"Resolution: {cam.resolution}")

    # Test single capture
    frame = cam.run()
    print(f"Captured frame shape: {frame.shape}")

    cam.shutdown()
    print("Done!")
