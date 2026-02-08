"""
Data collection for recording driving sessions.

Records (image, steering, throttle) tuples to disk for training
a self-driving model via behavioral cloning.
"""

import csv
import os
import queue
import threading
import time
from datetime import datetime

from PIL import Image

import config as cfg


class DataStore:
    """Manages recording sessions with a background writer thread."""

    def __init__(self):
        self._session_dir = None
        self._csv_file = None
        self._csv_writer = None
        self._frame_count = 0
        self._session_start = None
        self._queue = queue.Queue(maxsize=100)
        self._writer_thread = None
        self._running = False

    @property
    def is_recording(self):
        return self._running

    def start_session(self):
        """Create a new session directory and start the writer thread."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._session_dir = os.path.join(cfg.DATA_PATH, f"session_{timestamp}")
        os.makedirs(self._session_dir, exist_ok=True)

        csv_path = os.path.join(self._session_dir, "log.csv")
        self._csv_file = open(csv_path, "w", newline="")
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow(["frame", "timestamp", "steering", "throttle"])

        self._frame_count = 0
        self._session_start = time.time()

        self._running = True
        self._writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._writer_thread.start()

        print(f"Recording started: {self._session_dir}")

    def record(self, img, steering, throttle):
        """Enqueue a frame for background writing. Drops frames if queue is full."""
        if not self._running or img is None:
            return
        timestamp = time.time()
        try:
            self._queue.put_nowait((img.copy(), steering, throttle, timestamp))
        except queue.Full:
            pass

    def stop_session(self):
        """Stop recording, drain remaining frames, close files."""
        if not self._running:
            return

        self._running = False

        if self._writer_thread is not None:
            self._writer_thread.join(timeout=5.0)

        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None
            self._csv_writer = None

        duration = time.time() - self._session_start if self._session_start else 0
        print(f"Recording stopped: {self._frame_count} frames, {duration:.1f}s")
        print(f"  Saved to: {self._session_dir}")

        self._session_dir = None
        self._session_start = None

    def _writer_loop(self):
        """Background thread: dequeues frames, saves JPEGs, appends CSV rows."""
        while self._running or not self._queue.empty():
            try:
                img, steering, throttle, timestamp = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

            self._frame_count += 1
            frame_name = f"frame_{self._frame_count:05d}.jpg"
            frame_path = os.path.join(self._session_dir, frame_name)

            Image.fromarray(img).save(
                frame_path, format="JPEG", quality=cfg.JPEG_QUALITY
            )

            self._csv_writer.writerow(
                [
                    frame_name,
                    f"{timestamp:.3f}",
                    f"{steering:.4f}",
                    f"{throttle:.4f}",
                ]
            )
            self._csv_file.flush()

            self._queue.task_done()
