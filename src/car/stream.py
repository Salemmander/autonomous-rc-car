import json
import os
import time
import threading
from types import SimpleNamespace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import cv2
from src.car import config as cfg


class StreamHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        STATIC_DIR = os.path.join(cfg.CAR_PATH, "src", "static")
        CONTENT_TYPES = {
            ".html": "text/html",
            ".css": "text/css",
            ".js": "text/javascript",
        }

        if self.path == "/":
            self._serve_file(os.path.join(STATIC_DIR, "index.html"), "text/html")
        elif (
            self.path.startswith("/")
            and os.path.splitext(self.path)[1] in CONTENT_TYPES
        ):
            file_path = os.path.join(STATIC_DIR, self.path.lstrip("/"))
            ext = os.path.splitext(self.path)[1]
            self._serve_file(file_path, CONTENT_TYPES[ext])
        elif self.path == "/stream":
            self.send_response(200)
            self.send_header(
                "Content-Type", "multipart/x-mixed-replace; boundary=frame"
            )
            self.end_headers()
            try:
                while True:
                    frame = self.server.state.vehicle.get_frame()
                    if frame is None:
                        time.sleep(0.05)
                        continue
                    _, jpg = cv2.imencode(
                        ".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    )
                    data = jpg.tobytes()
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(data)}\r\n".encode())
                    self.wfile.write(b"\r\n")
                    self.wfile.write(data)
                    self.wfile.write(b"\r\n")
                    time.sleep(0.05)
            except (BrokenPipeError, ConnectionResetError):
                pass
        elif self.path == "/status":
            data = json.dumps(
                {
                    "steering": self.server.state.vehicle.current_steering,
                    "throttle": self.server.state.vehicle.current_throttle,
                    "recording": self.server.state.controller.recording,
                }
            )
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(data.encode())

    def _serve_file(self, path, content_type):
        if not os.path.isfile(path):
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.end_headers()
        with open(path, "rb") as f:
            self.wfile.write(f.read())

    def log_message(self, format: str, *args) -> None:
        pass


def start_stream_server(vehicle, controller, port=cfg.STREAM_PORT):
    server = ThreadingHTTPServer(("0.0.0.0", port), StreamHandler)
    server.state = SimpleNamespace(vehicle=vehicle, controller=controller)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(f"Dashboard : http://0.0.0.0:{port}/")
