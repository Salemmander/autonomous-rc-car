"""
Web controller for manual driving.

Tornado-based web server that serves the control interface and handles
steering/throttle commands via POST requests.
"""

import os
import io
import time
import threading
import tornado.ioloop
import tornado.web
import tornado.gen
from tornado import httpserver
from PIL import Image
import numpy as np


def get_ip_address():
    """Get the local IP address."""
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def arr_to_binary(arr):
    """Convert numpy array to JPEG binary."""
    if arr is None:
        # Return a small black image if no frame available
        arr = np.zeros((120, 160, 3), dtype=np.uint8)

    img = Image.fromarray(arr.astype('uint8'))
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=70)
    return buf.getvalue()


class LocalWebController(tornado.web.Application):
    """
    Web-based vehicle controller.

    Serves HTML/JS interface and accepts steering/throttle commands.
    """

    port = 8887

    def __init__(self):
        print('Starting Donkey Server...')

        this_dir = os.path.dirname(os.path.realpath(__file__))
        self.static_file_path = os.path.join(this_dir, 'templates', 'static')
        self.template_path = os.path.join(this_dir, 'templates')

        # Control state
        self.angle = 0.0
        self.throttle = 0.0
        self.mode = 'user'
        self.recording = False

        # Camera frame
        self.img_arr = None

        # Network info
        self.ip_address = get_ip_address()
        self.access_url = f'https://{self.ip_address}:{self.port}'

        handlers = [
            (r"/", tornado.web.RedirectHandler, dict(url="/drive")),
            (r"/drive", DriveAPI),
            (r"/video", VideoAPI),
            (r"/static/(.*)", tornado.web.StaticFileHandler, {"path": self.static_file_path}),
        ]

        settings = {
            'debug': True,
            'template_path': self.template_path,
        }

        super().__init__(handlers, **settings)

    def say_hello(self):
        """Print connection info."""
        print(f"\n{'='*50}")
        print(f"Donkey Car Web Controller")
        print(f"{'='*50}")
        print(f"Open in browser: {self.access_url}")
        print(f"{'='*50}\n")

    def update(self):
        """Start the Tornado web server (blocking)."""
        self.port = int(self.port)

        # SSL certificate paths
        cert_dir = os.path.dirname(os.path.realpath(__file__))
        cert_file = os.path.join(cert_dir, "server.crt")
        key_file = os.path.join(cert_dir, "server.key")

        # Generate self-signed cert if not present
        if not os.path.exists(cert_file) or not os.path.exists(key_file):
            self._generate_ssl_cert(cert_file, key_file)

        server = httpserver.HTTPServer(self, ssl_options={
            "certfile": cert_file,
            "keyfile": key_file,
        })
        server.listen(self.port)

        instance = tornado.ioloop.IOLoop.instance()
        instance.add_callback(self.say_hello)
        instance.start()

    def _generate_ssl_cert(self, cert_file, key_file):
        """Generate a self-signed SSL certificate."""
        print("Generating self-signed SSL certificate...")
        import subprocess
        subprocess.run([
            "openssl", "req", "-x509", "-newkey", "rsa:2048",
            "-keyout", key_file,
            "-out", cert_file,
            "-days", "365",
            "-nodes",
            "-subj", "/CN=donkeycar"
        ], check=True, capture_output=True)
        print("SSL certificate generated.")

    def run_threaded(self, img_arr=None):
        """
        Called by vehicle loop.

        Args:
            img_arr: Camera frame to display

        Returns:
            (angle, throttle, mode, recording)
        """
        self.img_arr = img_arr
        return self.angle, self.throttle, self.mode, self.recording

    def run(self, img_arr=None):
        """Non-threaded run (calls run_threaded)."""
        return self.run_threaded(img_arr)

    def shutdown(self):
        """Stop the server."""
        tornado.ioloop.IOLoop.instance().stop()


class DriveAPI(tornado.web.RequestHandler):
    """Handle drive page requests."""

    def get(self):
        """Render the control interface."""
        self.render("vehicle.html")

    def post(self):
        """Receive control commands."""
        data = tornado.escape.json_decode(self.request.body)
        self.application.angle = float(data.get('angle', 0))
        self.application.throttle = float(data.get('throttle', 0))
        self.application.mode = data.get('drive_mode', 'user')
        self.application.recording = data.get('recording', False)


class VideoAPI(tornado.web.RequestHandler):
    """Serve MJPEG video stream."""

    @tornado.web.asynchronous
    @tornado.gen.coroutine
    def get(self):
        self.set_header("Content-type", "multipart/x-mixed-replace;boundary=--boundarydonotcross")

        self.served_image_timestamp = time.time()
        my_boundary = "--boundarydonotcross\r\n"

        while True:
            interval = 0.1  # 10 FPS for video stream

            if self.served_image_timestamp + interval < time.time():
                img = arr_to_binary(self.application.img_arr)

                self.write(my_boundary)
                self.write("Content-type: image/jpeg\r\n")
                self.write(f"Content-length: {len(img)}\r\n\r\n")
                self.write(img)
                self.served_image_timestamp = time.time()

                try:
                    yield tornado.gen.Task(self.flush)
                except tornado.iostream.StreamClosedError:
                    break
            else:
                yield tornado.gen.sleep(interval)
