#include "camera.h"

#include <fstream>
#include <iostream>

int main() {
    CameraController camera(1280, 720);

    if (!camera.ok()) {
        std::cerr << "Failed to open camera\n";
        return 1;
    }

    std::cout << "camera ok: " << camera.getWidth() << "x" << camera.getHeight()
              << " (MJPEG)\n";

    // Grab a few frames so the stream settles.
    for (int i = 0; i < 5; ++i) {
        camera.capture();
    }

    if (!camera.hasFrame()) {
        std::cerr << "No frame captured\n";
        return 1;
    }

    auto frame = camera.getFrame();
    std::cout << "frame bytes: " << frame.size() << " (compressed JPEG)\n";

    if (frame.empty()) {
        std::cerr << "Unexpected empty frame\n";
        return 1;
    }

    // MJPEG buffers are valid JPEG files as-is.
    std::ofstream out("camera_test_frame.jpg", std::ios::binary);
    if (out) {
        out.write(reinterpret_cast<const char*>(frame.data()),
                  static_cast<std::streamsize>(frame.size()));
        std::cout << "wrote camera_test_frame.jpg\n";
    }

    std::cout << "camera_test ok\n";
    return 0;
}
