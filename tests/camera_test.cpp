#include "camera.h"

#include <fstream>
#include <iostream>

int main() {
    CameraController camera(1280, 720);

    if (!camera.ok()) {
        std::cerr << "Failed to open camera\n";
        return 1;
    }

    std::cout << "camera ok: " << camera.getWidth() << "x" << camera.getHeight() << "\n";

    // Grab a few frames so the stream settles.
    for (int i = 0; i < 5; ++i) {
        camera.capture();
    }

    if (!camera.hasFrame()) {
        std::cerr << "No frame captured\n";
        return 1;
    }

    auto frame = camera.getFrame();
    std::cout << "frame bytes: " << frame.size() << "\n";

    const std::size_t expected_yuyv =
        static_cast<std::size_t>(camera.getWidth()) * camera.getHeight() * 2;
    if (frame.size() != expected_yuyv && frame.size() == 0) {
        std::cerr << "Unexpected empty frame\n";
        return 1;
    }
    std::cout << "expected ~" << expected_yuyv << " bytes for YUYV "
              << camera.getWidth() << "x" << camera.getHeight() << "\n";

    // Optional: dump raw YUYV for inspection (e.g. convert with ffmpeg later).
    std::ofstream out("camera_test_frame.yuyv", std::ios::binary);
    if (out) {
        out.write(reinterpret_cast<const char*>(frame.data()),
                  static_cast<std::streamsize>(frame.size()));
        std::cout << "wrote camera_test_frame.yuyv\n";
    }

    std::cout << "camera_test ok\n";
    return 0;
}
