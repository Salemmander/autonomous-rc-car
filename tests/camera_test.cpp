#include "camera.h"

#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

int main() {
    CameraController camera(1280, 720);

    if (!camera.ok()) {
        std::cerr << "Failed to open camera\n";
        return 1;
    }

    std::cout << "camera ok: " << camera.getWidth() << "x" << camera.getHeight() << "\n";

    for (int i = 0; i < 5; ++i) {
        camera.capture();
    }

    if (!camera.hasFrame()) {
        std::cerr << "No frame captured\n";
        return 1;
    }

    auto frame = camera.getFrame();
    const std::size_t expected =
        static_cast<std::size_t>(camera.getWidth()) * camera.getHeight() * 3;
    std::cout << "frame bytes: " << frame.size() << " (expected RGB " << expected << ")\n";

    if (frame.size() != expected) {
        std::cerr << "Unexpected RGB frame size\n";
        return 1;
    }

    // OpenCV expects BGR for imwrite color images.
    cv::Mat rgb(camera.getHeight(), camera.getWidth(), CV_8UC3, frame.data());
    cv::Mat bgr;
    cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);
    if (!cv::imwrite("camera_test_frame.png", bgr)) {
        std::cerr << "Failed to write camera_test_frame.png\n";
        return 1;
    }
    std::cout << "wrote camera_test_frame.png\n";
    std::cout << "camera_test ok\n";
    return 0;
}
