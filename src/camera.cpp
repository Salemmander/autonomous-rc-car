#include "camera.h"

#include <iostream>
#include <mutex>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

struct CameraController::Impl {
    cv::VideoCapture cap;
    int width{0};
    int height{0};
    std::vector<std::uint8_t> latest_frame;
    bool has_frame{false};
    mutable std::mutex frame_mutex;
};

CameraController::CameraController(int width, int height) : impl_(std::make_unique<Impl>()) {
    // CAP_V4L2: USB UVC on /dev/video0
    if (!impl_->cap.open(0, cv::CAP_V4L2)) {
        std::cerr << "camera: failed to open /dev/video0\n";
        return;
    }

    impl_->cap.set(cv::CAP_PROP_FRAME_WIDTH, width);
    impl_->cap.set(cv::CAP_PROP_FRAME_HEIGHT, height);
    impl_->cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));

    impl_->width = static_cast<int>(impl_->cap.get(cv::CAP_PROP_FRAME_WIDTH));
    impl_->height = static_cast<int>(impl_->cap.get(cv::CAP_PROP_FRAME_HEIGHT));

    if (impl_->width != width || impl_->height != height) {
        std::cerr << "camera: requested " << width << "x" << height << ", got " << impl_->width
                  << "x" << impl_->height << "\n";
    }
}

CameraController::~CameraController() {
    if (impl_ && impl_->cap.isOpened()) {
        impl_->cap.release();
    }
}

bool CameraController::ok() const {
    return impl_ && impl_->cap.isOpened();
}

int CameraController::getWidth() const {
    return impl_ ? impl_->width : 0;
}

int CameraController::getHeight() const {
    return impl_ ? impl_->height : 0;
}

bool CameraController::hasFrame() const {
    if (!impl_) {
        return false;
    }
    std::lock_guard<std::mutex> lock(impl_->frame_mutex);
    return impl_->has_frame;
}

std::vector<std::uint8_t> CameraController::getFrame() const {
    if (!impl_) {
        return {};
    }
    std::lock_guard<std::mutex> lock(impl_->frame_mutex);
    return impl_->latest_frame;
}

void CameraController::capture() {
    if (!ok()) {
        return;
    }

    cv::Mat bgr;
    if (!impl_->cap.read(bgr) || bgr.empty()) {
        std::cerr << "camera: read failed\n";
        return;
    }

    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

    std::lock_guard<std::mutex> lock(impl_->frame_mutex);
    impl_->latest_frame.assign(rgb.data, rgb.data + rgb.total() * rgb.channels());
    impl_->width = rgb.cols;
    impl_->height = rgb.rows;
    impl_->has_frame = true;
}
