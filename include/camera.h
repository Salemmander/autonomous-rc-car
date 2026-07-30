#pragma once

#include <cstdint>
#include <iostream>
#include <mutex>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <vector>

class CameraController {
private:
    cv::VideoCapture cap_;
    int width_{0};
    int height_{0};

    // Latest RGB888 frame (H * W * 3).
    std::vector<std::uint8_t> latest_frame_;
    bool has_frame_{false};
    mutable std::mutex frame_mutex_;

public:
    CameraController(int width, int height) {
        // CAP_V4L2: USB UVC on /dev/video0
        if (!cap_.open(0, cv::CAP_V4L2)) {
            std::cerr << "camera: failed to open /dev/video0\n";
            return;
        }

        cap_.set(cv::CAP_PROP_FRAME_WIDTH, width);
        cap_.set(cv::CAP_PROP_FRAME_HEIGHT, height);
        cap_.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));

        width_ = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_WIDTH));
        height_ = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_HEIGHT));

        if (width_ != width || height_ != height) {
            std::cerr << "camera: requested " << width << "x" << height << ", got " << width_ << "x"
                      << height_ << "\n";
        }
    }

    ~CameraController() {
        if (cap_.isOpened()) {
            cap_.release();
        }
    }

    bool ok() const {
        return cap_.isOpened();
    }

    int getWidth() const {
        return width_;
    }

    int getHeight() const {
        return height_;
    }

    bool hasFrame() const {
        std::lock_guard<std::mutex> lock(frame_mutex_);
        return has_frame_;
    }

    // Latest RGB888 frame (H*W*3), or empty if none yet.
    std::vector<std::uint8_t> getFrame() const {
        std::lock_guard<std::mutex> lock(frame_mutex_);
        return latest_frame_;
    }

    void capture() {
        if (!cap_.isOpened()) {
            return;
        }

        cv::Mat bgr;
        if (!cap_.read(bgr) || bgr.empty()) {
            std::cerr << "camera: read failed\n";
            return;
        }

        cv::Mat rgb;
        cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

        std::lock_guard<std::mutex> lock(frame_mutex_);
        if (rgb.isContinuous()) {
            latest_frame_.assign(rgb.data, rgb.data + rgb.total() * rgb.channels());
        } else {
            cv::Mat contiguous;
            rgb.copyTo(contiguous);
            latest_frame_.assign(contiguous.data,
                                 contiguous.data + contiguous.total() * contiguous.channels());
        }
        width_ = rgb.cols;
        height_ = rgb.rows;
        has_frame_ = true;
    }
};
