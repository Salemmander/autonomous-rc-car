#pragma once

#include <cstdint>
#include <memory>
#include <vector>

// OpenCV stays in camera.cpp so includers don't pull it in.
class CameraController {
public:
    CameraController(int width, int height);
    ~CameraController();

    CameraController(const CameraController&) = delete;
    CameraController& operator=(const CameraController&) = delete;

    bool ok() const;
    int getWidth() const;
    int getHeight() const;
    bool hasFrame() const;

    // Last successfully captured RGB888 frame (H*W*3), or empty if none yet.
    // Failed/corrupt reads do not clear this — callers keep the last good image.
    std::vector<std::uint8_t> getFrame() const;

    // Try to grab a frame; only replaces last good on success.
    void capture();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
