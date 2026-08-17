#pragma once

#include "camera.h"
#include "steering.h"
#include "throttle.h"
#include <atomic>
#include <chrono>
#include <thread>
#include <vector>

class Vehicle {
private:
    SteeringController steeringctl;
    ThrottleController throttlectl;
    CameraController cameractl;

    float current_steering_angle{0.0};
    float current_throttle{0.0};

    std::atomic<bool> running{false};
    std::thread cam_thread;

public:
    Vehicle(int camera_width, int camera_height) : cameractl(camera_width, camera_height) {};
    ~Vehicle() {
        stop();
    }

    bool ok() const {
        return steeringctl.ok() && throttlectl.ok() && cameractl.ok();
    }
    void drive(float angle, float throttle) {
        if (!steeringctl.ok() || !throttlectl.ok()) {
            return;
        }
        steeringctl.setSteeringAngle(angle);
        throttlectl.setThrottle(throttle);

        current_steering_angle = angle;
        current_throttle = throttle;
    }

    std::vector<std::uint8_t> getFrame() const {
        return cameractl.getFrame();
    }

    int getWidth() const {
        return cameractl.getWidth();
    }

    int getHeight() const {
        return cameractl.getHeight();
    }

    void start() {
        if (running || cam_thread.joinable())
            return;

        running = true;
        cam_thread = std::thread([this] {
            while (running) {
                cameractl.capture();
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
        });
    }

    void stop() {
        running = false;
        drive(0.0, 0.0);

        if (cam_thread.joinable())
            cam_thread.join();
    }
};
