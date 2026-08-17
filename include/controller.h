#pragma once
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <linux/input.h>
#include <thread>
#include <unistd.h>

class XboxController {
private:
    const char* bus_path = "/dev/input/event5";
    int fd{-1};

    float forward_throttle_{};
    float reverse_throttle_{};

    float steering_{};

    std::atomic<bool> running_{false};

    std::thread ctrl_thread_;

public:
    XboxController() {
        fd = open(bus_path, O_RDONLY | O_NONBLOCK);

        if (fd < 0) {
            perror("open /dev/input/event5");
            return;
        }
    }

    ~XboxController() {
        stop();

        if (fd >= 0) {
            close(fd);
            fd = -1;
        }
    }

    void capture_controls(struct input_event& ev) {
        auto n = read(fd, &ev, sizeof(ev));
        if (n == sizeof(ev) && ev.type == EV_ABS) {
            if (ev.code == ABS_X) {

                float intermediate = ev.value / 32768.0f;
                if (std::abs(intermediate) < 0.02) {
                    intermediate = 0.0f;
                }

                steering_ = intermediate;
            }
            if (ev.code == ABS_Z) {
                reverse_throttle_ = ev.value / 255.0f;
            }
            if (ev.code == ABS_RZ) {
                forward_throttle_ = ev.value / 255.0f;
            }
        }
    }

    void get_input(float& steering, float& throttle) {
        steering = steering_;
        throttle = forward_throttle_ - reverse_throttle_;
    }

    void start() {
        if (running_ || ctrl_thread_.joinable())
            return;

        running_ = true;
        ctrl_thread_ = std::thread([this] {
            struct input_event ev;
            while (running_) {
                capture_controls(ev);
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        });
    }
    void stop() {
        running_ = false;
        if (ctrl_thread_.joinable())
            ctrl_thread_.join();
    };
};
