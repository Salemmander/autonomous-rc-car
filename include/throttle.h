
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <linux/gpio.h>
#include <sys/ioctl.h>
#include <thread>
#include <unistd.h>

class ThrottleController {
private:
    const char* bus_path = "/dev/gpiochip0";
    gpio_v2_line_request lines;
    std::atomic<int> duty{0};
    std::atomic<bool> running{true};

    int fd{-1};
    int pwm_freq{100};
    std::thread pwm_thread;

    void pwmLoop() {
        const auto period = std::chrono::duration<double>(1.0 / pwm_freq);

        while (running) {
            int d = duty;
            if (d <= 0) {
                setEna(false);
                std::this_thread::sleep_for(period);
            } else if (d >= 100) {
                setEna(true);
                std::this_thread::sleep_for(period);
            } else {
                setEna(true);
                std::this_thread::sleep_for(period * (d / 100.0));
                setEna(false);
                std::this_thread::sleep_for(period * (1.0 - d / 100.0));
            }
        }
    }

    void setEna(bool on) {
        gpio_v2_line_values v{};
        v.mask = 0b100;
        v.bits = on ? 0b100 : 0;
        if (ioctl(lines.fd, GPIO_V2_LINE_SET_VALUES_IOCTL, &v) < 0) {
            perror("ENA: ioctl GPIO_V2_LINE_SET_VALUES_IOCTL");
            return;
        }
    }

public:
    ThrottleController() {
        memset(&lines, 0, sizeof(lines));
        lines.offsets[0] = 19;
        lines.offsets[1] = 16;
        lines.offsets[2] = 13;
        lines.num_lines = 3;
        lines.config.flags = GPIO_V2_LINE_FLAG_OUTPUT;

        // set human readable "throttle" name for consumer. (not needed for throttle control)
        strncpy(lines.consumer, "throttle", sizeof(lines.consumer) - 1);

        fd = open(bus_path, O_RDONLY);
        if (fd < 0) {
            perror("open /dev/gpiochip0");
            return;
        }
        if (ioctl(fd, GPIO_V2_GET_LINE_IOCTL, &lines) < 0) {
            perror("init: ioctl GPIO_V2_GET_LINE_IOCTL");
            close(fd);
            fd = -1;
            return;
        }
        close(fd);
        fd = -1;

        pwm_thread = std::thread([this] { pwmLoop(); });
    }

    ~ThrottleController() {
        running = false;
        if (pwm_thread.joinable())
            pwm_thread.join();

        setEna(false);

        if (lines.fd >= 0)
            close(lines.fd);
        lines.fd = -1;
    }

    bool ok() const {
        return lines.fd >= 0;
    }

    void setThrottle(float throttle) {
        if (!ok()) {
            return;
        }
        gpio_v2_line_values v{};
        v.mask = 0b011;

        throttle = std::clamp(throttle, -1.0f, 1.0f);

        if (throttle > 0.01f) {
            // Forward IN1 HIGH IN2 LOW

            v.bits = 0b001;
            duty = int(std::abs(throttle) * 100);
        } else if (throttle < -0.01) {
            // Backward IN1 LOW IN2 HIGH

            v.bits = 0b010;
            duty = int(std::abs(throttle) * 100);
        } else {
            // STOP IN1 LOW IN2 LOW

            v.bits = 0b000;
            duty = 0;
        }
        if (ioctl(lines.fd, GPIO_V2_LINE_SET_VALUES_IOCTL, &v) < 0) {
            perror("Throttle: ioctl GPIO_V2_LINE_SET_VALUES_IOCTL");
            return;
        }
    }
};
