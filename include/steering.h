#pragma once
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <fcntl.h>
#include <linux/i2c-dev.h>
#include <sys/ioctl.h>
#include <unistd.h>

class SteeringController {
private:
    const char* bus_path = "/dev/i2c-1";
    const int address = 0x17;
    const int left_pulse = 0;
    const int right_pulse = 170;
    uint8_t command = 0xFF;
    uint8_t channel = 1;

    int fd = -1;

public:
    SteeringController() {
        fd = open(bus_path, O_RDWR);
        if (fd < 0) {
            perror("open /dev/i2c-1");
            return;
        }

        if (ioctl(fd, I2C_SLAVE, address) < 0) {
            perror("ioctl I2C_SLAVE");
            close(fd);
            fd = -1;
            return;
        }
    }
    ~SteeringController() {
        if (fd >= 0) {
            close(fd);
            fd = -1;
        }
    }

    bool ok() const {
        return fd >= 0;
    }

    void setSteeringAngle(float angle) {
        if (fd < 0) {
            return;
        }
        angle = std::clamp(angle, -1.0f, 1.0f);

        uint8_t pulse =
            static_cast<uint8_t>(left_pulse + (angle + 1.0f) / 2.0f * (right_pulse - left_pulse));

        uint8_t buf[3] = {command, channel, pulse};
        ssize_t n = write(fd, buf, 3);
        if (n != 3) {
            perror("write");
        }
    }
};
