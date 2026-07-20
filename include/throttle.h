
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <linux/gpio.h>
#include <sys/ioctl.h>
#include <unistd.h>

class ThrottleController {
private:
    const char* bus_path = "/dev/gpiochip0";
    gpio_v2_line_request lines;
    int fd = -1;

public:
    ThrottleController() {
        memset(&lines, 0, sizeof(lines));
        lines.offsets[0] = 19;
        lines.offsets[1] = 16;
        lines.offsets[2] = 13;
        lines.num_lines = 3;
        lines.config.flags = GPIO_V2_LINE_FLAG_OUTPUT;
        strncpy(lines.consumer, "throttle", sizeof(lines.consumer) - 1);

        fd = open(bus_path, O_RDONLY);
        if (fd < 0) {
            perror("open /dev/gpiochip0");
            return;
        }
        if (ioctl(fd, GPIO_V2_GET_LINE_IOCTL, &lines) < 0) {
            perror("ioctl GPIO_V2_GET_LINE_IOCTL");
            close(fd);
            fd = -1;
            return;
        }
    }
};
